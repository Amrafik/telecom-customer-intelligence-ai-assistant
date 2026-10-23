import json
import re
import threading
import time
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from telecom.config import Settings
from telecom.conversation import offline_route, scope_reply
from telecom.errors import AppError
from telecom.gemini_llm import GeminiLLM
from telecom.ml import CustomerStore, Predictor
from telecom.monitoring import event, request_id_var
from telecom.rag import UNCERTAIN, Retriever
from telecom.schemas import ChatRequest, ChatResponse, Source
from telecom.tools import ToolContext, ToolExecutor, tool_spec

RISK_WORDS = r"\b(churn|risk|predict|prediction|leaving|likelihood)\b"


def resolve_customer(request: ChatRequest) -> tuple[str | None, str | None]:
    ids = {match.upper() for match in re.findall(r"\bCUST-[A-Za-z0-9]+\b", request.message, re.I)}
    if any(not re.fullmatch(r"CUST-\d{3}", item) for item in ids):
        return None, "Use a fictional customer ID in the format CUST-001."
    if len(ids) > 1 or (ids and request.customer_id and request.customer_id not in ids):
        return None, "Please ask about one customer at a time and use the same ID as the selected customer."
    return next(iter(ids), request.customer_id), None


class State(TypedDict, total=False):
    question: str
    customer_id: str | None
    route: str
    intent: str
    mode: str
    previous_intent: str | None
    clarification: str
    context: ToolContext
    customer_answer: str
    knowledge_answer: str
    customer_sources: list[Source]
    knowledge_sources: list[Source]
    answer: str
    sources: list[Source]


class Assistant:
    def __init__(self, settings: Settings | None = None, *, llm=None, retriever=None, predictor=None):
        self.settings = settings or Settings()
        self.customers = CustomerStore()
        self._llm, self._retriever, self._predictor = llm, retriever, predictor
        self._resource_lock = threading.RLock()
        self.tools = ToolExecutor(self.customers, self.predictor, self.retriever)
        graph = StateGraph(State)
        graph.add_node("router", self._route)
        graph.add_node("customer_agent", self._customer_agent)
        graph.add_node("knowledge_agent", self._knowledge_agent)
        graph.add_node("clarify", self._clarify)
        graph.add_node("combine", self._combine)
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", lambda state: state["route"],
                                    {"customer": "customer_agent", "knowledge": "knowledge_agent",
                                     "both": "customer_agent", "clarify": "clarify"})
        graph.add_conditional_edges("customer_agent", lambda state: "knowledge" if state["route"] == "both" else "combine",
                                    {"knowledge": "knowledge_agent", "combine": "combine"})
        graph.add_edge("knowledge_agent", "combine")
        graph.add_edge("clarify", "combine")
        graph.add_edge("combine", END)
        self.graph = graph.compile()

    def llm(self):
        with self._resource_lock:
            if self._llm is None:
                self._llm = GeminiLLM(self.settings)
        return self._llm

    def predictor(self):
        with self._resource_lock:
            if self._predictor is None:
                self._predictor = Predictor()
        return self._predictor

    def retriever(self):
        with self._resource_lock:
            if self._retriever is None:
                self._retriever = Retriever(self.settings)
        return self._retriever

    def _route(self, state):
        if state.get("clarification"):
            intent = "clarify"
        else:
            intent = (offline_route(state["question"], state.get("previous_intent")) if self.settings.app_mode == "offline"
                      else self.llm().route(state["question"], state["customer_id"], state.get("previous_intent")))
        route = intent if intent in {"customer", "knowledge", "both"} else "clarify"
        if route in {"customer", "both"} and not state["customer_id"]:
            event("agent_selection", route="clarify", intent=intent, mode=self.settings.app_mode, provider="gemini")
            return {"route": "clarify", "intent": intent, "clarification": "Please provide a fictional customer ID, for example CUST-001."}
        event("agent_selection", route=route, intent=intent, mode=self.settings.app_mode, provider="gemini")
        return {"route": route, "intent": intent}

    def _run_tools(self, agent, state):
        results, history = {}, []
        allowed = ["search_faqs"] if agent == "knowledge" else ["get_customer", "predict_churn"]
        for step in range(1 if agent == "knowledge" else 3):
            if self.settings.app_mode == "offline":
                if agent == "knowledge":
                    name, args = "search_faqs", {"query": state["question"], "top_k": 3}
                elif step == 0:
                    name, args = "get_customer", {"customer_id": state["customer_id"]}
                elif step == 1 and re.search(RISK_WORDS, state["question"], re.I):
                    name, args = "predict_churn", {"customer_id": state["customer_id"]}
                else:
                    break
            else:
                call, output = self.llm().choose_tool(
                    agent, state["question"], state["customer_id"], history,
                    [tool_spec(name) for name in allowed], state.get("customer_answer", "")
                )
                if call is None:
                    break
                name, args = call.name, call.arguments
                history.extend(output)
            result = self.tools.execute(agent, name, args, state["context"])
            results[name] = result
            if self.settings.app_mode == "live":
                history.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps(result)})
            if not result["ok"] or name == "predict_churn":
                break
        return results

    def _customer_agent(self, state):
        try:
            results = self._run_tools("customer", state)
        except AppError as exc:
            state["context"].errors.append(exc.code)
            return {"customer_answer": str(exc), "customer_sources": []}
        failures = [r["error"]["message"] for r in results.values() if not r["ok"]]
        profile = results.get("get_customer", {}).get("data")
        prediction = results.get("predict_churn", {}).get("data")
        sources, parts = [], []
        if profile:
            parts.append(f"{profile['customer_id']} is {profile['display_name']} (fictional). "
                         f"Subscription: {profile['subscription_months']} months; tariff: {profile['tariff_plan']}; "
                         f"complaint recorded: {profile['complaints']}; failed calls: {profile['call_failures']}.")
            sources.append(Source(source_id=f"customer:{profile['customer_id']}", title="Fictional customer record",
                                  reference=f"sqlite:customers+usage_metrics#{profile['customer_id']}", excerpt=json.dumps(profile)))
        if prediction:
            parts.append(f"Estimated churn probability: {prediction['churn_probability']:.1%}. "
                         f"Classification at threshold 0.50: {'churn' if prediction['predicted_churn'] else 'retained'}. "
                         f"{prediction['caveat']} This score does not establish a cause of churn.")
            sources.append(Source(source_id=f"model:{prediction['model_version']}", title="Local churn model result",
                                  reference="reports/benchmark.json", excerpt=json.dumps(prediction)))
        parts.extend(failures)
        if not parts:
            state["context"].errors.append("no_customer_result")
            parts.append("The agent did not obtain customer facts. Please try a specific profile or churn question.")
        return {"customer_answer": "\n\n".join(parts), "customer_sources": sources}

    def _knowledge_agent(self, state):
        try:
            results = self._run_tools("knowledge", state)
            result = results.get("search_faqs")
            if not result:
                raise AppError("The agent did not retrieve FAQ evidence.", "no_retrieval_result")
            if not result["ok"]:
                return {"knowledge_answer": result["error"]["message"], "knowledge_sources": []}
            evidence = [Source.model_validate(item) for item in result["data"]]
            if self.settings.app_mode == "live":
                answer, sources = self.llm().grounded_answer(state["question"], evidence)
            elif evidence:
                # Verbatim retrieved chunks, explicitly labelled; no generated policy facts.
                answer = "Offline mode — retrieved FAQ evidence (check whether it answers your question):\n\n" + "\n\n".join(
                    f"{source.excerpt} [{source.source_id}]" for source in evidence)
                sources = evidence
            else:
                answer, sources = UNCERTAIN, []
            return {"knowledge_answer": answer, "knowledge_sources": sources}
        except AppError as exc:
            state["context"].errors.append(exc.code)
            return {"knowledge_answer": str(exc), "knowledge_sources": []}

    @staticmethod
    def _clarify(state):
        return {"answer": state.get("clarification") or scope_reply(state["intent"], state["mode"])}

    @staticmethod
    def _combine(state):
        parts = []
        if state.get("customer_answer"):
            parts.append("Customer intelligence:\n" + state["customer_answer"])
        if state.get("knowledge_answer"):
            parts.append("FAQ guidance:\n" + state["knowledge_answer"])
        return {"answer": "\n\n".join(parts) or state.get("answer", "Please clarify your question."),
                "sources": state.get("customer_sources", []) + state.get("knowledge_sources", [])}

    def chat(self, request: ChatRequest, request_id: str | None = None) -> ChatResponse:
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        customer_id, clarification = resolve_customer(request)
        context = ToolContext(customer_id, self.settings.max_tool_calls)
        try:
            result = self.graph.invoke({"question": request.message, "customer_id": customer_id,
                                        "previous_intent": request.previous_intent, "mode": self.settings.app_mode,
                                        "clarification": clarification or "", "context": context},
                                       config={"recursion_limit": 12})
            event("chat_completed", route=result["route"], intent=result["intent"], mode=self.settings.app_mode, provider="gemini",
                  status="error" if context.errors else "success", error_codes=context.errors,
                  latency_ms=round((time.perf_counter() - started) * 1000, 3))
        except Exception as exc:
            event("chat_failed", mode=self.settings.app_mode, provider="gemini", status="error",
                  error_code=exc.code if isinstance(exc, AppError) else "internal_error",
                  error_type=type(exc).__name__, latency_ms=round((time.perf_counter() - started) * 1000, 3))
            raise
        finally:
            request_id_var.reset(token)
        return ChatResponse(answer=result["answer"], route=result["route"], mode=self.settings.app_mode,
                            intent=result["intent"],
                            sources=result["sources"], tool_trace=context.trace, errors=context.errors,
                            request_id=request_id,
                            latency_ms=round((time.perf_counter() - started) * 1000, 3))
