"""Opt-in live Gemini checks; pytest imports only its pure evidence-check helper."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from telecom.api import create_app
from telecom.config import ROOT, Settings
from telecom.conversation import scope_reply
from telecom.gemini_llm import GeminiLLM
from telecom.rag import UNCERTAIN
from telecom.workflow import Assistant

CASES = [
    ("faq", "How do I activate international roaming?", "knowledge", ["search_faqs"], "roaming"),
    ("customer_lookup", "Show the profile for CUST-002.", "customer", ["get_customer"], None),
    ("churn_prediction", "Predict churn for CUST-001.", "customer", ["get_customer", "predict_churn"], None),
    ("combined", "Predict churn for CUST-001 and explain the cancellation policy.", "both", ["get_customer", "predict_churn", "search_faqs"], "cancellation"),
    ("invalid_customer", "Predict churn for CUST-999.", "customer", ["get_customer"], None),
    ("unsupported_price", "What is the exact DemoTel roaming price in Japan?", "knowledge", ["search_faqs"], None),
    ("malformed_customer", "Predict churn for CUST-1.", "clarify", [], None),
    ("scope_unrelated", "Write a chocolate cake recipe.", "clarify", [], None),
    ("scope_followup", "Why not?", "clarify", [], None),
    ("scope_capabilities", "What can you do?", "clarify", [], None),
    ("scope_architecture", "How do you work?", "clarify", [], None),
    ("scope_model", "what model do u use", "clarify", [], None),
    ("scope_gemini_model", "no like what gemini model are u", "clarify", [], None),
]
SCOPE_INTENTS = {"scope_unrelated": "out_of_scope", "scope_followup": "refusal_followup",
                 "scope_capabilities": "capabilities", "scope_architecture": "architecture",
                 "scope_model": "model_info", "scope_gemini_model": "model_info"}


def evidence_checks(body: dict, executed: list[dict], assistant: Assistant, case_id: str, faq_id: str | None) -> dict:
    """Compare returned evidence to actual tool outputs, the SQL row and pipeline calculation."""
    checks = {}
    retrieved = {row["source_id"]: row for call in executed if call["tool"] == "search_faqs" and call["result"]["ok"]
                 for row in call["result"]["data"]}
    faq_sources = [source for source in body["sources"] if source["source_id"].startswith("faq:")]
    checks["citations_match_executed_retrieval"] = all(source == retrieved.get(source["source_id"]) for source in faq_sources)
    if faq_id:
        checks["expected_policy_cited"] = any(source["source_id"].startswith(f"faq:{faq_id}:") for source in faq_sources)
        answer = body["answer"].lower()
        checks["expected_policy_fact_present"] = (("roaming" in answer and "app" in answer) if faq_id == "roaming"
                                                  else "cancel" in answer and ("support" in answer or "store" in answer))
    if case_id == "unsupported_price":
        checks["unsupported_price_returns_uncertainty"] = UNCERTAIN in body["answer"] and not faq_sources
    for call in executed:
        if not call["result"]["ok"]:
            continue
        data = call["result"]["data"]
        if call["tool"] == "get_customer":
            expected = assistant.customers.get(data["customer_id"]).model_dump()
            actual = [json.loads(source["excerpt"]) for source in body["sources"] if source["source_id"] == f"customer:{data['customer_id']}"]
            checks["customer_source_equals_sql_join"] = data == expected and actual == [expected]
        if call["tool"] == "predict_churn":
            expected = assistant.predictor().predict(assistant.customers.get(data["customer_id"])).model_dump()
            actual = [json.loads(source["excerpt"]) for source in body["sources"] if source["source_id"].startswith("model:")]
            checks["score_equals_actual_pipeline"] = data == expected and actual == [expected]
            checks["answer_displays_actual_score"] = f"{expected['churn_probability']:.1%}" in body["answer"]
    return checks


def evaluate_gemini(settings: Settings | None = None, *, spacing_seconds: float = 15,
                    case_ids: list[str] | None = None, output_path: Path | None = None):
    settings = settings or Settings(app_mode="live")
    if settings.app_mode != "live":
        raise ValueError("This command requires live Gemini settings.")
    if case_ids and not set(case_ids) <= {case[0] for case in CASES}:
        raise ValueError("Unknown live evaluation case.")
    if case_ids and "scope_followup" in case_ids and "scope_unrelated" not in case_ids:
        raise ValueError("The follow-up live check requires scope_unrelated first.")
    cases = [case for case in CASES if (case[0] not in SCOPE_INTENTS if not case_ids else case[0] in case_ids)]
    assistant = Assistant(settings)
    executed = []
    original_execute = assistant.tools.execute

    def observed_execute(agent, name, arguments, context):
        result = original_execute(agent, name, arguments, context)
        executed.append({"agent": agent, "tool": name, "result": result})
        return result

    assistant.tools.execute = observed_execute
    results, availability, generation_calls = [], {}, []
    previous_body = None
    if settings.gemini_api_key:
        try:
            llm = assistant.llm()
            original_generate = llm._generate
            def observed_generate(contents, config):
                generation_calls.append(1)
                return original_generate(contents, config)
            llm._generate = observed_generate
            model = llm.client.models.get(model=settings.gemini_model)
            availability = {"model": model.name, "supported_actions": model.supported_actions}
        except Exception as exc:
            availability = {"check": "unavailable", "error_type": type(exc).__name__}
    else:
        availability = {"check": "missing_api_key"}
    try:
        with TestClient(create_app(assistant=assistant), raise_server_exceptions=False) as client:
            for number, (case_id, question, route, tools, faq_id) in enumerate(cases):
                executed.clear()
                calls_before = len(generation_calls)
                started = time.perf_counter()
                request = {"message": question}
                if case_id == "scope_followup" and previous_body and previous_body.get("intent") == "out_of_scope":
                    request["previous_intent"] = previous_body["intent"]
                response = client.post("/chat", json=request)
                body = response.json()
                checks = {"http_200": response.status_code == 200}
                if response.status_code == 200:
                    checks.update({"expected_route": body["route"] == route,
                                   "actual_tools_in_order": [call["tool"] for call in executed] == tools,
                                   "trace_matches_execution": [trace["tool"] for trace in body["tool_trace"]] == tools})
                    if case_id in SCOPE_INTENTS:
                        intent = SCOPE_INTENTS[case_id]
                        checks.update(expected_intent=body["intent"] == intent,
                                      bounded_truthful_response=body["answer"] == scope_reply(
                                          intent, "live", gemini_model=settings.gemini_model,
                                          embedding_model=settings.embedding_model),
                                      no_fabricated_sources=not body["sources"],
                                      one_actual_classification=len(generation_calls) - calls_before == 1)
                        if case_id == "scope_followup":
                            checks["context_from_actual_prior_refusal"] = request.get("previous_intent") == "out_of_scope"
                    if case_id == "invalid_customer":
                        checks["explicit_missing_customer"] = body["errors"] == ["customer_not_found"]
                        checks["no_fabricated_sources"] = not body["sources"]
                        checks["lookup_failed_before_prediction"] = bool(executed) and not executed[0]["result"]["ok"]
                    else:
                        checks["no_agent_errors"] = not body["errors"]
                        checks["tools_succeeded"] = all(call["result"]["ok"] for call in executed)
                        checks.update(evidence_checks(body, executed, assistant, case_id, faq_id))
                results.append({"id": case_id, "passed": all(checks.values()), "checks": checks,
                                "http_status": response.status_code, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                                "request": request, "generate_content_calls": len(generation_calls) - calls_before,
                                "response": body, "executed_tools": list(executed)})
                previous_body = body
                print(f"{case_id}: {'PASS' if results[-1]['passed'] else 'FAIL'}; HTTP {response.status_code}; tools={[call['tool'] for call in executed]}", flush=True)
                # Bounded spacing helps small free-tier quotas; no automatic repeated requests.
                if number < len(cases) - 1 and spacing_seconds:
                    time.sleep(spacing_seconds)
    finally:
        if isinstance(assistant._llm, GeminiLLM):
            assistant._llm.client.close()
    report = {"provider": "gemini", "model": settings.gemini_model, "mode": "live",
              "run_at": datetime.now(ZoneInfo("Africa/Cairo")).isoformat(), "model_availability": availability,
              "passed": sum(case["passed"] for case in results), "total": len(cases), "cases": results,
              "generate_content_calls": len(generation_calls),
              "scope": "Actual Gemini requests, unchanged LangGraph, FastAPI and registry; evidence compared to executed FAISS search, SQLite JOIN and sklearn inference.",
              "limitations": "Small smoke set; source identity and selected policy facts are checked, but this is not an exhaustive semantic entailment or security evaluation. Free-tier access and quotas are account-specific."}
    (ROOT / "reports").mkdir(exist_ok=True)
    filename = "gemini_live_audit.json" if case_ids else "gemini_live_evaluation.json"
    destination = output_path or ROOT / "reports" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Live Gemini: {report['passed']}/{report['total']} passed; report saved.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opt-in live Gemini verification (uses API quota).")
    parser.add_argument("--case", dest="case_ids", action="append", choices=[case[0] for case in CASES],
                        help="Run only the named case; repeat for a minimal quota-conscious audit.")
    parser.add_argument("--output", type=Path, help="Save a separate report without replacing earlier live evidence.")
    arguments = parser.parse_args()
    result = evaluate_gemini(case_ids=arguments.case_ids, output_path=arguments.output)
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
