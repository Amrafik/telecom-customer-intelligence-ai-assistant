import json
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import Field, ValidationError

from telecom.errors import AppError
from telecom.ml import CustomerStore, Predictor
from telecom.monitoring import event
from telecom.rag import Retriever
from telecom.schemas import CustomerInput, StrictModel, ToolTrace


class SearchInput(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(ge=1, le=5, strict=True)


SCHEMAS = {"search_faqs": SearchInput, "get_customer": CustomerInput, "predict_churn": CustomerInput}
DESCRIPTIONS = {
    "search_faqs": "Search fictional telecom policies; returns scored source excerpts. Never guess policy facts.",
    "get_customer": "Look up a fictional customer profile by the exact customer ID provided by the user.",
    "predict_churn": "Predict a fictional customer's churn probability. Requires a successful get_customer call first."
}
ALLOWED = {"knowledge": {"search_faqs"}, "customer": {"get_customer", "predict_churn"}}


def tool_spec(name: str) -> dict:
    # Mirror the executor's required fields and extra-field rejection in the declaration.
    schema = SCHEMAS[name].model_json_schema()
    schema["required"] = list(schema["properties"])
    return {"name": name, "description": DESCRIPTIONS[name], "parameters": schema}


@dataclass
class ToolContext:
    customer_id: str | None
    max_calls: int = 5
    attempts: int = 0
    seen: set = field(default_factory=set)
    looked_up: set = field(default_factory=set)
    trace: list[ToolTrace] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ToolExecutor:
    def __init__(self, customers: CustomerStore, predictor: Callable[[], Predictor], retriever: Callable[[], Retriever]):
        self.customers, self.predictor, self.retriever = customers, predictor, retriever

    def execute(self, agent: str, name: str, arguments: dict | str, context: ToolContext) -> dict:
        started = time.perf_counter()
        try:
            if context.attempts >= context.max_calls:
                raise AppError("The request reached its tool-call limit.", "tool_budget_exceeded", 400)
            context.attempts += 1  # Invalid attempts count too.
            if name not in ALLOWED.get(agent, set()):
                raise AppError("This tool is not allowed for this agent.", "tool_not_allowed", 400)
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            validated = SCHEMAS[name].model_validate(arguments)
            signature = (name, validated.model_dump_json())
            if signature in context.seen:
                raise AppError("Repeated identical tool call was blocked.", "repeated_tool_call", 400)
            context.seen.add(signature)
            if name == "search_faqs":
                data = [source.model_dump() for source in self.retriever().search(validated.query, validated.top_k)]
            else:
                if validated.customer_id != context.customer_id:
                    raise AppError("Tool customer ID must match the user's selected customer.", "customer_id_mismatch", 400)
                customer = self.customers.get(validated.customer_id)
                if name == "get_customer":
                    context.looked_up.add(customer.customer_id)
                    data = customer.model_dump()
                else:
                    if customer.customer_id not in context.looked_up:
                        raise AppError("Look up the customer before predicting churn.", "lookup_required", 400)
                    data = self.predictor().predict(customer).model_dump()
            context.trace.append(ToolTrace(agent=agent, tool=name, status="success"))
            return {"ok": True, "data": data}
        except (ValidationError, json.JSONDecodeError, TypeError):
            return self._failure(context, agent, name, "invalid_tool_input", "Tool inputs failed validation.")
        except AppError as exc:
            return self._failure(context, agent, name, exc.code, str(exc))
        except Exception:
            return self._failure(context, agent, name, "tool_execution_error", "The local tool could not complete the request.")
        finally:
            trace = context.trace[-1]
            event("tool_execution", agent=agent, tool=name if name in SCHEMAS else "unregistered",
                  status=trace.status, error_code=trace.code,
                  latency_ms=round((time.perf_counter() - started) * 1000, 3))

    @staticmethod
    def _failure(context, agent, name, code, message):
        context.trace.append(ToolTrace(agent=agent, tool=name, status="error", code=code))
        context.errors.append(code)
        return {"ok": False, "error": {"code": code, "message": message}}
