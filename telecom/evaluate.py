"""Small transparent checks; offline checks do not measure LLM answer quality."""
import argparse
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from telecom.api import create_app
from telecom.config import ROOT, Settings
from telecom.schemas import ChatRequest
from telecom.tools import ToolContext
from telecom.workflow import Assistant


def check_case(case, assistant, client):
    kind = case["kind"]
    if kind == "retrieval":
        sources = assistant.retriever().search(case["question"])
        assert any(source.source_id.startswith(case["expected"]) for source in sources)
        return {"source_ids": [source.source_id for source in sources]}
    if kind == "retrieval_miss":
        assert assistant.retriever().search(case["question"]) == []
    elif kind == "route":
        assert assistant.chat(ChatRequest(message=case["question"])).route == case["expected"]
    elif kind == "lookup":
        row = assistant.customers.get(case["customer_id"])
        assert row.fictional and row.calls == case["expected_calls"] and row.display_name == "Demo Customer A"
    elif kind == "unknown_customer":
        assert client.get(f"/customers/{case['customer_id']}").status_code == 404
    elif kind == "prediction":
        payload = {"customer_id": case["customer_id"]}
        first, second = client.post("/predict", json=payload), client.post("/predict", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json() and 0 <= first.json()["churn_probability"] <= 1
        return first.json()
    elif kind == "combined":
        result = assistant.chat(ChatRequest(message=case["question"]))
        assert result.route == "both" and not result.errors
        assert [trace.tool for trace in result.tool_trace] == ["get_customer", "predict_churn", "search_faqs"]
        assert any(source.source_id.startswith("faq:cancellation:") for source in result.sources)
    elif kind in {"clarification", "conflicting_ids"}:
        result = assistant.chat(ChatRequest(message=case["question"], customer_id="CUST-001" if kind == "conflicting_ids" else None))
        assert result.route == "clarify" and not result.tool_trace
    elif kind == "bad_request":
        assert client.post("/chat", json=case["body"]).status_code == 422
    elif kind == "bad_predict":
        assert client.post("/predict", json=case["body"]).status_code == 422
    else:
        context = ToolContext("CUST-001", max_calls=1 if kind == "tool_budget" else 5)
        execute = lambda agent, tool, customer_id: assistant.tools.execute(agent, tool, {"customer_id": customer_id}, context)
        if kind in {"duplicate_call", "tool_budget"}:
            assert execute("customer", "get_customer", "CUST-001")["ok"]
        error_case = {
            "lookup_required": ("customer", "predict_churn", "CUST-001", "lookup_required"),
            "duplicate_call": ("customer", "get_customer", "CUST-001", "repeated_tool_call"),
            "tool_budget": ("customer", "predict_churn", "CUST-001", "tool_budget_exceeded"),
            "wrong_id": ("customer", "get_customer", "CUST-002", "customer_id_mismatch"),
            "wrong_tool": ("knowledge", "get_customer", "CUST-001", "tool_not_allowed")
        }
        agent, tool, customer_id, code = error_case[kind]
        assert execute(agent, tool, customer_id)["error"]["code"] == code
    return {}


def evaluate_offline():
    assistant = Assistant(Settings(app_mode="offline"))
    cases = json.loads((ROOT / "data/evaluation_cases.json").read_text(encoding="utf-8"))
    results = []
    with TestClient(create_app(assistant=assistant)) as client:
        for case in cases:
            started = time.perf_counter()
            try:
                detail = check_case(case, assistant, client)
                result = {"id": case["id"], "passed": True, "detail": detail}
            except Exception as exc:
                result = {"id": case["id"], "passed": False, "error_type": type(exc).__name__}
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            results.append(result)
    report = {"mode": "offline", "run_at": datetime.now(ZoneInfo("Africa/Cairo")).isoformat(),
              "scope": "Real SQL, sklearn inference, FAISS retrieval, LangGraph and REST; rule-based routing and extractive answers; no Gemini calls.",
              "passed": sum(result["passed"] for result in results), "total": len(results), "cases": results}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def evaluate_live():
    from telecom.evaluate_gemini import evaluate_gemini
    return evaluate_gemini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Makes live Gemini requests; requires a working key and quota.")
    report = evaluate_live() if parser.parse_args().live else evaluate_offline()
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
