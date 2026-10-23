"""Small transparent checks; offline checks do not measure LLM answer quality."""
import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from telecom.api import create_app
from telecom.config import ROOT, Settings
from telecom.evaluate_gemini import evidence_checks
from telecom.ml import Predictor
from telecom.schemas import ChatRequest
from telecom.tools import ToolContext
from telecom.workflow import Assistant


def check_chat(case, assistant):
    executed = []
    original_execute = assistant.tools.execute

    def observe(agent, name, arguments, context):
        result = original_execute(agent, name, arguments, context)
        executed.append({"tool": name, "result": result})
        return result

    assistant.tools.execute = observe
    try:
        result = assistant.chat(ChatRequest(message=case["question"], customer_id=case.get("selected_customer_id"),
                                            previous_intent=case.get("previous_intent")))
    finally:
        assistant.tools.execute = original_execute
    assert result.route == case["expected"], "Unexpected graph route"
    if "expected_intent" in case:
        assert result.intent == case["expected_intent"], "Unexpected classified intent"
    if "answer_contains" in case:
        assert case["answer_contains"] in result.answer, "Missing scoped response"
    assert [call["tool"] for call in executed] == case["expected_tools"], "Unexpected executed tools"
    statuses = ["success" if call["result"]["ok"] else "error" for call in executed]
    assert statuses == case.get("expected_statuses", ["success"] * len(executed))
    assert [(trace.tool, trace.status) for trace in result.tool_trace] == list(zip(case["expected_tools"], statuses))
    assert result.errors == case.get("expected_errors", []), "Unexpected agent errors"
    assert {source.source_id.split(":")[0] for source in result.sources} == set(case["expected_source_types"])
    if case.get("expected_customer_id"):
        assert any(source.source_id == f"customer:{case['expected_customer_id']}" for source in result.sources)
    if case.get("expected_faq"):
        assert any(source.source_id.startswith(f"faq:{case['expected_faq']}:") for source in result.sources)
    checks = evidence_checks(result.model_dump(), executed, assistant, case["id"], None)
    assert all(checks.values()), "Returned evidence differs from executed tools"
    return {"route": result.route, "intent": result.intent, "tools": case["expected_tools"], "tool_statuses": statuses,
            "error_codes": result.errors, "source_ids": [source.source_id for source in result.sources],
            "evidence_checks": checks}


def check_case(case, assistant, client):
    kind = case["kind"]
    if kind == "retrieval":
        sources = assistant.retriever().search(case["question"])
        assert any(source.source_id.startswith(case["expected"]) for source in sources)
        documents = {item["source_id"]: item for item in assistant.retriever().documents}
        assert all(source.excerpt == documents[source.source_id]["excerpt"]
                   and source.score >= assistant.settings.retrieval_min_score for source in sources)
        return {"source_ids": [source.source_id for source in sources], "scores": [source.score for source in sources]}
    if kind == "retrieval_miss":
        assert assistant.retriever().search(case["question"]) == []
        return {"source_ids": []}
    elif kind in {"route", "chat", "combined", "clarification", "conflicting_ids"}:
        return check_chat(case, assistant)
    elif kind == "missing_model":
        with TemporaryDirectory(prefix="telecom-evaluation-") as directory:
            path = Path(directory) / "missing-model.joblib"
            isolated = Assistant(assistant.settings, retriever=assistant.retriever())
            isolated.tools.predictor = lambda: Predictor(path)
            detail = check_chat(case, isolated)
        return {**detail, "fault_injection": "Real model loader pointed to an absent temporary artifact; no mocked prediction."}
    elif kind == "lookup":
        row = assistant.customers.get(case["customer_id"])
        assert row.fictional and row.calls == case["expected_calls"] and row.display_name == case["expected_name"]
        return row.model_dump()
    elif kind == "unknown_customer":
        response = client.get(f"/customers/{case['customer_id']}")
        assert response.status_code == 404 and response.json()["code"] == "customer_not_found"
        return {"http_status": response.status_code, "error_code": response.json()["code"]}
    elif kind == "prediction":
        payload = {"customer_id": case["customer_id"]}
        first, second = client.post("/predict", json=payload), client.post("/predict", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json() and 0 <= first.json()["churn_probability"] <= 1
        assert first.json() == assistant.predictor().predict(assistant.customers.get(case["customer_id"])).model_dump()
        return first.json()
    elif kind == "bad_request":
        assert client.post("/chat", json=case["body"]).status_code == 422
        return {"http_status": 422}
    elif kind == "bad_predict":
        assert client.post("/predict", json=case["body"]).status_code == 422
        return {"http_status": 422}
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
        if kind not in error_case:
            raise ValueError(f"Unknown evaluation kind: {kind}")
        agent, tool, customer_id, code = error_case[kind]
        assert execute(agent, tool, customer_id)["error"]["code"] == code
        if kind == "wrong_tool":
            assert execute("customer", "delete_account", "CUST-001")["error"]["code"] == "tool_not_allowed"
        return {"error_code": code, "attempts": context.attempts,
                "tool_trace": [trace.model_dump() for trace in context.trace]}
    return {}


def evaluate_offline():
    assistant = Assistant(Settings(app_mode="offline"))
    cases = json.loads((ROOT / "data/evaluation_cases.json").read_text(encoding="utf-8"))
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    results = []
    with TestClient(create_app(assistant=assistant)) as client:
        for case in cases:
            started = time.perf_counter()
            try:
                detail = check_case(case, assistant, client)
                result = {"id": case["id"], "passed": True, "detail": detail}
            except Exception as exc:
                result = {"id": case["id"], "passed": False, "error_type": type(exc).__name__}
            result.update(kind=case["kind"], mode="offline", latency_ms=round((time.perf_counter() - started) * 1000, 3))
            results.append(result)
    report = {"mode": "offline", "run_at": datetime.now(ZoneInfo("Africa/Cairo")).isoformat(),
              "scope": "Real SQL, sklearn inference, FAISS retrieval, LangGraph and REST; rule-based routing and extractive answers. One real missing-artifact fault injection; no mocked model outputs or Gemini calls.",
              "provider_calls": 0, "case_set_version": 3, "case_kinds": dict(Counter(case["kind"] for case in cases)),
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
