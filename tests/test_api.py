import json

import pytest

from telecom.config import ROOT


def test_health_checks_actual_local_dependencies(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert set(response.json()["dependencies"].values()) == {"ready"}
    assert response.json()["mode"] == "offline"


def test_customer_and_prediction_endpoints(client):
    row = client.get("/customers/CUST-001")
    score = client.post("/predict", json={"customer_id": "CUST-001"})
    assert row.status_code == score.status_code == 200
    assert row.json()["fictional"] is True and row.json()["calls"] == 35
    assert 0 <= score.json()["churn_probability"] <= 1


def test_chat_has_sources_trace_and_correlated_headers(client):
    response = client.post("/chat", json={"message": "How do I activate roaming?"})
    body = response.json()
    assert response.status_code == 200 and body["sources"] and body["tool_trace"]
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-Ms"]) >= body["latency_ms"]


@pytest.mark.parametrize("path", ["/customers/CUST-999", "/customers/CUST-998"])
def test_unknown_customers_return_404(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   "}, {"message": "x" * 2001},
                                     {"message": "hello", "customer_id": "bad"}, {"message": "hello", "extra": True}])
def test_chat_rejects_invalid_inputs(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_prediction_rejects_sql_injection(client):
    assert client.post("/predict", json={"customer_id": "CUST-001' OR 1=1 --"}).status_code == 422


def test_logs_contain_timestamps_latency_selections_and_failures(client):
    response = client.post("/chat", json={"message": "Predict churn for CUST-999"})
    request_id = response.headers["X-Request-ID"]
    rows = [json.loads(line) for line in (ROOT / "logs/app.jsonl").read_text().splitlines()]
    correlated = [row for row in rows if row["request_id"] == request_id]
    assert {row["event"] for row in correlated} >= {"agent_selection", "tool_execution", "chat_completed", "http_request"}
    assert all("timestamp" in row for row in correlated)
    assert any(row["status"] == "error" for row in correlated if "status" in row)
    assert all("message" not in row and "api_key" not in row and "customer_id" not in row for row in correlated)


def test_missing_model_returns_503(client, assistant, monkeypatch):
    from telecom.errors import AppError
    def unavailable():
        raise AppError("Train the churn model first", "model_unavailable")
    monkeypatch.setattr(assistant, "predictor", unavailable)
    response = client.post("/predict", json={"customer_id": "CUST-001"})
    assert response.status_code == 503 and response.json()["code"] == "model_unavailable"
