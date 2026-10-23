"""Official Gemini SDK contracts using simulated HTTP responses; never a live key."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from google import genai
from google.genai import types
from pydantic import ValidationError

from telecom.api import create_app
from telecom.config import Settings
from telecom.errors import AppError
from telecom.gemini_llm import GeminiLLM, native_history
from telecom.rag import UNCERTAIN
from telecom.schemas import ChatRequest, Source
from telecom.tools import tool_spec
from telecom.workflow import Assistant

pytestmark = pytest.mark.mocked_llm


def payload(parts):
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}


def structured(data):
    return payload([{"text": json.dumps(data)}])


def adapter(handler):
    settings = Settings(app_mode="live", gemini_api_key="unit-test-only")
    native = genai.Client(api_key="unit-test-only", vertexai=False,
        http_options=types.HttpOptions(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                       retry_options=types.HttpRetryOptions(attempts=1)))
    return GeminiLLM(settings, client=native)


def test_gemini_graph_contract_preserves_signatures_and_executes_real_tools(assistant):
    requests = []

    def provider_fixture(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["x-goog-api-key"] == "unit-test-only"
        assert "key=" not in str(request.url)
        schema = body["generationConfig"].get("responseJsonSchema", {}).get("title")
        if schema == "RouteDecision":
            data = structured({"route": "both"})
        elif schema == "GroundedAnswer":
            data = structured({"supported": True, "answer": "Contact support to request cancellation; identity verification is required.",
                               "source_ids": ["faq:cancellation:0"]})
        else:
            declaration = body["tools"][0]["functionDeclarations"][0]
            assert declaration["parameters_json_schema"]["additionalProperties"] is False
            if declaration["name"] == "get_customer":
                name = "predict_churn" if len(body["contents"]) > 1 else "get_customer"
                arguments = {"customer_id": "CUST-001"}
                if name == "predict_churn":
                    assert body["contents"][1]["parts"][0]["thoughtSignature"] == "dGVzdA=="
                    actual = body["contents"][2]["parts"][0]["functionResponse"]
                    assert actual["name"] == "get_customer" and actual["response"]["data"]["calls"] == 35
                    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
            else:
                shared = json.loads(body["contents"][0]["parts"][0]["text"])
                assert "Estimated churn probability" in shared["other_agent_findings"]
                name, arguments = "search_faqs", {"query": "How do I cancel my contract?", "top_k": 3}
            data = payload([{"functionCall": {"name": name, "args": arguments}, "thoughtSignature": "dGVzdA=="}])
        return httpx.Response(200, json=data)

    llm = adapter(provider_fixture)
    service = Assistant(llm.settings, llm=llm, retriever=assistant.retriever(), predictor=assistant.predictor())
    try:
        result = service.chat(ChatRequest(message="Predict churn for CUST-001 and explain cancellation policy"))
        assert [trace.tool for trace in result.tool_trace] == ["get_customer", "predict_churn", "search_faqs"]
        assert not result.errors and result.route == "both" and len(requests) == 5
        assert "94.9%" in result.answer
        assert {source.source_id.split(":")[0] for source in result.sources} == {"customer", "model", "faq"}
    finally:
        llm.client.close()


def test_gemini_auto_can_stop_after_profile_lookup():
    llm = adapter(lambda request: httpx.Response(200, json=payload([{"text": "Enough profile facts obtained."}])))
    call = types.Content(role="model", parts=[types.Part.from_function_call(name="get_customer", args={"customer_id": "CUST-001"})])
    history = [call, {"type": "function_call_output", "call_id": "example", "output": '{"ok":true,"data":{}}'}]
    try:
        chosen, _ = llm.choose_tool("customer", "Show profile", "CUST-001", history, [tool_spec("get_customer")])
        assert chosen is None
    finally:
        llm.client.close()


@pytest.mark.parametrize("question,selected", [
    ("Show the profile for CUST-002.", None),
    ("Show the usage details for cust-002.", None),
    ("Show this customer's profile.", "CUST-002"),
])
def test_profile_graph_uses_resolved_id_and_only_customer_lookup(question, selected):
    requests = []

    def provider_fixture(request):
        body = json.loads(request.content)
        requests.append(body)
        schema = body["generationConfig"].get("responseJsonSchema", {}).get("title")
        if schema == "RouteDecision":
            task = json.loads(body["contents"][0]["parts"][0]["text"])
            assert task == {"message": question, "selected_customer_id": "CUST-002"}
            response = structured({"route": "customer"})
        elif len(body["contents"]) == 1:
            task = json.loads(body["contents"][0]["parts"][0]["text"])
            assert task["customer_id"] == "CUST-002"
            response = payload([{"functionCall": {"name": "get_customer", "args": {"customer_id": "CUST-002"}}}])
        else:
            record = body["contents"][2]["parts"][0]["functionResponse"]["response"]["data"]
            assert record["customer_id"] == "CUST-002" and record["calls"] == 220
            response = payload([{"text": "Profile obtained."}])
        return httpx.Response(200, json=response)

    llm = adapter(provider_fixture)
    try:
        result = Assistant(llm.settings, llm=llm).chat(ChatRequest(message=question, customer_id=selected))
        assert result.route == "customer" and not result.errors
        assert [(trace.tool, trace.status) for trace in result.tool_trace] == [("get_customer", "success")]
        assert [source.source_id for source in result.sources] == ["customer:CUST-002"]
        assert json.loads(result.sources[0].excerpt)["calls"] == 220
        assert "Demo Customer B" in result.answer and len(requests) == 3
    finally:
        llm.client.close()


@pytest.mark.parametrize("parts", [
    [{"text": "No call"}],
    [{"functionCall": {"name": "get_customer", "args": {}}}, {"functionCall": {"name": "predict_churn", "args": {}}}]
])
def test_initial_step_requires_exactly_one_call(parts):
    llm = adapter(lambda request: httpx.Response(200, json=payload(parts)))
    try:
        with pytest.raises(AppError) as caught:
            llm.choose_tool("customer", "Show profile", "CUST-001", [], [tool_spec("get_customer")])
        assert caught.value.code == "llm_invalid_output"
    finally:
        llm.client.close()


@pytest.mark.parametrize("status,code", [(400, "llm_request_rejected"), (401, "llm_authentication_failed"), (403, "llm_access_denied"),
                                        (404, "llm_model_unavailable"), (429, "llm_rate_limited"), (503, "llm_unavailable")])
def test_provider_errors_are_sanitized_and_not_retried(status, code):
    requests = []
    def fixture(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": status, "message": "secret-provider-detail"}})
    llm = adapter(fixture)
    try:
        with pytest.raises(AppError) as caught:
            llm.route("How do I roam?", None)
        assert caught.value.code == code and len(requests) == 1
        assert "secret-provider-detail" not in str(caught.value) and caught.value.__suppress_context__
    finally:
        llm.client.close()


@pytest.mark.parametrize("text", ["not JSON", '{"route":"invented"}'])
def test_structured_output_is_validated(text):
    llm = adapter(lambda request: httpx.Response(200, json=payload([{"text": text}])))
    try:
        with pytest.raises(AppError) as caught:
            llm.route("How do I roam?", None)
        assert caught.value.code == "llm_invalid_output"
    finally:
        llm.client.close()


@pytest.mark.parametrize("supported,ids,expected", [(False, [], "uncertain"), (True, ["invented"], "invalid")])
def test_grounding_uncertainty_and_citation_validation(supported, ids, expected):
    llm = adapter(lambda request: httpx.Response(200, json=structured({"supported": supported, "answer": "Fixture text", "source_ids": ids})))
    source = Source(source_id="faq:roaming:0", title="Roaming", reference="data/faqs.json#roaming", excerpt="Enable roaming in the app.")
    try:
        if expected == "uncertain":
            assert llm.grounded_answer("Unknown price?", [source]) == (UNCERTAIN, [])
        else:
            with pytest.raises(AppError) as caught:
                llm.grounded_answer("How do I roam?", [source])
            assert caught.value.code == "llm_invalid_citation"
    finally:
        llm.client.close()


def test_missing_key_and_invalid_mode_are_rejected():
    with pytest.raises(AppError) as caught:
        GeminiLLM(Settings(gemini_api_key=""))
    assert caught.value.code == "api_key_missing"
    with pytest.raises(ValidationError):
        Settings(app_mode="unknown")
    assert "unit-test-secret" not in repr(Settings(gemini_api_key="unit-test-secret"))


def test_lazy_gemini_construction_and_health(assistant, monkeypatch):
    sentinel = object()
    monkeypatch.setattr("telecom.workflow.GeminiLLM", lambda settings: sentinel)
    service = Assistant(Settings(app_mode="live", gemini_api_key="unit-test-only"),
                        retriever=assistant.retriever(), predictor=assistant.predictor())
    assert service.llm() is sentinel
    health = TestClient(create_app(assistant=service)).get("/health")
    assert health.status_code == 200 and health.json()["llm_provider"] == "gemini"
    assert health.json()["llm_key_configured"]


def test_offline_gemini_never_constructs_client(assistant, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline mode cannot instantiate an LLM")
    monkeypatch.setattr("telecom.workflow.GeminiLLM", forbidden)
    service = Assistant(Settings(app_mode="offline", gemini_api_key=""), retriever=assistant.retriever())
    assert service.chat(ChatRequest(message="How do I activate roaming?")).sources


def test_invalid_history_is_rejected():
    with pytest.raises(AppError):
        native_history([{"type": "function_call_output", "output": "{}"}])


def test_network_timeout_is_sanitized():
    def timeout(request):
        raise httpx.ReadTimeout("secret-provider-detail", request=request)
    llm = adapter(timeout)
    try:
        with pytest.raises(AppError) as caught:
            llm.route("How do I roam?", None)
        assert caught.value.code == "llm_unavailable"
        assert "secret-provider-detail" not in str(caught.value)
    finally:
        llm.client.close()


def test_no_evidence_skips_gemini_generation():
    def forbidden(request):
        raise AssertionError("No request is allowed without evidence")
    llm = adapter(forbidden)
    try:
        assert llm.grounded_answer("Unknown policy?", []) == (UNCERTAIN, [])
    finally:
        llm.client.close()
