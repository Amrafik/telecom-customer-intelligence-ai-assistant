import pytest

from telecom.schemas import ChatRequest
from telecom.tools import ToolContext
from telecom.workflow import Assistant


@pytest.mark.parametrize("question,route", [
    ("Explain roaming activation", "knowledge"),
    ("Show CUST-001 profile", "customer"),
    ("Predict churn for CUST-001 and explain the cancellation policy", "both"),
    ("Write a chocolate cake recipe", "clarify"),
    ("Predict churn", "clarify")])
def test_graph_routes_and_terminates(assistant, question, route):
    assert assistant.chat(ChatRequest(message=question)).route == route


def test_both_agents_share_and_combine_real_evidence(assistant):
    result = assistant.chat(ChatRequest(message="Predict churn for CUST-001 and explain the cancellation policy"))
    assert [trace.tool for trace in result.tool_trace] == ["get_customer", "predict_churn", "search_faqs"]
    assert {source.source_id.split(":")[0] for source in result.sources} == {"customer", "model", "faq"}
    assert "Estimated churn probability:" in result.answer and "cancellation" in result.answer
    assert not result.errors


@pytest.mark.parametrize("message,selected", [
    ("Show CUST-001 and CUST-002", None),
    ("Show CUST-002", "CUST-001"),
    ("Predict churn for CUST-1", None)])
def test_ambiguous_or_malformed_ids_do_not_execute_tools(assistant, message, selected):
    result = assistant.chat(ChatRequest(message=message, customer_id=selected))
    assert result.route == "clarify" and result.tool_trace == []


def test_unknown_customer_is_explicit_failure(assistant):
    result = assistant.chat(ChatRequest(message="Predict churn for CUST-999"))
    assert result.errors == ["customer_not_found"] and not result.sources


@pytest.mark.parametrize("agent,tool,args,error", [
    ("customer", "delete_database", {}, "tool_not_allowed"),
    ("knowledge", "get_customer", {"customer_id": "CUST-001"}, "tool_not_allowed"),
    ("customer", "get_customer", {"customer_id": "CUST-002"}, "customer_id_mismatch"),
    ("customer", "get_customer", {"customer_id": "CUST-1"}, "invalid_tool_input"),
    ("customer", "get_customer", {"customer_id": "CUST-001", "unexpected": True}, "invalid_tool_input"),
    ("knowledge", "search_faqs", {"query": "roaming", "top_k": 99}, "invalid_tool_input"),
    ("customer", "predict_churn", {"customer_id": "CUST-001"}, "lookup_required"),
    ("knowledge", "search_faqs", "not json", "invalid_tool_input")])
def test_tool_validation_and_allowlist(assistant, agent, tool, args, error):
    result = assistant.tools.execute(agent, tool, args, ToolContext("CUST-001"))
    assert not result["ok"] and result["error"]["code"] == error


def test_repeated_calls_and_budget_stop_execution(assistant):
    context = ToolContext("CUST-001", max_calls=2)
    args = {"customer_id": "CUST-001"}
    assert assistant.tools.execute("customer", "get_customer", args, context)["ok"]
    assert assistant.tools.execute("customer", "get_customer", args, context)["error"]["code"] == "repeated_tool_call"
    assert assistant.tools.execute("customer", "predict_churn", args, context)["error"]["code"] == "tool_budget_exceeded"
    assert context.attempts == 2


def test_offline_mode_never_constructs_gemini_client(monkeypatch, assistant):
    def forbidden(*args, **kwargs):
        raise AssertionError("Gemini must not be called in offline mode")
    monkeypatch.setattr("telecom.workflow.GeminiLLM", forbidden)
    result = assistant.chat(ChatRequest(message="How do I activate roaming?"))
    assert result.mode == "offline" and result.sources
