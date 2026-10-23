import copy

import pytest

from telecom.evaluate_gemini import evidence_checks
from telecom.schemas import ChatRequest


@pytest.fixture
def observed_combined(assistant, monkeypatch):
    calls = []
    original = assistant.tools.execute
    def observe(agent, name, arguments, context):
        result = original(agent, name, arguments, context)
        calls.append({"tool": name, "result": result})
        return result
    monkeypatch.setattr(assistant.tools, "execute", observe)
    result = assistant.chat(ChatRequest(message="Predict churn for CUST-001 and explain the cancellation policy")).model_dump()
    return result, calls


def test_live_evaluator_requires_actual_retrieved_excerpts(assistant, observed_combined):
    body, calls = observed_combined
    checks = evidence_checks(body, calls, assistant, "combined", "cancellation")
    assert all(checks.values())
    tampered = copy.deepcopy(body)
    faq = next(source for source in tampered["sources"] if source["source_id"].startswith("faq:"))
    faq["excerpt"] = "Invented policy text"
    assert not evidence_checks(tampered, calls, assistant, "combined", "cancellation")["citations_match_executed_retrieval"]


def test_live_evaluator_rejects_fabricated_model_scores(assistant, observed_combined):
    body, calls = observed_combined
    tampered = copy.deepcopy(body)
    model = next(source for source in tampered["sources"] if source["source_id"].startswith("model:"))
    model["excerpt"] = '{"churn_probability":0.123}'
    assert not evidence_checks(tampered, calls, assistant, "combined", "cancellation")["score_equals_actual_pipeline"]
