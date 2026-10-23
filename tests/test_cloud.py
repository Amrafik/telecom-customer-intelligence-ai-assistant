import importlib.metadata
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from telecom import cloud_resources, cloud_runtime, rag
from telecom.cloud_resources import MODEL_DIR, load_trusted_model, offline_settings, prepare_assistant
from telecom.cloud_runtime import CloudLimits, CloudRuntime, UsageLimiter, guarded_app
from telecom.config import ROOT, Settings


def test_cloud_settings_ignore_dotenv_and_live_environment(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("APP_MODE=invalid-local-mode\nGEMINI_API_KEY=do-not-use-this-local-value\n")
    monkeypatch.setitem(Settings.model_config, "env_file", path)
    monkeypatch.setenv("APP_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "do-not-use-this-environment-value")
    settings = offline_settings()
    assert settings.app_mode == "offline" and settings.gemini_api_key == ""


def test_packaged_model_matches_existing_model(assistant):
    predictor = load_trusted_model()
    customer = assistant.customers.get("CUST-001")
    assert predictor.predict(customer) == assistant.predictor().predict(customer)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_benchmark_fingerprint_survives_git_line_endings(tmp_path, monkeypatch, newline):
    source = (ROOT / "reports/benchmark.json").read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/benchmark.json").write_bytes(source.replace(b"\n", newline))
    monkeypatch.setattr(cloud_resources, "ROOT", tmp_path)
    assert load_trusted_model(MODEL_DIR).bundle["model_version"].startswith("uci563-")


@pytest.mark.parametrize("fault", ["artifact", "sklearn", "benchmark"])
def test_artifact_rejected_before_deserialization(tmp_path, monkeypatch, fault):
    for name in ["churn_pipeline.joblib", "manifest.json"]:
        shutil.copyfile(MODEL_DIR / name, tmp_path / name)
    if fault == "artifact":
        (tmp_path / "churn_pipeline.joblib").write_bytes(b"untrusted")
    else:
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["sklearn" if fault == "sklearn" else "benchmark_sha256"] = "wrong"
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    def forbidden_load(*args):
        raise AssertionError("Rejected bytes must not be deserialized.")
    monkeypatch.setattr(cloud_resources, "Predictor", forbidden_load)
    with pytest.raises(ValueError):
        load_trusted_model(tmp_path)


def test_cold_faiss_sql_initialization_and_warm_reuse(tmp_path, monkeypatch, assistant):
    embedder = assistant.retriever().embedder
    loads = []
    def local_embedder(path, **kwargs):
        loads.append(str(path))
        return embedder
    monkeypatch.setattr(rag, "load_embedder", local_embedder)
    service = prepare_assistant(tmp_path)
    assert service.settings.app_mode == "offline" and not service.settings.gemini_api_key
    assert service.customers.path == (tmp_path / "customers.sqlite3").resolve()
    assert service.customers.get("CUST-002").calls == assistant.customers.get("CUST-002").calls
    assert service.retriever().search("How do I activate roaming?")[0].source_id == "faq:roaming:0"
    assert service.predictor().predict(service.customers.get("CUST-001")) == assistant.predictor().predict(assistant.customers.get("CUST-001"))
    assert len(loads) == 1, "Cold build must reuse the already loaded embedder."
    def unexpected_build(*args, **kwargs):
        raise AssertionError("A valid persisted index must not be rebuilt.")
    monkeypatch.setattr(cloud_resources, "build_index", unexpected_build)
    restored = prepare_assistant(tmp_path)
    assert len(loads) == 2, "A new process/service loads one embedder from its saved cache."
    assert restored.retriever().index.ntotal == 11


def test_limiter_concurrent_admission_and_minute_limit(tmp_path):
    limiter = UsageLimiter(tmp_path / "usage.sqlite3", CloudLimits(requests_per_minute=1))
    barrier = threading.Barrier(8)
    def acquire():
        barrier.wait()
        return limiter.acquire()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: acquire(), range(8)))
    assert results.count(None) == 1
    assert sum(result and result[0] == "demo_busy" for result in results if result) == 7
    limiter.release()
    assert limiter.acquire()[0] == "demo_minute_limit"


def test_daily_limit_survives_new_limiter(tmp_path):
    path = tmp_path / "usage.sqlite3"
    limits = CloudLimits(requests_per_day=1)
    limiter = UsageLimiter(path, limits)
    assert limiter.acquire() is None
    limiter.release()
    assert UsageLimiter(path, limits).acquire()[0] == "demo_daily_limit"


def test_health_auth_validation_and_failed_requests_count(assistant, tmp_path):
    limiter = UsageLimiter(tmp_path / "usage.sqlite3", CloudLimits(requests_per_day=1))
    with TestClient(guarded_app(assistant, limiter, "internal-test-token")) as client:
        assert client.get("/health").status_code == 401
        headers = {"X-Cloud-Token": "internal-test-token"}
        health = client.get("/health", headers=headers)
        assert health.status_code == 200 and health.json()["mode"] == "offline"
        assert health.json()["deployment"] == "private cloud trial (offline)"
        assert client.post("/predict", json={"customer_id": "bad"}, headers=headers).status_code == 422
        limited = client.post("/predict", json={"customer_id": "CUST-001"}, headers=headers)
        assert limited.status_code == 429 and limited.json()["code"] == "demo_daily_limit"
        assert "Retry-After" in limited.headers
        assert client.get("/health", headers=headers).status_code == 200


def test_concurrency_guard_and_release_after_failure(assistant, tmp_path, monkeypatch):
    limiter = UsageLimiter(tmp_path / "usage.sqlite3", CloudLimits())
    with TestClient(guarded_app(assistant, limiter, "token"), raise_server_exceptions=False) as client:
        headers = {"X-Cloud-Token": "token"}
        assert limiter.acquire() is None
        assert client.get("/customers/CUST-001", headers=headers).json()["code"] == "demo_busy"
        limiter.release()
        with monkeypatch.context() as patch:
            def fail(*args):
                raise RuntimeError("deliberate failure")
            patch.setattr(assistant.customers, "get", fail)
            assert client.get("/customers/CUST-001", headers=headers).status_code == 500
        assert client.get("/customers/CUST-001", headers=headers).status_code == 200


def test_operator_disable_keeps_health_available(assistant, tmp_path):
    limiter = UsageLimiter(tmp_path / "usage.sqlite3", CloudLimits(enabled=False))
    with TestClient(guarded_app(assistant, limiter, "token")) as client:
        headers = {"X-Cloud-Token": "token"}
        assert client.get("/health", headers=headers).status_code == 200
        assert client.get("/customers/CUST-001", headers=headers).json()["code"] == "demo_disabled"


@pytest.mark.parametrize("setting", ["app_mode", "gemini_api_key"])
def test_live_settings_cannot_enter_cloud_backend(assistant, tmp_path, monkeypatch, setting):
    monkeypatch.setattr(assistant.settings, setting, "live" if setting == "app_mode" else "test-only-key")
    with pytest.raises(ValueError, match="offline"):
        guarded_app(assistant, UsageLimiter(tmp_path / "usage.sqlite3", CloudLimits()), "token")


def test_server_start_once_parallel_reruns_and_restart(assistant, tmp_path):
    calls = []
    def factory(path):
        calls.append(path)
        return assistant
    runtime = CloudRuntime(tmp_path, factory=factory)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            urls = list(pool.map(lambda _: runtime.ensure_started().url, range(4)))
        assert len(set(urls)) == 1 and len(calls) == 1
        with httpx.Client(headers=runtime.headers, trust_env=False) as client:
            assert client.get(runtime.url + "/health").status_code == 200
            assert client.post(runtime.url + "/predict", json={"customer_id": "CUST-001"}).status_code == 200
        thread = runtime.thread
        runtime.stop()
        assert not thread.is_alive()
        runtime.ensure_started()
        assert runtime.thread is not thread and len(calls) == 1
        assert runtime.assistant is assistant
    finally:
        runtime.stop()


def test_unready_server_stops_and_can_retry(assistant, tmp_path, monkeypatch):
    runtime = CloudRuntime(tmp_path, factory=lambda path: assistant)
    try:
        with monkeypatch.context() as patch:
            from telecom.errors import AppError
            def unavailable():
                raise AppError("Missing artifact", "model_unavailable")
            patch.setattr(assistant, "predictor", unavailable)
            with pytest.raises(RuntimeError, match="ready"):
                runtime.ensure_started(timeout=0.3)
        assert not runtime.thread.is_alive()
        runtime.ensure_started()
        assert runtime.server.started
    finally:
        runtime.stop()


def test_process_registry_survives_streamlit_cache_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_runtime, "_runtime", None)
    first = cloud_runtime.get_runtime(CloudLimits())
    from streamlit import cache_resource
    cache_resource.clear()
    assert cloud_runtime.get_runtime(CloudLimits()) is first
    with pytest.raises(ValueError, match="Restart"):
        cloud_runtime.get_runtime(CloudLimits(requests_per_day=1))


def test_cloud_ui_chats_and_reruns_share_backend(assistant, tmp_path, monkeypatch):
    import streamlit as st
    previous_toolbar = st.get_option("client.toolbarMode")
    runtime = CloudRuntime(tmp_path, factory=lambda path: assistant)
    monkeypatch.setattr(cloud_runtime, "get_runtime", lambda limits: runtime)
    try:
        ui = AppTest.from_file(str(ROOT / "deploy/streamlit/app.py"), default_timeout=40).run()
        assert not ui.exception and len(ui.chat_input) == 1
        notice = ui.info[0].value
        assert "deterministic routing" in notice and "no Gemini API calls" in notice
        assert all(component in notice for component in ["SQLite", "Scikit-learn", "FAISS", "LangGraph"])
        captions = "\n".join(item.value for item in ui.caption)
        assert "full local version" in captions and "verified in recorded live tests" in captions
        assert "https://github.com/Amrafik/telecom-customer-intelligence-ai-assistant" in captions
        assert st.get_option("client.toolbarMode") == "viewer"
        thread = runtime.thread
        ui.chat_input[0].set_value("Predict churn for CUST-001 and explain cancellation.").run()
        assert not ui.exception
        result = ui.session_state["messages"][-1]["result"]
        assert result["route"] == "both" and result["mode"] == "offline" and not result["errors"]
        assert [trace["tool"] for trace in result["tool_trace"]] == ["get_customer", "predict_churn", "search_faqs"]
        ui.run()
        assert not ui.exception and runtime.thread is thread
        ui.button[0].click().run()
        assert ui.session_state["messages"] == []
    finally:
        runtime.stop()
        from streamlit import cache_resource
        cache_resource.clear()
        st.set_option("client.toolbarMode", previous_toolbar)


def test_cloud_dependency_pins_match_local_runtime():
    from packaging.requirements import Requirement
    lines = (ROOT / "deploy/streamlit/requirements.txt").read_text().splitlines()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.name == "torch":
            assert "download.pytorch.org/whl/cpu/" in requirement.url and "cp312" in requirement.url
        else:
            assert requirement.specifier.contains(importlib.metadata.version(requirement.name))
