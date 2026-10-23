import socket
import threading
import time

import uvicorn
from streamlit.testing.v1 import AppTest

from telecom.api import create_app
from telecom.config import ROOT


def test_streamlit_chat_uses_real_fastapi(assistant, monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(assistant=assistant), log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        monkeypatch.setenv("BACKEND_URL", f"http://127.0.0.1:{port}")
        ui = AppTest.from_file(str(ROOT / "telecom/ui.py"), default_timeout=40).run()
        assert not ui.exception
        ui.chat_input[0].set_value("Predict churn for CUST-001").run()
        assert not ui.exception
        messages = ui.session_state["messages"]
        assert len(messages) == 2 and messages[-1]["result"]["route"] == "customer"
        assert "Estimated churn probability:" in messages[-1]["result"]["answer"]
        assert messages[-1]["result"]["sources"]
        ui.chat_input[0].set_value("How do I activate international roaming?").run()
        assert not ui.exception
        messages = ui.session_state["messages"]
        assert len(messages) == 4 and messages[-1]["result"]["route"] == "knowledge"
        assert len(ui.chat_message) == 4
        assert not ui.get("doc_string"), "Replaying chat history displayed an internal Streamlit object."
        ui.run()
        assert not ui.exception and len(ui.chat_message) == 4
        assert not ui.get("doc_string")
        ui.chat_input[0].set_value("Write a cake recipe").run()
        assert ui.session_state["messages"][-1]["result"]["intent"] == "out_of_scope"
        ui.chat_input[0].set_value("Why not?").run()
        result = ui.session_state["messages"][-1]["result"]
        assert not ui.exception and result["intent"] == "refusal_followup"
        assert "focused on telecom" in result["answer"] and not result["tool_trace"]
        ui.button[0].click().run()
        assert not ui.exception
        assert ui.session_state["messages"] == [] and not ui.chat_message
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
