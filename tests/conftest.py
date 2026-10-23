import logging
import socket

import pytest
from fastapi.testclient import TestClient

from telecom.api import create_app
from telecom.config import Settings
from telecom.workflow import Assistant


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    from telecom import monitoring
    logger = logging.getLogger("telecom.events")
    previous = list(logger.handlers)
    for handler in previous:
        logger.removeHandler(handler)
    monkeypatch.setattr(monitoring, "ROOT", tmp_path)
    try:
        yield tmp_path
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in previous:
            logger.addHandler(handler)


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Tests are free: only loopback sockets and mock HTTP transports are allowed."""
    original = socket.socket.connect

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network is disabled during tests.")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setenv("APP_MODE", "offline")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-only-not-a-real-key")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="session")
def assistant():
    return Assistant(Settings(app_mode="offline", gemini_api_key=""))


@pytest.fixture
def client(assistant):
    with TestClient(create_app(assistant=assistant)) as instance:
        yield instance
