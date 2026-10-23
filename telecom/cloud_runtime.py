"""One internal offline backend, with shared limits for a small private trial."""
import atexit
import hmac
import secrets
import socket
import sqlite3
import threading
import time
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from telecom.api_factory import create_app
from telecom.cloud_resources import prepare_assistant
from telecom.config import ROOT
from telecom.monitoring import event


class CloudLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    requests_per_minute: int = Field(default=6, ge=1, le=30)
    requests_per_day: int = Field(default=100, ge=1, le=1000)


class UsageLimiter:
    def __init__(self, path: Path, limits: CloudLimits):
        self.path, self.limits = path, limits
        self._lock = threading.Lock()
        self._recent, self._active = deque(), False
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS daily_usage (day TEXT PRIMARY KEY, requests INTEGER NOT NULL)")

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            while self._recent and now - self._recent[0] >= 60:
                self._recent.popleft()
            if not self.limits.enabled:
                return "demo_disabled", 60
            if self._active:
                return "demo_busy", 1
            if len(self._recent) >= self.limits.requests_per_minute:
                return "demo_minute_limit", max(1, int(61 - (now - self._recent[0])))
            day = datetime.now(timezone.utc).date().isoformat()
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT requests FROM daily_usage WHERE day = ?", (day,)).fetchone()
                if row and row[0] >= self.limits.requests_per_day:
                    return "demo_daily_limit", 3600
                db.execute("INSERT INTO daily_usage VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET requests = requests + 1", (day,))
            self._recent.append(now)
            self._active = True
        return None

    def release(self):
        with self._lock:
            self._active = False


def guarded_app(assistant, limiter: UsageLimiter, token: str):
    if assistant.settings.app_mode != "offline" or assistant.settings.gemini_api_key:
        raise ValueError("This cloud trial supports offline mode without credentials only.")
    app = create_app(assistant=assistant, deployment="private cloud trial (offline)")

    @app.middleware("http")
    async def cloud_guard(request, call_next):
        supplied = request.headers.get("X-Cloud-Token", "")
        if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            return JSONResponse(status_code=401, content={"code": "internal_client_required"})
        if request.url.path == "/health":
            return await call_next(request)
        rejection = limiter.acquire()
        if rejection:
            code, retry = rejection
            event("cloud_request_rejected", status="error", error_code=code)
            return JSONResponse(status_code=429, content={"code": code, "detail": "Demo request allowance is unavailable. Try again later."},
                                headers={"Retry-After": str(retry)})
        try:
            return await call_next(request)
        finally:
            limiter.release()

    return app


class CloudRuntime:
    def __init__(self, cache_dir: Path = ROOT / ".cache/cloud", limits: CloudLimits | None = None, *, factory=prepare_assistant):
        self.cache_dir = cache_dir
        self.limits = limits or CloudLimits()
        self.factory = factory
        self._lock = threading.RLock()
        self.assistant = self.limiter = self.server = self.thread = None
        self.headers = {"X-Cloud-Token": secrets.token_urlsafe(32)}
        self.url = ""
        atexit.register(self.stop)

    def ensure_started(self, timeout: float = 30):
        with self._lock:
            if self.thread and self.thread.is_alive() and self.server.started:
                return self
            if self.thread and self.thread.is_alive():
                raise RuntimeError("The previous internal backend is still stopping.")
            if self.assistant is None:
                self.assistant = self.factory(self.cache_dir)
                self.limiter = UsageLimiter(self.cache_dir / "usage.sqlite3", self.limits)
            app = guarded_app(self.assistant, self.limiter, self.headers["X-Cloud-Token"])
            sock = socket.socket()
            self.socket = sock
            sock.bind(("127.0.0.1", 0))
            self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
            self.server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False, lifespan="off"))
            self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]}, daemon=True)
            self.thread.start()
            deadline = time.monotonic() + timeout
            while self.thread.is_alive() and time.monotonic() < deadline:
                try:
                    response = httpx.get(self.url + "/health", headers=self.headers, timeout=1, trust_env=False)
                    if response.status_code == 200:
                        return self
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            self.stop()
            raise RuntimeError("The internal offline backend did not become ready.")

    def stop(self):
        with self._lock:
            if self.server:
                self.server.should_exit = True
            if self.thread:
                self.thread.join(timeout=5)
            if hasattr(self, "socket") and (not self.thread or not self.thread.is_alive()):
                self.socket.close()


_runtime = None
_runtime_lock = threading.Lock()


def get_runtime(limits: CloudLimits):
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = CloudRuntime(limits=limits)
        elif _runtime.limits != limits:
            raise ValueError("Restart the cloud application after changing its limits.")
        return _runtime
