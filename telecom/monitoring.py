import json
import logging
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from telecom.config import ROOT

request_id_var = ContextVar("request_id", default="local")
_lock = threading.Lock()


def event(name: str, **fields):
    logger = logging.getLogger("telecom.events")
    with _lock:
        if not logger.handlers:
            path = ROOT / "logs"
            path.mkdir(exist_ok=True)
            handler = RotatingFileHandler(path / "app.jsonl", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
    logger.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": name,
                            "request_id": request_id_var.get(), **fields}))
