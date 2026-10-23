"""Cloud-only resources; never train a model or read local credentials at startup."""
import hashlib
import json
from pathlib import Path

import sklearn

from telecom.config import ROOT, Settings
from telecom.errors import AppError
from telecom.init_db import initialise_database
from telecom.ml import CustomerStore, Predictor
from telecom.rag import Retriever, build_index
from telecom.workflow import Assistant

MODEL_DIR = ROOT / "deploy/streamlit/artifacts"


def offline_settings():
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    defaults.update(app_mode="offline", gemini_api_key="")
    return Settings(_env_file=None, **defaults)


def load_trusted_model(directory: Path = MODEL_DIR):
    path = directory / "churn_pipeline.joblib"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Cloud model checksum mismatch; do not load this artifact.")
    if sklearn.__version__ != manifest["sklearn"]:
        raise ValueError("Cloud model requires its recorded scikit-learn version.")
    benchmark_bytes = (ROOT / "reports/benchmark.json").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(benchmark_bytes).hexdigest() != manifest["benchmark_sha256"]:
        raise ValueError("Cloud model benchmark fingerprint does not match.")
    predictor = Predictor(path)
    if (predictor.bundle["model_version"] != manifest["model_version"]
            or predictor.bundle["dataset_sha256"] != manifest["dataset_sha256"]):
        raise ValueError("Cloud model provenance does not match its manifest.")
    return predictor


def prepare_assistant(cache_dir: Path):
    settings = offline_settings()
    predictor = load_trusted_model()
    cache_dir.mkdir(parents=True, exist_ok=True)
    database = cache_dir / "customers.sqlite3"
    initialise_database(database)
    index_dir, embedding_dir = cache_dir / "index", cache_dir / "embeddings"
    try:
        retriever = Retriever(settings, index_dir, embedding_dir=embedding_dir)
    except AppError:
        embedder = build_index(settings, index_dir=index_dir, embedding_dir=embedding_dir)
        retriever = Retriever(settings, index_dir, embedding_dir=embedding_dir, embedder=embedder)
    assistant = Assistant(settings, predictor=predictor, retriever=retriever)
    assistant.customers = CustomerStore(database)
    assistant.tools.customers = assistant.customers
    assistant.customers.get("CUST-001")
    return assistant
