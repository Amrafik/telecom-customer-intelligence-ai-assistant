"""Copy the project's trusted, already trained model; never fit at deployment time."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telecom.ml import Predictor


def package():
    source = ROOT / "models/churn_pipeline.joblib"
    predictor = Predictor(source)
    benchmark_path = ROOT / "reports/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (predictor.bundle["model_version"] != benchmark["model_version"]
            or predictor.bundle["dataset_sha256"] != benchmark["dataset_sha256"]):
        raise ValueError("The local model does not match the recorded benchmark.")
    destination = ROOT / "deploy/streamlit/artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / source.name)
    manifest = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model_version": benchmark["model_version"], "dataset_sha256": benchmark["dataset_sha256"],
                "sklearn": benchmark["environment"]["sklearn"],
                "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "benchmark_hash_format": "File bytes with CRLF normalized to LF for cross-platform Git checkouts.",
                "origin": "Byte-for-byte copy of the existing locally trained public-data pipeline. No retraining."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Cloud model packaged with checksum and benchmark provenance.")


if __name__ == "__main__":
    package()
