import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np

from telecom.config import ROOT, Settings
from telecom.errors import AppError
from telecom.schemas import Source

FAQ_PATH = ROOT / "data/faqs.json"
INDEX_DIR = ROOT / "artifacts/index"
EMBEDDING_DIR = ROOT / "models/embeddings"
UNCERTAIN = "I don't have enough information in the fictional DemoTel FAQs to answer that. Please contact DemoTel support to confirm."


def chunks(documents: list[dict], size: int = 50, overlap: int = 10) -> list[dict]:
    if size <= overlap or overlap < 0:
        raise ValueError("Chunk size must exceed nonnegative overlap.")
    result = []
    for doc in documents:
        words = doc["text"].split()
        for number, start in enumerate(range(0, len(words), size - overlap)):
            result.append({"source_id": f"faq:{doc['id']}:{number}", "title": doc["title"],
                           "reference": f"data/faqs.json#{doc['id']}",
                           "excerpt": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
    return result


def load_embedder(path_or_name, **kwargs):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path_or_name), device="cpu", trust_remote_code=False, **kwargs)


def build_index(settings: Settings | None = None, *, index_dir: Path = INDEX_DIR,
                embedding_dir: Path = EMBEDDING_DIR):
    settings = settings or Settings()
    embedder = load_embedder(settings.embedding_model, revision=settings.embedding_revision)
    documents = chunks(json.loads(FAQ_PATH.read_text(encoding="utf-8")))
    vectors = embedder.encode([f"{doc['title']}: {doc['excerpt']}" for doc in documents],
                             normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedder.save(str(embedding_dir))
    index_path = index_dir / "faqs.faiss"
    faiss.write_index(index, str(index_path))
    metadata = {"model": settings.embedding_model, "revision": settings.embedding_revision,
                "faqs_sha256": hashlib.sha256(FAQ_PATH.read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "dimension": vectors.shape[1], "chunk_words": 50, "overlap_words": 10,
                "documents": documents}
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(documents)} chunks, {vectors.shape[1]} dimensions, CPU embeddings and FAISS index.")
    return embedder


class Retriever:
    def __init__(self, settings: Settings, index_dir: Path = INDEX_DIR, *,
                 embedding_dir: Path = EMBEDDING_DIR, embedder=None):
        try:
            metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
            path = index_dir / "faqs.faiss"
            valid = (metadata["faqs_sha256"] == hashlib.sha256(FAQ_PATH.read_bytes()).hexdigest()
                     and metadata["index_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                     and metadata["model"] == settings.embedding_model
                     and metadata["revision"] == settings.embedding_revision
                     and metadata["documents"] == chunks(json.loads(FAQ_PATH.read_text(encoding="utf-8"))))
            if not valid:
                raise ValueError("Stale index")
            self.index = faiss.read_index(str(path))
            self.documents = metadata["documents"]
            if self.index.ntotal != len(self.documents) or self.index.d != metadata["dimension"]:
                raise ValueError("Index metadata mismatch")
            self.embedder = embedder if embedder is not None else load_embedder(embedding_dir, local_files_only=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise AppError("FAQ index unavailable or stale. Run python -m telecom.build_index.", "index_unavailable") from exc
        self.threshold = settings.retrieval_min_score

    def search(self, query: str, top_k: int = 3) -> list[Source]:
        if not query.strip() or not 1 <= top_k <= 5:
            raise ValueError("Provide a question and top_k from 1 to 5.")
        vector = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(np.asarray(vector, dtype="float32"), min(top_k, self.index.ntotal))
        return [Source(**self.documents[int(i)], score=round(float(score), 6))
                for score, i in zip(scores[0], indices[0]) if i >= 0 and score >= self.threshold]
