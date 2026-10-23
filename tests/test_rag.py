import pytest

from telecom.config import Settings
from telecom.errors import AppError
from telecom.rag import Retriever, chunks


def test_chunk_overlap_and_source_references():
    docs = [{"id": "test", "title": "Test", "text": "one two three four five six seven eight nine"}]
    result = chunks(docs, size=5, overlap=2)
    assert result[0]["excerpt"] == "one two three four five"
    assert result[1]["excerpt"].startswith("four five six")
    assert all(row["reference"] == "data/faqs.json#test" for row in result)
    with pytest.raises(ValueError):
        chunks(docs, size=5, overlap=5)


@pytest.mark.integration
@pytest.mark.parametrize("question,prefix", [
    ("How do I activate international roaming?", "faq:roaming:"),
    ("How do I cancel my contract?", "faq:cancellation:"),
    ("How much is Flex Plus per month?", "faq:plans:")])
def test_semantic_retrieval_hits_expected_source(assistant, question, prefix):
    sources = assistant.retriever().search(question)
    assert sources and sources[0].source_id.startswith(prefix)
    assert all(source.score >= 0.35 and source.excerpt for source in sources)


def test_retrieval_rejects_unrelated_question(assistant):
    assert assistant.retriever().search("What is the capital of Mars?") == []


def test_missing_index_is_actionable(tmp_path):
    with pytest.raises(AppError, match="index unavailable or stale"):
        Retriever(Settings(), tmp_path)


def test_stale_index_is_rejected(tmp_path):
    import json
    from telecom.rag import INDEX_DIR
    meta = json.loads((INDEX_DIR / "metadata.json").read_text())
    meta["faqs_sha256"] = "changed"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "faqs.faiss").write_bytes((INDEX_DIR / "faqs.faiss").read_bytes())
    with pytest.raises(AppError):
        Retriever(Settings(), tmp_path)


def test_index_manifest_cannot_substitute_source_text(tmp_path):
    import json
    from telecom.rag import INDEX_DIR
    meta = json.loads((INDEX_DIR / "metadata.json").read_text())
    meta["documents"][0]["excerpt"] = "Invented policy not present in source"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "faqs.faiss").write_bytes((INDEX_DIR / "faqs.faiss").read_bytes())
    with pytest.raises(AppError) as caught:
        Retriever(Settings(), tmp_path)
    assert caught.value.code == "index_unavailable"
