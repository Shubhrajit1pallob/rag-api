import pytest

from app.models.document import DocumentIngestRequest, DocumentMetadata, normalize_tags
from app.pipeline.ingest import (
    EmptyDocumentError,
    build_stored_document,
    compute_content_hash,
    ingest_document,
)

CONTENT = "## Intro\nHello world.\n\n## Usage\nRun it daily."


def make_request(content=CONTENT, tags=("Python", "GitHub Actions"), **meta):
    fields = dict(
        title="Doc", category="project", tags=list(tags), featured=False,
        status="published", date="2026-01-01",
    )
    fields.update(meta)
    return DocumentIngestRequest(
        document_id="doc-1", content=content, metadata=DocumentMetadata(**fields)
    )


def ingest(request, store, embedder, model="voyage-3", max_chunk_size=400):
    return ingest_document(
        request, store, embedder, embedding_model=model, max_chunk_size=max_chunk_size
    )


def test_hash_is_stable_and_content_only():
    assert compute_content_hash("a") == compute_content_hash("a")
    assert compute_content_hash("a") != compute_content_hash("b")
    assert len(compute_content_hash("a")) == 64


def test_normalize_tags_trims_drops_empty_dedupes_keeps_order():
    assert normalize_tags([" Python ", "", "  ", "C++", "Python", "C#", "C++"]) == [
        "Python", "C++", "C#",
    ]


def test_normalize_tags_dedupes_case_insensitively_keeping_first_casing():
    assert normalize_tags(["Python", "python", "PYTHON", "C++", "c#", "C#"]) == [
        "Python", "C++", "c#",
    ]


def test_stored_document_carries_all_fields():
    doc = build_stored_document(make_request(tags=(" Python", "")), "h", "voyage-3", 400)
    assert doc.document_id == "doc-1"
    assert doc.tags == ["Python"]
    assert doc.content_hash == "h"
    assert doc.embedding_model == "voyage-3"
    assert doc.max_chunk_size == 400


def test_created_then_unchanged(store, fake_embedder):
    first = ingest(make_request(), store, fake_embedder)
    assert first.status == "created"
    assert first.chunk_count >= 1
    calls = fake_embedder.calls
    second = ingest(make_request(), store, fake_embedder)
    assert second.status == "unchanged"
    assert second.chunk_count == first.chunk_count
    assert fake_embedder.calls == calls


def test_content_updated_reembeds(store, fake_embedder):
    ingest(make_request(), store, fake_embedder)
    calls = fake_embedder.calls
    result = ingest(make_request(content="## New\nDifferent text."), store, fake_embedder)
    assert result.status == "content_updated"
    assert fake_embedder.calls == calls + 1
    assert store.get_document("doc-1").content_hash == compute_content_hash(
        "## New\nDifferent text."
    )


def test_metadata_updated_does_not_reembed(store, fake_embedder):
    ingest(make_request(), store, fake_embedder)
    calls = fake_embedder.calls
    result = ingest(make_request(tags=("Python",), title="Renamed"), store, fake_embedder)
    assert result.status == "metadata_updated"
    assert result.chunk_count == store.count_document_chunks("doc-1")
    assert fake_embedder.calls == calls
    doc = store.get_document("doc-1")
    assert doc.title == "Renamed"
    assert doc.tags == ["Python"]


def test_tag_order_change_is_a_metadata_update(store, fake_embedder):
    ingest(make_request(tags=("A", "B")), store, fake_embedder)
    result = ingest(make_request(tags=("B", "A")), store, fake_embedder)
    assert result.status == "metadata_updated"
    assert store.get_document("doc-1").tags == ["B", "A"]


def test_empty_content_raises(store, fake_embedder):
    with pytest.raises(EmptyDocumentError):
        ingest(make_request(content="   \n"), store, fake_embedder)
    assert fake_embedder.calls == 0
    assert store.get_document("doc-1") is None


def test_model_change_triggers_reembed(store, fake_embedder):
    ingest(make_request(), store, fake_embedder, model="model-a")
    calls = fake_embedder.calls
    result = ingest(make_request(), store, fake_embedder, model="model-b")
    assert result.status == "content_updated"
    assert fake_embedder.calls == calls + 1
    assert store.get_document("doc-1").embedding_model == "model-b"


def test_max_chunk_size_change_triggers_reembed(store, fake_embedder):
    content = "## Long\n" + " ".join(f"w{i}" for i in range(30))
    first = ingest(make_request(content=content), store, fake_embedder, max_chunk_size=400)
    calls = fake_embedder.calls
    result = ingest(make_request(content=content), store, fake_embedder, max_chunk_size=10)
    assert result.status == "content_updated"
    assert fake_embedder.calls == calls + 1
    assert result.chunk_count > first.chunk_count
    assert store.get_document("doc-1").max_chunk_size == 10
