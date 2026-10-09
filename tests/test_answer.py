from types import SimpleNamespace

import pytest

from app.models.query import QueryFilters, QueryRequest
from app.pipeline.answer import NO_RESULTS_ANSWER, answer_question
from app.pipeline.embedder import EmbeddedChunk
from app.pipeline.chunker import Chunk
from app.vectorstore.base import RetrievedChunk, StoredDocument
from tests.conftest import FakeVectorStore


class StubStore:
    """Returns canned hits regardless of the query."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def query(self, embedding, top_k, filters):
        self.calls.append((top_k, filters))
        return list(self.hits)


class FakeClaude:
    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="the answer")])


def hit(distance, doc="d1"):
    return RetrievedChunk("text", doc, f"Title {doc}", "Sec", distance)


def run(store, embedder, claude, request=None, *, min_similarity=0.0, default_top_k=5):
    return answer_question(
        request or QueryRequest(question="q"),
        store, embedder, claude,
        embedding_model="voyage-3", claude_model="m",
        default_top_k=default_top_k, min_similarity=min_similarity,
    )


def test_embeds_with_query_input_type(fake_embedder):
    run(StubStore([]), fake_embedder, FakeClaude())
    assert fake_embedder.requests[0]["input_type"] == "query"
    assert fake_embedder.requests[0]["model"] == "voyage-3"


def test_returns_answer_and_sources(fake_embedder):
    claude = FakeClaude()
    result = run(StubStore([hit(0.1, "a"), hit(0.2, "b")]), fake_embedder, claude)
    assert result.answer == "the answer"
    assert [s.document_id for s in result.sources] == ["a", "b"]
    assert result.sources[0].distance == 0.1
    assert claude.calls == 1


def test_no_hits_skips_claude(fake_embedder):
    claude = FakeClaude()
    result = run(StubStore([]), fake_embedder, claude)
    assert result.answer == NO_RESULTS_ANSWER and result.sources == []
    assert claude.calls == 0


def test_similarity_cutoff_drops_weak_hits(fake_embedder):
    claude = FakeClaude()
    # similarity = 1 - distance: 0.9, 0.5, 0.2
    store = StubStore([hit(0.1, "a"), hit(0.5, "b"), hit(0.8, "c")])
    result = run(store, fake_embedder, claude, min_similarity=0.5)
    assert [s.document_id for s in result.sources] == ["a", "b"]  # boundary is inclusive
    assert claude.calls == 1


def test_all_hits_below_cutoff_skips_claude(fake_embedder):
    claude = FakeClaude()
    result = run(StubStore([hit(0.8), hit(0.9)]), fake_embedder, claude, min_similarity=0.5)
    assert result == run(StubStore([]), fake_embedder, FakeClaude())
    assert claude.calls == 0


def test_zero_cutoff_keeps_everything(fake_embedder):
    result = run(StubStore([hit(1.5)]), fake_embedder, FakeClaude(), min_similarity=0.0)
    assert len(result.sources) == 1


def test_top_k_request_overrides_default(fake_embedder):
    store = StubStore([])
    run(store, fake_embedder, FakeClaude(), QueryRequest(question="q", top_k=3), default_top_k=5)
    run(store, fake_embedder, FakeClaude(), default_top_k=5)
    assert [k for k, _ in store.calls] == [3, 5]


def test_filters_are_passed_through_normalized(fake_embedder):
    store = StubStore([])
    request = QueryRequest(
        question="q",
        filters=QueryFilters(category="project", featured=True, tags=[" Python ", "", "Python"]),
    )
    run(store, fake_embedder, FakeClaude(), request)
    _, filters = store.calls[0]
    assert (filters.category, filters.featured, filters.tags) == ("project", True, ["Python"])


def _doc(doc_id, tags, category="project", featured=False):
    return StoredDocument(doc_id, doc_id, category, "published", "2026-01-01", featured,
                          tags, "h", "m", 400)


def _seed(store, doc, vector):
    store.replace_document(doc, [EmbeddedChunk(Chunk("body", "Sec", 1), vector)])


@pytest.fixture
def seeded():
    store = FakeVectorStore()
    _seed(store, _doc("cpp", ["C++"]), [1.0, 0.0])
    _seed(store, _doc("csharp", ["C#"], category="blog", featured=True), [1.0, 0.1])
    _seed(store, _doc("both", ["C++", "C#", "Python"]), [1.0, 0.2])
    return store


class _Embedder:
    def embed(self, texts, model, input_type):
        return SimpleNamespace(embeddings=[[1.0, 0.0]])


def ids(store, **filters):
    request = QueryRequest(question="q", filters=QueryFilters(**filters))
    result = run(store, _Embedder(), FakeClaude(), request)
    return {s.document_id for s in result.sources}


def test_cpp_and_csharp_tags_stay_distinct(seeded):
    assert ids(seeded, tags=["C++"]) == {"cpp", "both"}
    assert ids(seeded, tags=["C#"]) == {"csharp", "both"}


def test_all_tags_must_match(seeded):
    assert ids(seeded, tags=["C++", "C#"]) == {"both"}
    assert ids(seeded, tags=["C++", "Rust"]) == set()


def test_category_and_featured_filters(seeded):
    assert ids(seeded, category="blog") == {"csharp"}
    assert ids(seeded, featured=True) == {"csharp"}
    assert ids(seeded, category="project", featured=False) == {"cpp", "both"}
    assert ids(seeded) == {"cpp", "csharp", "both"}


def test_filters_combine_with_tags(seeded):
    assert ids(seeded, category="project", tags=["C#"]) == {"both"}
