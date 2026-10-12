import math

import pytest

from app.pipeline.embedder import EmbeddedChunk
from app.vectorstore.base import RetrievedChunk, SearchFilters, StoredDocument


class FakeEmbeddingClient:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = 0
        self.requests: list[dict] = []

    def embed(self, texts, model, input_type):
        self.calls += 1
        self.requests.append({"texts": texts, "model": model, "input_type": input_type})
        vectors = [
            [float((hash(t) >> i) % 7 + 1) for i in range(self.dim)] for t in texts
        ]
        return type("Result", (), {"embeddings": vectors})()


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1 - dot / norm


class FakeVectorStore:
    """In-memory VectorStore. Filter semantics mirror PostgresVectorStore's SQL."""

    def __init__(self):
        self.documents: dict[str, StoredDocument] = {}
        self.chunks: dict[str, list[EmbeddedChunk]] = {}
        self.queries: list[tuple[int, SearchFilters]] = []

    def get_document(self, document_id):
        doc = self.documents.get(document_id)
        return None if doc is None else StoredDocument(**{**doc.__dict__, "tags": list(doc.tags)})

    def count_document_chunks(self, document_id):
        return len(self.chunks.get(document_id, []))

    def replace_document(self, document, embedded_chunks):
        self.documents[document.document_id] = document
        self.chunks[document.document_id] = list(embedded_chunks)
        return len(embedded_chunks)

    def update_document_metadata(self, document):
        self.documents[document.document_id] = document
        return len(self.chunks[document.document_id])

    def delete_document(self, document_id):
        if document_id not in self.documents:
            return None
        del self.documents[document_id]
        return len(self.chunks.pop(document_id))

    def query(self, embedding, top_k, filters):
        self.queries.append((top_k, filters))
        hits = []
        for doc_id, doc in self.documents.items():
            if filters.category is not None and doc.category != filters.category:
                continue
            if filters.featured is not None and doc.featured != filters.featured:
                continue
            # Mirrors tag_keys @> lower(filter tags): case-insensitive containment.
            if not {t.lower() for t in filters.tags} <= {t.lower() for t in doc.tags}:
                continue
            for ec in self.chunks[doc_id]:
                hits.append(
                    RetrievedChunk(
                        text=ec.chunk.text,
                        document_id=doc_id,
                        title=doc.title,
                        section_title=ec.chunk.title,
                        distance=_cosine_distance(embedding, ec.embedding),
                    )
                )
        return sorted(hits, key=lambda h: h.distance)[:top_k]


class FakeQueryBudget:
    """In-memory QueryBudget with the same contract as the SQL upsert."""

    def __init__(self, used: int = 0):
        self.used = used
        self.limits_seen: list[int] = []

    def try_consume(self, limit):
        self.limits_seen.append(limit)
        if self.used >= limit:
            return False
        self.used += 1
        return True


@pytest.fixture
def store():
    return FakeVectorStore()


@pytest.fixture
def budget():
    return FakeQueryBudget()


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingClient()
