from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings, get_settings
from app.main import app

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
BODY = {
    "document_id": "d1",
    "content": "## Intro\nCaching speeds things up.",
    "metadata": {
        "title": "Cache Doc", "category": "project", "tags": ["Python"],
        "featured": True, "status": "published", "date": "2026-01-01",
    },
}


class FakeClaude:
    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="fake answer")])


@pytest.fixture
def claude():
    return FakeClaude()


@pytest.fixture
def client(store, fake_embedder, claude):
    app.dependency_overrides[get_settings] = lambda: Settings(rag_api_key=KEY, database_url="postgresql://unused")
    app.dependency_overrides[deps.get_vector_store] = lambda: store
    app.dependency_overrides[deps.get_embedding_client] = lambda: fake_embedder
    app.dependency_overrides[deps.get_claude_client] = lambda: claude
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_auth_rejected(client, headers):
    assert client.post("/documents", json=BODY, headers=headers).status_code == 401
    assert client.post("/query", json={"question": "q"}, headers=headers).status_code == 401
    assert client.delete("/documents/d1", headers=headers).status_code == 401


def test_documents_created_then_unchanged(client):
    first = client.post("/documents", json=BODY, headers=HEADERS)
    assert first.status_code == 200
    assert first.json()["status"] == "created"
    second = client.post("/documents", json=BODY, headers=HEADERS)
    assert second.json()["status"] == "unchanged"


def test_query_happy_path(client, claude):
    client.post("/documents", json=BODY, headers=HEADERS)
    resp = client.post(
        "/query",
        json={"question": "What about caching?", "filters": {"category": "project", "tags": ["Python"]}},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "fake answer"
    assert data["sources"][0]["document_id"] == "d1"
    assert data["sources"][0]["title"] == "Cache Doc"
    assert claude.calls == 1


def test_query_with_nothing_indexed_skips_claude(client, claude):
    resp = client.post("/query", json={"question": "anything"}, headers=HEADERS)
    assert resp.json() == {
        "answer": "I couldn't find anything relevant to that question.",
        "sources": [],
    }
    assert claude.calls == 0


def test_empty_content_is_422(client):
    body = {**BODY, "content": "   "}
    assert client.post("/documents", json=body, headers=HEADERS).status_code == 422


def test_empty_api_key_setting_is_rejected():
    with pytest.raises(ValidationError):
        Settings(rag_api_key="")


def test_empty_api_key_header_is_401(client):
    resp = client.post("/documents", json=BODY, headers={"X-API-Key": ""})
    assert resp.status_code == 401


def test_delete_document_returns_chunk_count_then_404(client, store):
    client.post("/documents", json=BODY, headers=HEADERS)
    expected = store.count_document_chunks("d1")
    resp = client.delete("/documents/d1", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"document_id": "d1", "deleted_chunks": expected}
    assert store.get_document("d1") is None
    assert client.delete("/documents/d1", headers=HEADERS).status_code == 404


def test_delete_unknown_document_is_404(client):
    assert client.delete("/documents/nope", headers=HEADERS).status_code == 404


def test_delete_makes_no_embedding_or_claude_calls(client, fake_embedder, claude):
    client.post("/documents", json=BODY, headers=HEADERS)
    calls = fake_embedder.calls
    client.delete("/documents/d1", headers=HEADERS)
    assert fake_embedder.calls == calls
    assert claude.calls == 0


def test_query_filter_excludes_non_matching_documents(client, claude):
    client.post("/documents", json=BODY, headers=HEADERS)
    resp = client.post(
        "/query",
        json={"question": "caching", "filters": {"category": "blog"}},
        headers=HEADERS,
    )
    assert resp.json()["sources"] == []
    assert claude.calls == 0
