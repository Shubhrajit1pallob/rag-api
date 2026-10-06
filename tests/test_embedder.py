from app.pipeline.chunker import Chunk
from app.pipeline.embedder import DEFAULT_EMBEDDING_MODEL, EmbeddedChunk, embed_chunks


class _FakeEmbeddingsResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeClient:
    """Stands in for voyageai.Client so tests need no network access or API key."""

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self.calls = []

    def embed(self, texts, model, input_type):
        self.calls.append({"texts": texts, "model": model, "input_type": input_type})
        return _FakeEmbeddingsResult(self._embeddings)


def test_empty_chunks_returns_empty_list_without_calling_client():
    client = _FakeClient(embeddings=[])
    assert embed_chunks([], client=client) == []
    assert client.calls == []


def test_embeds_each_chunk_and_pairs_with_its_vector():
    chunks = [
        Chunk(text="first chunk", title="A", level=1),
        Chunk(text="second chunk", title="B", level=1),
    ]
    client = _FakeClient(embeddings=[[0.1, 0.2], [0.3, 0.4]])

    result = embed_chunks(chunks, client=client)

    assert result == [
        EmbeddedChunk(chunk=chunks[0], embedding=[0.1, 0.2]),
        EmbeddedChunk(chunk=chunks[1], embedding=[0.3, 0.4]),
    ]


def test_sends_chunk_text_to_the_client():
    chunks = [Chunk(text="hello world", title="A", level=1)]
    client = _FakeClient(embeddings=[[0.0]])

    embed_chunks(chunks, client=client)

    assert client.calls[0]["texts"] == ["hello world"]


def test_default_input_type_is_document():
    chunks = [Chunk(text="hello", title="A", level=1)]
    client = _FakeClient(embeddings=[[0.0]])

    embed_chunks(chunks, client=client)

    assert client.calls[0]["input_type"] == "document"


def test_input_type_can_be_overridden_for_query_embedding():
    chunks = [Chunk(text="hello", title="A", level=1)]
    client = _FakeClient(embeddings=[[0.0]])

    embed_chunks(chunks, client=client, input_type="query")

    assert client.calls[0]["input_type"] == "query"


def test_default_model_is_the_module_default():
    chunks = [Chunk(text="hello", title="A", level=1)]
    client = _FakeClient(embeddings=[[0.0]])

    embed_chunks(chunks, client=client)

    assert client.calls[0]["model"] == DEFAULT_EMBEDDING_MODEL


def test_model_can_be_overridden():
    chunks = [Chunk(text="hello", title="A", level=1)]
    client = _FakeClient(embeddings=[[0.0]])

    embed_chunks(chunks, client=client, model="voyage-3-lite")

    assert client.calls[0]["model"] == "voyage-3-lite"
