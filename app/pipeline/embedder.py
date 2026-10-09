from dataclasses import dataclass
from typing import Any, Protocol

from app.pipeline.chunker import Chunk


class EmbeddingClient(Protocol):
    """Anything shaped like voyageai.Client's .embed() method.

    The return type is left as Any rather than the SDK's actual response class,
    since this code only relies on the response having an .embeddings attribute
    (a list of float vectors) -- the exact class path isn't worth pinning down.
    """

    def embed(self, texts: list[str], model: str, input_type: str) -> Any: ...


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector, ready to upsert into the vector store."""

    chunk: Chunk
    embedding: list[float]


def embed_chunks(
    chunks: list[Chunk],
    *,
    model: str,
    client: EmbeddingClient,
    input_type: str = "document",
) -> list[EmbeddedChunk]:
    """Embed each chunk's text via Voyage AI.

    input_type must be "document" when embedding content for storage and "query"
    when embedding a user's question at retrieval time -- Voyage's model treats
    the two asymmetrically, so mixing them up degrades retrieval quality. client
    is required so callers (and tests) choose it explicitly -- there is no hidden
    fallback to a real voyageai.Client().
    """
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    result = client.embed(texts, model=model, input_type=input_type)

    return [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, result.embeddings)
    ]


def embed_query(
    text: str,
    *,
    model: str,
    client: EmbeddingClient,
) -> list[float]:
    """Embed a user's question with input_type="query" for retrieval."""
    result = client.embed([text], model=model, input_type="query")
    return result.embeddings[0]
