import os
from dataclasses import dataclass
from typing import Any, Protocol

import voyageai

from app.pipeline.chunker import Chunk

DEFAULT_EMBEDDING_MODEL = os.environ.get("VOYAGE_EMBEDDING_MODEL", "voyage-3")


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
    model: str = DEFAULT_EMBEDDING_MODEL,
    input_type: str = "document",
    client: EmbeddingClient | None = None,
) -> list[EmbeddedChunk]:
    """Embed each chunk's text via Voyage AI.

    input_type must be "document" when embedding content for storage and "query"
    when embedding a user's question at retrieval time -- Voyage's model treats
    the two asymmetrically, so mixing them up degrades retrieval quality. client
    defaults to a real voyageai.Client() but can be swapped for a fake in tests
    so embedding tests don't need a real API key or network access.
    """
    if not chunks:
        return []

    if client is None:
        client = voyageai.Client()

    texts = [chunk.text for chunk in chunks]
    result = client.embed(texts, model=model, input_type=input_type)

    return [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, result.embeddings)
    ]
