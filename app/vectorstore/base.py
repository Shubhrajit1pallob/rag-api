from dataclasses import dataclass, field
from typing import Protocol

from app.pipeline.embedder import EmbeddedChunk


@dataclass
class StoredDocument:
    document_id: str
    title: str
    category: str
    status: str
    date: str
    featured: bool
    tags: list[str]
    content_hash: str
    embedding_model: str
    max_chunk_size: int
    chunker_version: int


@dataclass
class RetrievedChunk:
    text: str
    document_id: str
    title: str
    section_title: str
    distance: float


@dataclass
class SearchFilters:
    category: str | None = None
    featured: bool | None = None
    tags: list[str] = field(default_factory=list)  # all must match


class VectorStore(Protocol):
    def get_document(self, document_id: str) -> StoredDocument | None: ...

    def count_document_chunks(self, document_id: str) -> int: ...

    def replace_document(
        self, document: StoredDocument, embedded_chunks: list[EmbeddedChunk]
    ) -> int:
        """Upsert the document row and swap its chunks, atomically. Returns the chunk count."""
        ...

    def update_document_metadata(self, document: StoredDocument) -> int:
        """Update only the document row. Returns the chunk count."""
        ...

    def delete_document(self, document_id: str) -> int | None:
        """Returns the number of chunks deleted, or None if the document didn't exist."""
        ...

    def query(
        self, embedding: list[float], top_k: int, filters: SearchFilters
    ) -> list[RetrievedChunk]: ...
