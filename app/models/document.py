from typing import Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Facts about a document, kept separate from its embeddable text --
    mirrors the generic ingestion contract (no portfolio-specific fields)."""

    title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    featured: bool = False
    status: str
    date: str  # consumer's raw date string; normalized downstream in the pipeline


class DocumentIngestRequest(BaseModel):
    """Request body for POST /documents."""

    document_id: str  # stable key used for idempotency/upserts
    content: str  # markdown text with ## headers; the chunker splits on this
    metadata: DocumentMetadata


def normalize_tags(tags: list[str]) -> list[str]:
    """Trim, drop empties and duplicates; keep first-seen order and original casing."""
    seen: dict[str, None] = {}
    for tag in tags:
        stripped = tag.strip()
        if stripped:
            seen.setdefault(stripped)
    return list(seen)


class DocumentIngestResponse(BaseModel):
    """Response body for POST /documents."""

    document_id: str
    status: Literal["created", "content_updated", "metadata_updated", "unchanged"]
    chunk_count: int


class DocumentDeleteResponse(BaseModel):
    """Response body for DELETE /documents/{document_id}."""

    document_id: str
    deleted_chunks: int
