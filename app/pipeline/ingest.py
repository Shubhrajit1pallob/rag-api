import hashlib

from app.models.document import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    normalize_tags,
)
from app.pipeline.chunker import chunk_sections
from app.pipeline.cleaner import clean_sections
from app.pipeline.embedder import EmbeddingClient, embed_chunks
from app.pipeline.parser import parse_markdown
from app.vectorstore.base import StoredDocument, VectorStore


class EmptyDocumentError(ValueError):
    pass


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_stored_document(
    request: DocumentIngestRequest,
    content_hash: str,
    embedding_model: str,
    max_chunk_size: int,
) -> StoredDocument:
    meta = request.metadata
    return StoredDocument(
        document_id=request.document_id,
        title=meta.title,
        category=meta.category,
        status=meta.status,
        date=meta.date,
        featured=meta.featured,
        tags=normalize_tags(meta.tags),
        content_hash=content_hash,
        embedding_model=embedding_model,
        max_chunk_size=max_chunk_size,
    )


def ingest_document(
    request: DocumentIngestRequest,
    store: VectorStore,
    embedding_client: EmbeddingClient,
    *,
    embedding_model: str,
    max_chunk_size: int,
) -> DocumentIngestResponse:
    document = build_stored_document(
        request, compute_content_hash(request.content), embedding_model, max_chunk_size
    )
    stored = store.get_document(request.document_id)

    # Content, embedding model and chunk size together decide what the chunks
    # and vectors look like; if all match, only metadata can have changed.
    if (
        stored is not None
        and stored.content_hash == document.content_hash
        and stored.embedding_model == document.embedding_model
        and stored.max_chunk_size == document.max_chunk_size
    ):
        if stored == document:
            return DocumentIngestResponse(
                document_id=request.document_id,
                status="unchanged",
                chunk_count=store.count_document_chunks(request.document_id),
            )
        count = store.update_document_metadata(document)
        return DocumentIngestResponse(
            document_id=request.document_id,
            status="metadata_updated",
            chunk_count=count,
        )

    chunks = chunk_sections(
        clean_sections(parse_markdown(request.content)), max_chunk_size
    )
    if not chunks:
        raise EmptyDocumentError("document has no indexable content")

    embedded = embed_chunks(chunks, model=embedding_model, client=embedding_client)
    count = store.replace_document(document, embedded)
    return DocumentIngestResponse(
        document_id=request.document_id,
        status="created" if stored is None else "content_updated",
        chunk_count=count,
    )
