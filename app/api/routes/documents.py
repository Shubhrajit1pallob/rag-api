from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import (
    get_embedding_client,
    get_vector_store,
    require_admin_key,
)
from app.config import Settings, get_settings
from app.models.document import (
    DocumentDeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
)
from app.pipeline.embedder import EmbeddingClient
from app.pipeline.ingest import EmptyDocumentError, ingest_document
from app.vectorstore.base import VectorStore

router = APIRouter(dependencies=[Depends(require_admin_key)])


@router.post("/documents", response_model=DocumentIngestResponse)
def post_document(
    request: DocumentIngestRequest,
    store: VectorStore = Depends(get_vector_store),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    settings: Settings = Depends(get_settings),
) -> DocumentIngestResponse:
    try:
        return ingest_document(
            request,
            store,
            embedding_client,
            embedding_model=settings.voyage_embedding_model,
            max_chunk_size=settings.max_chunk_size,
        )
    except EmptyDocumentError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    store: VectorStore = Depends(get_vector_store),
) -> DocumentDeleteResponse:
    deleted = store.delete_document(document_id)
    if deleted is None:
        raise HTTPException(404, "Document not found")
    return DocumentDeleteResponse(document_id=document_id, deleted_chunks=deleted)
