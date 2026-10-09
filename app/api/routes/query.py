from fastapi import APIRouter, Depends

from app.api.deps import (
    get_claude_client,
    get_embedding_client,
    get_vector_store,
    require_api_key,
)
from app.config import Settings, get_settings
from app.generation.claude_client import ClaudeClient
from app.models.query import QueryRequest, QueryResponse
from app.pipeline.answer import answer_question
from app.pipeline.embedder import EmbeddingClient
from app.vectorstore.base import VectorStore

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResponse)
def post_query(
    request: QueryRequest,
    store: VectorStore = Depends(get_vector_store),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    claude: ClaudeClient = Depends(get_claude_client),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    return answer_question(
        request,
        store,
        embedding_client,
        claude,
        embedding_model=settings.voyage_embedding_model,
        claude_model=settings.claude_model,
        default_top_k=settings.retrieval_top_k,
        min_similarity=settings.min_similarity,
    )
