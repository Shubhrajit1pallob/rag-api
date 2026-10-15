from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import (
    get_claude_client,
    get_embedding_client,
    get_query_budget,
    get_vector_store,
    require_query_key,
)
from app.config import Settings, get_settings
from app.generation.claude_client import ClaudeClient, build_system_prompt
from app.limits.base import QueryBudget, seconds_until_utc_midnight
from app.models.query import QueryRequest, QueryResponse
from app.pipeline.answer import answer_question
from app.pipeline.embedder import EmbeddingClient
from app.vectorstore.base import VectorStore

router = APIRouter(dependencies=[Depends(require_query_key)])


@router.post("/query", response_model=QueryResponse)
def post_query(
    request: QueryRequest,
    store: VectorStore = Depends(get_vector_store),
    budget: QueryBudget = Depends(get_query_budget),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    claude: ClaudeClient = Depends(get_claude_client),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    # Order matters: auth (router dependency) -> body validation -> these checks.
    # The budget is consumed here, not in a dependency, so invalid requests are free.
    if len(request.question) > settings.max_question_chars:
        raise HTTPException(
            422, f"question exceeds the {settings.max_question_chars}-character limit"
        )
    if not budget.try_consume(settings.daily_query_limit):
        raise HTTPException(
            429,
            "Daily query limit reached, try again after midnight UTC",
            headers={"Retry-After": str(seconds_until_utc_midnight())},
        )
    return answer_question(
        request,
        store,
        embedding_client,
        claude,
        embedding_model=settings.voyage_embedding_model,
        claude_model=settings.claude_model,
        claude_max_tokens=settings.claude_max_tokens,
        system_prompt=build_system_prompt(settings.assistant_instructions),
        default_top_k=settings.retrieval_top_k,
        min_similarity=settings.min_similarity,
    )
