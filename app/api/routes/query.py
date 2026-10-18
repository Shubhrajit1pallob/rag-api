import logging

from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.deps import (
    get_answer_cache,
    get_claude_client,
    get_embedding_client,
    get_query_budget,
    get_vector_store,
    require_query_key,
)
from app import metrics
from app.cache.base import AnswerCache
from app.cache.key import build_cache_key
from app.config import Settings, get_settings
from app.generation.claude_client import ClaudeClient, build_system_prompt
from app.limits.base import QueryBudget, seconds_until_utc_midnight
from app.models.query import QueryRequest, QueryResponse
from app.pipeline.answer import answer_question
from app.pipeline.embedder import EmbeddingClient
from app.vectorstore.base import VectorStore

logger = logging.getLogger("rag_api")

router = APIRouter(dependencies=[Depends(require_query_key)])


# The cache only saves money; it must never be the reason a query fails. Any error is
# logged and treated as a miss (lookup) or ignored (store).
def _cache_get(cache: AnswerCache, key: str, ttl: int) -> QueryResponse | None:
    try:
        return cache.get(key, ttl)
    except Exception:
        metrics.cache_errors.labels("get").inc()
        logger.warning("Answer cache lookup failed; treating as a miss", exc_info=True)
        return None


def _cache_put(cache: AnswerCache, key: str, result: QueryResponse, ttl: int) -> None:
    try:
        cache.put(key, result, ttl)
    except Exception:
        metrics.cache_errors.labels("put").inc()
        logger.warning("Answer cache store failed; continuing without it", exc_info=True)


@router.post("/query", response_model=QueryResponse)
def post_query(
    request: QueryRequest,
    response: Response,
    store: VectorStore = Depends(get_vector_store),
    budget: QueryBudget = Depends(get_query_budget),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    claude: ClaudeClient = Depends(get_claude_client),
    settings: Settings = Depends(get_settings),
    cache: AnswerCache = Depends(get_answer_cache),
) -> QueryResponse:
    # Order matters: auth (router dependency) -> body validation -> length -> cache
    # lookup -> budget -> the paid pipeline. The budget is consumed here, not in a
    # dependency, so invalid requests are free, and cache hits (which cost no Voyage or
    # Claude call) do not use any of it.
    if len(request.question) > settings.max_question_chars:
        raise HTTPException(
            422, f"question exceeds the {settings.max_question_chars}-character limit"
        )
    system_prompt = build_system_prompt(settings.assistant_instructions)
    ttl = settings.answer_cache_ttl_seconds
    key = None
    if ttl > 0:
        key = build_cache_key(
            request,
            top_k=request.top_k or settings.retrieval_top_k,
            min_similarity=settings.min_similarity,
            embedding_model=settings.voyage_embedding_model,
            claude_model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            system_prompt=system_prompt,
        )
        cached = _cache_get(cache, key, ttl)
        if cached is not None:
            metrics.query_outcomes.labels("cache_hit").inc()
            response.headers["X-Cache"] = "HIT"
            return cached
    if not budget.try_consume(settings.daily_query_limit):
        metrics.query_outcomes.labels("budget_rejected").inc()
        raise HTTPException(
            429,
            "Daily query limit reached, try again after midnight UTC",
            headers={"Retry-After": str(seconds_until_utc_midnight())},
        )
    result = answer_question(
        request,
        store,
        embedding_client,
        claude,
        embedding_model=settings.voyage_embedding_model,
        claude_model=settings.claude_model,
        claude_max_tokens=settings.claude_max_tokens,
        system_prompt=system_prompt,
        default_top_k=settings.retrieval_top_k,
        min_similarity=settings.min_similarity,
    )
    if key is not None:
        metrics.query_outcomes.labels("cache_miss").inc()
        _cache_put(cache, key, result, ttl)
        response.headers["X-Cache"] = "MISS"
    else:
        metrics.query_outcomes.labels("cache_disabled").inc()
    return result
