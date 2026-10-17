import secrets
from functools import lru_cache

import anthropic
import voyageai
from fastapi import Depends, Header, HTTPException, status
from psycopg_pool import ConnectionPool

from app.cache.postgres import PostgresAnswerCache
from app.config import Settings, get_settings
from app.generation import claude_client
from app.limits.postgres import PostgresQueryBudget
from app.vectorstore.postgres import PostgresVectorStore, create_pool


def _matches_any(provided: str | None, *expected: str) -> bool:
    if provided is None:
        return False
    # Compare against every key (list, not a generator) so timing doesn't reveal
    # which key, if any, matched first.
    results = [
        secrets.compare_digest(provided.encode(), key.encode()) for key in expected
    ]
    return any(results)


def _unauthorized() -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


def require_admin_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Admin-only routes (ingest, delete)."""
    if not _matches_any(x_api_key, settings.rag_admin_api_key):
        raise _unauthorized()


def require_query_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Query routes: the query key or the admin key."""
    if not _matches_any(
        x_api_key, settings.rag_query_api_key, settings.rag_admin_api_key
    ):
        raise _unauthorized()


@lru_cache
def get_pool() -> ConnectionPool:
    settings = get_settings()
    pool = create_pool(
        settings.database_url, settings.db_pool_min_size, settings.db_pool_max_size
    )
    # Don't wait for the first connection: if the database is paused or down, the
    # pool keeps retrying in the background and requests fail with PoolTimeout (503).
    pool.open(wait=False)
    return pool


@lru_cache
def get_vector_store() -> PostgresVectorStore:
    return PostgresVectorStore(get_pool(), get_settings().embedding_dimensions)


@lru_cache
def get_query_budget() -> PostgresQueryBudget:
    return PostgresQueryBudget(get_pool())


@lru_cache
def get_answer_cache() -> PostgresAnswerCache:
    return PostgresAnswerCache(get_pool())


@lru_cache
def get_embedding_client() -> voyageai.Client:
    settings = get_settings()
    return voyageai.Client(
        api_key=settings.voyage_api_key,
        timeout=settings.upstream_timeout_seconds,
        max_retries=settings.upstream_max_retries,
    )


@lru_cache
def get_claude_client() -> anthropic.Anthropic:
    return claude_client.get_claude_client(get_settings())
