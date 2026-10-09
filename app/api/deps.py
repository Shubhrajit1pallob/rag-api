import secrets
from functools import lru_cache

import anthropic
import voyageai
from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.generation import claude_client
from app.vectorstore.postgres import PostgresVectorStore, create_pool


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode(), settings.rag_api_key.encode()
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


@lru_cache
def get_vector_store() -> PostgresVectorStore:
    settings = get_settings()
    pool = create_pool(
        settings.database_url, settings.db_pool_min_size, settings.db_pool_max_size
    )
    pool.open(wait=True, timeout=10)
    return PostgresVectorStore(pool, settings.embedding_dimensions)


@lru_cache
def get_embedding_client() -> voyageai.Client:
    return voyageai.Client(api_key=get_settings().voyage_api_key)


@lru_cache
def get_claude_client() -> anthropic.Anthropic:
    return claude_client.get_claude_client(get_settings())
