from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rag_query_api_key: str = Field(min_length=1)
    rag_admin_api_key: str = Field(min_length=1)
    voyage_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_embedding_model: str = "voyage-3"
    claude_model: str = "claude-haiku-4-5-20251001"
    database_url: str = Field(min_length=1)
    embedding_dimensions: int = 1024
    db_pool_min_size: int = 1
    db_pool_max_size: int = 5
    max_chunk_size: int = 400
    retrieval_top_k: int = 5
    min_similarity: float = 0.0  # 0 means off
    daily_query_limit: int = Field(default=300, ge=1)
    claude_max_tokens: int = 400
    max_question_chars: int = 500
    upstream_timeout_seconds: float = 20.0
    upstream_max_retries: int = 1

    @model_validator(mode="after")
    def _keys_must_differ(self) -> "Settings":
        if self.rag_query_api_key == self.rag_admin_api_key:
            raise ValueError("RAG_QUERY_API_KEY and RAG_ADMIN_API_KEY must be different")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
