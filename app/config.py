from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rag_api_key: str = Field(min_length=1)
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
