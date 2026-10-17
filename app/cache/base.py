from typing import Protocol

from app.models.query import QueryResponse


class AnswerCache(Protocol):
    def get(self, key: str, ttl_seconds: int) -> QueryResponse | None:
        """The stored response for `key`, or None if absent or older than `ttl_seconds`."""
        ...

    def put(self, key: str, response: QueryResponse, ttl_seconds: int) -> None:
        """Store `response` under `key` (replacing any entry) and drop entries older than the TTL."""
        ...
