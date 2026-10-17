from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.models.query import QueryResponse

_GET_SQL = """
select answer, sources from answer_cache
where key = %s and created_at > now() - make_interval(secs => %s)
"""

_PRUNE_SQL = "delete from answer_cache where created_at <= now() - make_interval(secs => %s)"

_PUT_SQL = """
insert into answer_cache (key, answer, sources) values (%s, %s, %s)
on conflict (key) do update set
    answer = excluded.answer, sources = excluded.sources, created_at = now()
"""


class PostgresAnswerCache:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def get(self, key: str, ttl_seconds: int) -> QueryResponse | None:
        with self._pool.connection() as conn:
            row = conn.execute(_GET_SQL, (key, ttl_seconds)).fetchone()
        if row is None:
            return None
        answer, sources = row
        return QueryResponse.model_validate({"answer": answer, "sources": sources})

    def put(self, key: str, response: QueryResponse, ttl_seconds: int) -> None:
        sources = response.model_dump()["sources"]
        # One transaction: expired rows are pruned on every store, which keeps the table small.
        with self._pool.connection() as conn:
            conn.execute(_PRUNE_SQL, (ttl_seconds,))
            conn.execute(_PUT_SQL, (key, response.answer, Jsonb(sources)))
