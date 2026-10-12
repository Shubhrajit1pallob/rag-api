from psycopg_pool import ConnectionPool

# The WHERE on the conflict branch makes the upsert atomic: once the day's count
# reaches the limit no row is updated or returned, which means "rejected".
_CONSUME_SQL = """
insert into query_usage (day, count) values ((now() at time zone 'utc')::date, 1)
on conflict (day) do update set count = query_usage.count + 1
  where query_usage.count < %s
returning count
"""


class PostgresQueryBudget:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def try_consume(self, limit: int) -> bool:
        # limit >= 1 (enforced by Settings), so the first insert of a day always fits.
        with self._pool.connection() as conn:
            return conn.execute(_CONSUME_SQL, (limit,)).fetchone() is not None
