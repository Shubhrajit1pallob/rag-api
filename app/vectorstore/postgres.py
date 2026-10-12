from psycopg import Connection
from psycopg_pool import ConnectionPool
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.pipeline.embedder import EmbeddedChunk
from app.vectorstore.base import RetrievedChunk, SearchFilters, StoredDocument

# tag_keys holds the lowercased tags so tag filters are case-insensitive while
# `tags` keeps the casing as sent. It is derived in SQL, so it is not part of
# StoredDocument and can never disagree with `tags`.
_TAG_KEYS_SQL = "array(select lower(t) from unnest(%s::text[]) as t)"

_DOCUMENT_COLUMNS = (
    "document_id, title, category, status, date, featured, tags, "
    "content_hash, embedding_model, max_chunk_size"
)


def _configure(conn: Connection) -> None:
    register_vector(conn)


def create_pool(database_url: str, min_size: int, max_size: int) -> ConnectionPool:
    """Build a pool that is opened by the caller; pgvector types are registered per connection."""
    return ConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        open=False,
    )


def _document_params(document: StoredDocument) -> tuple:
    return (
        document.document_id,
        document.title,
        document.category,
        document.status,
        document.date,
        document.featured,
        document.tags,
        document.content_hash,
        document.embedding_model,
        document.max_chunk_size,
    )


class PostgresVectorStore:
    def __init__(self, pool: ConnectionPool, embedding_dimensions: int = 1024):
        self._pool = pool
        self._dimensions = embedding_dimensions

    def _check_dimensions(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, expected {self._dimensions}"
            )

    def get_document(self, document_id: str) -> StoredDocument | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"select {_DOCUMENT_COLUMNS} from documents where document_id = %s",
                (document_id,),
            ).fetchone()
        return None if row is None else StoredDocument(*row)

    def count_document_chunks(self, document_id: str) -> int:
        with self._pool.connection() as conn:
            return self._count_chunks(conn, document_id)

    @staticmethod
    def _count_chunks(conn: Connection, document_id: str) -> int:
        row = conn.execute(
            "select count(*) from document_chunks where document_id = %s",
            (document_id,),
        ).fetchone()
        return row[0]

    def replace_document(
        self, document: StoredDocument, embedded_chunks: list[EmbeddedChunk]
    ) -> int:
        for ec in embedded_chunks:
            self._check_dimensions(ec.embedding)
        # The connection context manager wraps this block in one transaction:
        # commit on success, rollback on any error.
        with self._pool.connection() as conn:
            conn.execute(
                f"""
                insert into documents ({_DOCUMENT_COLUMNS}, tag_keys)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {_TAG_KEYS_SQL})
                on conflict (document_id) do update set
                    title = excluded.title,
                    category = excluded.category,
                    status = excluded.status,
                    date = excluded.date,
                    featured = excluded.featured,
                    tags = excluded.tags,
                    tag_keys = excluded.tag_keys,
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    max_chunk_size = excluded.max_chunk_size,
                    updated_at = now()
                """,
                (*_document_params(document), document.tags),
            )
            conn.execute(
                "delete from document_chunks where document_id = %s",
                (document.document_id,),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into document_chunks
                        (document_id, chunk_index, section_title, content, embedding)
                    values (%s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            document.document_id,
                            i,
                            ec.chunk.title,
                            ec.chunk.text,
                            Vector(ec.embedding),
                        )
                        for i, ec in enumerate(embedded_chunks)
                    ],
                )
        return len(embedded_chunks)

    def update_document_metadata(self, document: StoredDocument) -> int:
        with self._pool.connection() as conn:
            conn.execute(
                f"""
                update documents set
                    title = %s, category = %s, status = %s, date = %s,
                    featured = %s, tags = %s, tag_keys = {_TAG_KEYS_SQL}, content_hash = %s,
                    embedding_model = %s, max_chunk_size = %s, updated_at = now()
                where document_id = %s
                """,
                (
                    document.title,
                    document.category,
                    document.status,
                    document.date,
                    document.featured,
                    document.tags,
                    document.tags,
                    document.content_hash,
                    document.embedding_model,
                    document.max_chunk_size,
                    document.document_id,
                ),
            )
            return self._count_chunks(conn, document.document_id)

    def delete_document(self, document_id: str) -> int | None:
        with self._pool.connection() as conn:
            count = self._count_chunks(conn, document_id)
            deleted = conn.execute(
                "delete from documents where document_id = %s", (document_id,)
            ).rowcount
        return count if deleted else None

    def query(
        self, embedding: list[float], top_k: int, filters: SearchFilters
    ) -> list[RetrievedChunk]:
        self._check_dimensions(embedding)
        conditions: list[str] = []
        params: list[object] = [Vector(embedding)]
        if filters.category is not None:
            conditions.append("d.category = %s")
            params.append(filters.category)
        if filters.featured is not None:
            conditions.append("d.featured = %s")
            params.append(filters.featured)
        if filters.tags:
            conditions.append("d.tag_keys @> %s::text[]")
            params.append([tag.lower() for tag in filters.tags])
        where = f"where {' and '.join(conditions)}" if conditions else ""
        params.append(top_k)

        sql = f"""
            select c.content, d.document_id, d.title, c.section_title,
                   c.embedding <=> %s::vector as distance
            from document_chunks c
            join documents d on d.document_id = c.document_id
            {where}
            order by distance
            limit %s
        """
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [RetrievedChunk(*row) for row in rows]
