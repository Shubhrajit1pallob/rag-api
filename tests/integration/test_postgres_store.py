"""Exercises PostgresVectorStore against a real Postgres with pgvector.

Skipped unless TEST_DATABASE_URL is set. Point it at a DISPOSABLE database: the
fixture drops and recreates the rag-api tables in the public schema. The
connecting role's search_path must include `extensions` (as the production
rag_api role's does) so the `vector` type resolves.
"""

import os
from pathlib import Path

import pytest

from app.cache.postgres import PostgresAnswerCache
from app.models.query import QueryResponse, Source
from app.pipeline.chunker import Chunk
from app.pipeline.embedder import EmbeddedChunk
from app.vectorstore.base import SearchFilters, StoredDocument

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
# CI sets REQUIRE_TEST_DB=1 so a missing database fails the job instead of skipping
# every test and going green. Local runs leave it unset and skip without a database.
if os.environ.get("REQUIRE_TEST_DB") == "1" and not DATABASE_URL:
    raise RuntimeError("REQUIRE_TEST_DB=1 but TEST_DATABASE_URL is not set")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
DIM = 1024


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (DIM - len(head))


def doc(doc_id="d1", **overrides) -> StoredDocument:
    fields = dict(
        document_id=doc_id, title=f"Title {doc_id}", category="project",
        status="published", date="2026-01-01", featured=False, tags=["Python"],
        content_hash="h1", embedding_model="voyage-3", max_chunk_size=400,
        chunker_version=2,
    )
    fields.update(overrides)
    return StoredDocument(**fields)


def chunks(*vectors: list[float]) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(Chunk(text=f"text {i}", title=f"Sec {i}", level=1), v)
        for i, v in enumerate(vectors)
    ]


@pytest.fixture
def store():
    import psycopg

    from app.vectorstore.postgres import PostgresVectorStore, create_pool

    # A plain connection applies the schema: the pool's register_vector needs the
    # extension to exist already. Supabase ships an `extensions` schema; plain
    # Postgres needs it created first.
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("create schema if not exists extensions")
        conn.execute("drop table if exists public.document_chunks")
        conn.execute("drop table if exists public.documents")
        conn.execute("drop table if exists public.query_usage")
        conn.execute("drop table if exists public.answer_cache")
        conn.execute(SCHEMA.read_text())

    pool = create_pool(DATABASE_URL, 1, 3)
    pool.open(wait=True, timeout=10)
    yield PostgresVectorStore(pool, DIM)
    pool.close()


def test_replace_and_get_roundtrip(store):
    assert store.get_document("d1") is None
    assert store.replace_document(doc(tags=["C++", "C#"]), chunks(vec(1), vec(0, 1))) == 2
    assert store.get_document("d1") == doc(tags=["C++", "C#"])
    assert store.count_document_chunks("d1") == 2


def test_replace_swaps_chunks(store):
    store.replace_document(doc(), chunks(vec(1), vec(0, 1), vec(0, 0, 1)))
    assert store.replace_document(doc(content_hash="h2"), chunks(vec(1))) == 1
    assert store.count_document_chunks("d1") == 1
    assert store.get_document("d1").content_hash == "h2"


def test_update_metadata_keeps_chunks(store):
    store.replace_document(doc(), chunks(vec(1), vec(0, 1)))
    assert store.update_document_metadata(doc(title="Renamed", tags=["Go"])) == 2
    stored = store.get_document("d1")
    assert (stored.title, stored.tags) == ("Renamed", ["Go"])


def test_delete_cascades_and_reports(store):
    store.replace_document(doc(), chunks(vec(1), vec(0, 1)))
    assert store.delete_document("d1") == 2
    assert store.get_document("d1") is None
    assert store.count_document_chunks("d1") == 0
    assert store.delete_document("d1") is None


def test_query_orders_by_cosine_distance(store):
    store.replace_document(doc(), chunks(vec(1), vec(1, 1), vec(0, 1)))
    hits = store.query(vec(1), 3, SearchFilters())
    assert [h.text for h in hits] == ["text 0", "text 1", "text 2"]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[2].distance == pytest.approx(1.0, abs=1e-6)
    assert (hits[0].document_id, hits[0].title, hits[0].section_title) == ("d1", "Title d1", "Sec 0")
    assert len(store.query(vec(1), 2, SearchFilters())) == 2


def test_query_filters(store):
    store.replace_document(doc("cpp", tags=["C++"]), chunks(vec(1)))
    store.replace_document(doc("cs", tags=["C#"], category="blog", featured=True), chunks(vec(1)))
    store.replace_document(doc("both", tags=["C++", "C#"]), chunks(vec(1)))

    def ids(**kw):
        return {h.document_id for h in store.query(vec(1), 10, SearchFilters(**kw))}

    assert ids() == {"cpp", "cs", "both"}
    assert ids(tags=["C++"]) == {"cpp", "both"}
    assert ids(tags=["C#"]) == {"cs", "both"}
    assert ids(tags=["C++", "C#"]) == {"both"}
    assert ids(category="blog") == {"cs"}
    assert ids(featured=True) == {"cs"}
    assert ids(category="project", tags=["C#"]) == {"both"}


def test_tag_filter_is_case_insensitive_and_tags_keep_their_casing(store):
    store.replace_document(doc("py", tags=["Python", "C++"]), chunks(vec(1)))
    store.replace_document(doc("cs", tags=["C#"]), chunks(vec(1)))

    def ids(*tags):
        return {h.document_id for h in store.query(vec(1), 10, SearchFilters(tags=list(tags)))}

    assert ids("python") == {"py"}
    assert ids("PYTHON", "c++") == {"py"}
    assert ids("c++") == {"py"} and ids("c#") == {"cs"}  # C++ and C# stay distinct
    assert ids("python", "c#") == set()
    assert store.get_document("py").tags == ["Python", "C++"]


def test_update_metadata_refreshes_tag_keys(store):
    store.replace_document(doc(tags=["Python"]), chunks(vec(1)))
    store.update_document_metadata(doc(tags=["Go"]))
    assert store.query(vec(1), 10, SearchFilters(tags=["python"])) == []
    assert len(store.query(vec(1), 10, SearchFilters(tags=["GO"]))) == 1


def test_query_budget_counts_atomically_and_rejects_at_limit(store):
    from app.limits.postgres import PostgresQueryBudget

    with store._pool.connection() as conn:
        conn.execute("delete from query_usage")
    budget = PostgresQueryBudget(store._pool)
    assert [budget.try_consume(2) for _ in range(4)] == [True, True, False, False]
    with store._pool.connection() as conn:
        assert conn.execute("select count from query_usage").fetchone()[0] == 2  # rejects don't count
    assert budget.try_consume(3) is True  # raising the limit frees headroom
    assert budget.try_consume(3) is False


def test_wrong_dimension_is_rejected(store):
    with pytest.raises(ValueError):
        store.query([1.0, 2.0], 3, SearchFilters())


# --- answer cache -------------------------------------------------------------


@pytest.fixture
def cache(store):
    return PostgresAnswerCache(store._pool)


def cached_response(text="an answer") -> QueryResponse:
    source = Source(document_id="d1", title="T", section_title="S", distance=0.25)
    return QueryResponse(answer=text, sources=[source])


def backdate(store, key, interval="2 days"):
    with store._pool.connection() as conn:
        conn.execute(
            "update answer_cache set created_at = now() - %s::interval where key = %s",
            (interval, key),
        )


def count_cache_rows(store) -> int:
    with store._pool.connection() as conn:
        return conn.execute("select count(*) from answer_cache").fetchone()[0]


def test_cache_roundtrip_and_missing_key(cache):
    assert cache.get("k1", 60) is None
    cache.put("k1", cached_response(), 60)
    assert cache.get("k1", 60) == cached_response()
    assert cache.get("other", 60) is None


def test_cache_put_replaces_an_existing_entry(cache):
    cache.put("k1", cached_response("old"), 60)
    cache.put("k1", cached_response("new"), 60)
    assert cache.get("k1", 60).answer == "new"


def test_cache_entry_older_than_the_ttl_is_not_served(store, cache):
    cache.put("k1", cached_response(), 3600)
    backdate(store, "k1", "2 hours")
    assert cache.get("k1", 3600) is None  # expired for a 1 hour TTL
    assert cache.get("k1", 86400) == cached_response()  # still fresh for a 1 day TTL


def test_cache_put_prunes_expired_entries(store, cache):
    cache.put("old", cached_response(), 3600)
    backdate(store, "old", "2 hours")
    cache.put("new", cached_response(), 3600)
    assert count_cache_rows(store) == 1
    assert cache.get("new", 3600) is not None


def test_replacing_a_document_empties_the_cache(store, cache):
    cache.put("k1", cached_response(), 60)
    store.replace_document(doc(), chunks(vec(1)))
    assert cache.get("k1", 60) is None


def test_metadata_update_empties_the_cache(store, cache):
    store.replace_document(doc(), chunks(vec(1)))
    cache.put("k1", cached_response(), 60)
    store.update_document_metadata(doc(title="Renamed"))
    assert cache.get("k1", 60) is None


def test_deleting_a_document_empties_the_cache(store, cache):
    store.replace_document(doc(), chunks(vec(1)))
    cache.put("k1", cached_response(), 60)
    assert store.delete_document("d1") == 1
    assert cache.get("k1", 60) is None


def test_deleting_a_missing_document_still_runs_the_statement_trigger(store, cache):
    # A statement-level trigger fires even when no row matches; clearing is harmless.
    cache.put("k1", cached_response(), 60)
    assert store.delete_document("never-existed") is None
    assert cache.get("k1", 60) is None
