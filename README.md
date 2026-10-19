# rag-api

A small FastAPI service that answers questions about a set of markdown documents.
You ingest documents; it chunks and embeds them (Voyage), stores the vectors in
Postgres with pgvector, and answers questions by retrieving the closest chunks and
asking Claude to answer from them only.

## How it works

**Ingest** (`POST /documents`, admin key)
1. The markdown is parsed into sections (split on headers) and cleaned.
2. Small sibling sections are merged up to `MAX_CHUNK_SIZE` words; a section larger
   than that is split into windows. A merged chunk is labelled with all its section
   titles, joined by ` / `.
3. Voyage embeds each chunk; the chunks and vectors are stored in Postgres.
4. A content hash, the embedding model, the chunk size and the chunker version are
   stored per document. If all of them match on the next ingest, nothing is
   re-embedded (a metadata-only change just updates the row).

**Query** (`POST /query`, query or admin key)
1. Check the key, then the question length.
2. Look the request up in the answer cache (see below). A hit returns immediately
   and uses no budget, Voyage call or Claude call.
3. On a miss, consume one unit of the global daily budget.
4. Embed the question and run a nearest-neighbour search (optionally filtered by
   category, featured and tags).
5. Drop chunks below `MIN_SIMILARITY`. If none remain, return a fixed "no results"
   answer without calling Claude.
6. Ask Claude, with the chunks wrapped in escaped `<context>` tags (each carrying
   its document's title, section, and, when set, date and status), for a short
   answer. The response contains the answer and its sources. It is stored in the
   cache.

**Answer cache.** An identical question (same filters, `top_k`, models, token cap and
full system prompt) is answered from a Postgres table instead of calling Voyage and
Claude again. Details:
- The cache key is a sha256 of the normalized request (question trimmed, whitespace
  collapsed and lowercased; tags lowercased and sorted). Only the hash is stored, never
  the question. A reworded question is a different key and is a miss.
- Any insert, update or delete on `documents` empties the whole cache, in the same
  transaction (a database trigger). Entries also expire after
  `ANSWER_CACHE_TTL_SECONDS`.
- `/query` responses carry `X-Cache: HIT` or `MISS` when the cache is enabled.
- The cache never causes a query to fail: a cache error is logged and treated as a miss.
- Known limit: if documents are re-ingested while a query is being answered, that
  answer can be stored just after the cache was emptied and then be served until the
  TTL expires. Avoid ingesting during live traffic, or set the TTL low.
- Because hits do not use the daily budget, `DAILY_QUERY_LIMIT` caps paid (uncached)
  queries; hits cost one database read.

## Endpoints

All requests send the key in the `X-API-Key` header.

| Method and path | Key | Purpose |
|---|---|---|
| `GET /health` | none | Liveness: `{"status": "ok"}` |
| `POST /documents` | admin | Ingest or update a document |
| `DELETE /documents/{document_id}` | admin | Delete a document and its chunks |
| `POST /query` | query or admin | Ask a question |

`POST /documents` body:

```json
{
  "document_id": "my-doc",
  "content": "## Overview\nMarkdown text...",
  "metadata": {
    "title": "My doc", "category": "project", "tags": ["Python"],
    "featured": false, "status": "published", "date": "2026-01-01"
  }
}
```

The response status is one of `created`, `content_updated`, `metadata_updated`,
`unchanged`, with the chunk count.

`POST /query` body: `question` (required), `top_k` (1-8, optional) and `filters`
(`category`, `featured`, `tags`; all given tags must match, case-insensitively).
Response: `answer` and `sources` (`document_id`, `title`, `section_title`, `distance`).

| Status | Meaning |
|---|---|
| 401 | Missing or wrong key (a query key cannot call admin routes) |
| 404 | `DELETE` for a document that does not exist |
| 422 | Invalid body, a document with no indexable content, or a question longer than `MAX_QUESTION_CHARS` |
| 429 | The daily query budget is used up; `Retry-After` gives the seconds until midnight UTC |
| 503 | Voyage, Anthropic or the database is unavailable or timed out |

## Configuration

Set as environment variables, or in a `.env` file for local runs (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `RAG_QUERY_API_KEY` | required | Key for `/query`. Must differ from the admin key |
| `RAG_ADMIN_API_KEY` | required | Key for ingest and delete (also accepted by `/query`) |
| `DATABASE_URL` | required | Postgres connection string (Supabase: the session pooler, port 5432) |
| `VOYAGE_API_KEY` | empty | Voyage key, used for embeddings |
| `ANTHROPIC_API_KEY` | empty | Anthropic key, used for answers |
| `VOYAGE_EMBEDDING_MODEL` | `voyage-3` | Embedding model; changing it re-embeds documents on next ingest |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Answer model |
| `EMBEDDING_DIMENSIONS` | `1024` | Must match the `vector(1024)` column |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `5` | Connection pool bounds |
| `MAX_CHUNK_SIZE` | `400` | Words per chunk |
| `RETRIEVAL_TOP_K` | `5` | Default chunks retrieved |
| `MIN_SIMILARITY` | `0.0` | Drop chunks with similarity (1 - cosine distance) below this; 0 is off |
| `DAILY_QUERY_LIMIT` | `300` | Total `/query` calls per UTC day, all callers (>= 1) |
| `CLAUDE_MAX_TOKENS` | `400` | Cap on answer length; a cut-off answer is trimmed to its last full sentence |
| `ASSISTANT_INSTRUCTIONS` | empty | Optional persona text (max 2000 characters), appended after the built-in rules, which always take priority. Quote it in `.env` if it has line breaks |
| `MAX_QUESTION_CHARS` | `500` | Longest accepted question |
| `ANSWER_CACHE_TTL_SECONDS` | `604800` | How long a stored answer is served (7 days); `0` turns the answer cache off. Needs migration `004` |
| `METRICS_PORT` | `0` | Port of the separate Prometheus metrics server (`/metrics`, no authentication); `0` turns it off. See Metrics |
| `UPSTREAM_TIMEOUT_SECONDS` | `20` | Timeout for Voyage and Anthropic calls |
| `UPSTREAM_MAX_RETRIES` | `1` | Retries for those calls |

## Database setup (Supabase or any Postgres with pgvector)

- **Fresh database:** run `db/schema.sql`. It always describes the full current
  schema. The commented block at its end creates the least-privilege `rag_api` role;
  run it separately with a real generated password.
- **Existing database:** run the files in `db/migrations/` in order, once each, in
  the Supabase SQL editor. **Run a migration before deploying the release that needs
  it.** Migrations are meant to be harmless to the previous release.
  `002_safety.sql` contains one statement (`create policy`) that is not re-runnable.
  `003_chunker_version.sql` adds `documents.chunker_version`; after the release that
  uses it is deployed, each document is re-chunked and re-embedded once on its next
  ingest. `004_answer_cache.sql` adds the `answer_cache` table and the trigger on
  `documents` that empties it; run it before deploying the release that uses the cache.
  `005_cache_function_search_path.sql` pins the `search_path` of that trigger function
  (the Supabase Security Advisor flags it otherwise); it is re-runnable and may run
  before or after any release.

Row-level security is enabled on every table; the app connects as `rag_api`, which
has only the grants it needs.

## Running locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # fill in real values
.venv/bin/uvicorn app.main:app --reload
```

### Tests

- **Unit tests only (default):** `.venv/bin/python -m pytest`. Everything uses fakes,
  so no keys or database are needed. The Postgres tests are skipped.
- **With a local pgvector database:** start a throwaway Postgres, for example
  `docker run --rm -e POSTGRES_PASSWORD=test -p 5432:5432 pgvector/pgvector:pg17`,
  then
  `TEST_DATABASE_URL='postgresql://postgres:test@localhost:5432/postgres?options=-csearch_path%3Dpublic,extensions' .venv/bin/python -m pytest tests/integration -v`.
  **Use a disposable database:** the tests drop and recreate the rag-api tables in
  `public`. Set `REQUIRE_TEST_DB=1` to make a missing `TEST_DATABASE_URL` an error
  instead of a skip (CI does this).

## Metrics

Prometheus metrics are off by default. Set `METRICS_PORT` (for example `9090`) and the
app serves `GET /metrics` on that port, from a thread in the same process. The API port
does not serve `/metrics` (it returns 404). The metrics port has **no authentication**:
it is meant to be reachable only by Prometheus, so restrict it at the network level (in
Kubernetes, with a NetworkPolicy that allows only the monitoring namespace to that port).
If the port cannot be bound, the app logs an error and starts without metrics.

| Metric | Labels | Meaning |
|---|---|---|
| `rag_http_requests_total` | `method`, `route`, `status` | Requests handled. `route` is the route template (`/documents/{document_id}`); requests matching no route share `unmatched` |
| `rag_http_request_duration_seconds` | `method`, `route` | Request duration histogram |
| `rag_query_outcomes_total` | `outcome` | Final result of `/query` requests that passed validation (below) |
| `rag_cache_errors_total` | `operation` (`get`, `put`) | Answer cache errors that were swallowed |
| `rag_upstream_errors_total` | `service` (`voyage`, `anthropic`, `database`, `other`) | Failures turned into a generic 503 |

`outcome` values are mutually exclusive: `cache_hit`; `cache_miss` (looked up, missed,
then answered); `cache_disabled` (cache off, answered); `budget_rejected` (429). A query
that fails with a 503 is in none of them; see the 503 count in
`rag_http_requests_total` and `rag_upstream_errors_total`. The cache hit rate is
`cache_hit / (cache_hit + cache_miss)`.

Labels never contain questions, document ids, tags, keys or error text. Counters live in
memory and reset on restart (use `rate()` in queries). The container runs one process,
so there is no multi-process aggregation; running several workers would need a different
setup. There is no token-usage or cost metric.

The tests check the metric values and the exposition text, and a local run of the app
was scraped over HTTP. It has not been scraped by a real Prometheus. The cluster side
(container and Service port, NetworkPolicy, ServiceMonitor) lives in the deployment
repository, not here.

## Docker and releases

The `Dockerfile` runs the app as a non-root user with `uvicorn` on port 8000.

`.github/workflows/ci.yml` runs on pull requests, pushes to `main` and releases:
tests, a dependency audit (`pip-audit`), the Postgres integration tests against a
pinned pgvector service container, and an image build with a Trivy scan.

- Pull requests and pushes to `main` build and scan the image but never push it.
- Publishing a stable GitHub Release (a `vX.Y.Z` tag on a commit that is on `main`)
  builds, scans and pushes the image to GHCR as
  `ghcr.io/shubhrajit1pallob/rag-api` with the tags `sha-<full commit sha>`,
  `<version>` and `latest`, with provenance and an SBOM.
- After that, a job opens a pull request in the deployment repository bumping the
  image tag. It never merges; merging is what triggers the rollout.

The container package must be made public in GHCR for a cluster to pull it without
credentials.

## Security design

- Two keys: a query key (safe to expose to a public web process) and an admin key.
  Keys are compared in constant time.
- A global daily query budget (atomic in the database) is a hard spend cap, plus caps
  on question length and answer length.
- Prompt-injection hardening: retrieved text and the question are escaped inside
  `<context>` and `<question>` tags and the system prompt treats them as data. This
  reduces risk but is not a guarantee.
- The system prompt's safety rules always take priority over any persona text.
- Row-level security plus a least-privilege database role; upstream errors become a
  generic 503 without leaking details.

## License

MIT, see `LICENSE`.
