# Date and Status in the Answer Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Claude each retrieved document's `date` and `status`, and a prompt rule for using them, so "latest / most recent" questions are answered by comparing dates instead of guessing.

**Architecture:** The `documents` table already stores `date` and `status` (`db/schema.sql:17-18`), but `PostgresVectorStore.query` does not select them and `_format_context` only emits `title` and `section`. Carry the two fields on `RetrievedChunk`, emit them as attributes on the `<document>` tag when non-empty, and add a short recency rule to the system prompt. No migration, no re-embedding, no API change.

**Tech Stack:** Python 3.12, FastAPI, psycopg 3, pgvector, pytest (fakes for unit tests, a real pgvector Postgres for `tests/integration`).

**Spec:** Diagnosis in the Background section below. This is Fix 2 of the "chatbot named the wrong latest work" debug. The companion plan is `2026-10-19-portfolio-timeline-document.md` (Fix 1). The two are independent; ship the portfolio plan first.

## Global Constraints

- Never claim a feature is implemented when it is not (project rule). The README must describe only what ships.
- Safety rules stay word-for-word and in order (`tests/test_generation.py::SAFETY_SENTENCES`). Do not edit `SAFETY_RULES` or `STYLE_RULES`.
- Prompt-injection hardening stays: every new attribute value goes through `escape(..., quote=True)`.
- `RetrievedChunk` is constructed positionally in tests (`RetrievedChunk("b", "d", "T", "S", 0.1)`); new fields must be keyword-defaulted at the end so those keep working.
- Existing exact-string test `test_prompt_structure_wraps_context_documents_and_question` must keep passing unchanged: attributes appear only when the value is non-empty.
- No migration. Do not touch `db/`.
- Commit or push only when the user asks. The branch below is for working; do not merge.

## Background

The portfolio chatbot answered "What is the latest work that he has done?" with Elastic Face Recognition (2025.10-2026.01) instead of rag-api (2026.09-2026.10). Causes found:

1. `app/generation/claude_client.py::_format_context` sends only `title`, `section` and text to Claude. The `date` metadata is dropped.
2. The portfolio writes `Date: ...` into the body of each project, so only the first chunk of a document carries it.
3. Retrieval is pure similarity, so "latest" has no signal. (Handled by the portfolio plan.)

This plan fixes 1. Limit that remains: Claude can only compare the documents it was given, so the rule must say not to claim "newest overall" from a subset.

## File Structure

- Modify `app/vectorstore/base.py`: `RetrievedChunk` gains `date` and `status`.
- Modify `app/vectorstore/postgres.py`: `query` selects `d.date, d.status`.
- Modify `app/generation/claude_client.py`: `_format_context` attributes; new `RECENCY_RULES` appended to `SYSTEM_PROMPT`.
- Modify `tests/conftest.py`: `FakeVectorStore.query` fills the new fields.
- Modify `tests/test_generation.py`: new prompt and attribute tests.
- Modify `tests/integration/test_postgres_store.py`: assert the real query returns them.
- Modify `README.md`: Query step 6.

---

### Task 1: Carry date and status through retrieval

**Files:**
- Modify: `app/vectorstore/base.py:22-28`
- Modify: `app/vectorstore/postgres.py:186-197`
- Modify: `tests/conftest.py:72-80`
- Test: `tests/integration/test_postgres_store.py`, `tests/test_answer.py`

**Interfaces:**
- Produces: `RetrievedChunk(text, document_id, title, section_title, distance, date="", status="")`. Task 2 reads `chunk.date` and `chunk.status`.

- [ ] **Step 0: Branch**

```bash
cd /Users/shawn47/Desktop/rag-api
git switch -c feat/date-status-in-prompt
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/test_postgres_store.py` (it already has `store`, `doc`, `chunks`, `vec`, `SearchFilters` in scope):

```python
def test_query_returns_document_date_and_status(store):
    store.replace_document(
        doc(date="2026.09-2026.10", status="Building"), chunks(vec(1))
    )
    [hit] = store.query(vec(1), top_k=1, filters=SearchFilters())
    assert hit.date == "2026.09-2026.10"
    assert hit.status == "Building"
```

Append to `tests/test_answer.py` (read the file first and reuse its existing imports; the test below is self-contained):

```python
def test_fake_store_hits_carry_date_and_status(store):
    from app.pipeline.chunker import Chunk
    from app.pipeline.embedder import EmbeddedChunk
    from app.vectorstore.base import SearchFilters, StoredDocument

    store.replace_document(
        StoredDocument(
            document_id="d1", title="T", category="c", status="Building",
            date="2026.09-2026.10", featured=False, tags=[], content_hash="h",
            embedding_model="m", max_chunk_size=400, chunker_version=2,
        ),
        [EmbeddedChunk(Chunk(text="body", title="S", level=1), [1.0, 2.0])],
    )
    [hit] = store.query([1.0, 2.0], top_k=1, filters=SearchFilters())
    assert (hit.date, hit.status) == ("2026.09-2026.10", "Building")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_answer.py::test_fake_store_hits_carry_date_and_status -v`
Expected: FAIL with `AttributeError: 'RetrievedChunk' object has no attribute 'date'`.
(The integration test is skipped without `TEST_DATABASE_URL`; run it in Step 4.)

- [ ] **Step 3: Implement**

`app/vectorstore/base.py`, replace the `RetrievedChunk` dataclass:

```python
@dataclass
class RetrievedChunk:
    text: str
    document_id: str
    title: str
    section_title: str
    distance: float
    date: str = ""
    status: str = ""
```

`app/vectorstore/postgres.py`, in `query`, replace the select list (the `distance` expression stays the first query parameter):

```python
        sql = f"""
            select c.content, d.document_id, d.title, c.section_title,
                   c.embedding <=> %s::vector as distance,
                   d.date, d.status
            from document_chunks c
            join documents d on d.document_id = c.document_id
            {where}
            order by distance
            limit %s
        """
```

`tests/conftest.py`, in `FakeVectorStore.query`, extend the `RetrievedChunk(...)` call:

```python
                    RetrievedChunk(
                        text=ec.chunk.text,
                        document_id=doc_id,
                        title=doc.title,
                        section_title=ec.chunk.title,
                        distance=_cosine_distance(embedding, ec.embedding),
                        date=doc.date,
                        status=doc.status,
                    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, Postgres tests skipped.

With a disposable pgvector database (see README "Tests"):
`TEST_DATABASE_URL='postgresql://postgres:test@localhost:5432/postgres?options=-csearch_path%3Dpublic,extensions' .venv/bin/python -m pytest tests/integration -v`
Expected: `test_query_returns_document_date_and_status` passes.

- [ ] **Step 5: Commit (only if the user has asked for commits)**

```bash
git add app/vectorstore/base.py app/vectorstore/postgres.py tests/conftest.py tests/test_answer.py tests/integration/test_postgres_store.py
git commit -m "feat: return document date and status with retrieved chunks"
```

---

### Task 2: Emit date and status in the prompt, and add the recency rule

**Files:**
- Modify: `app/generation/claude_client.py:22-32`, `app/generation/claude_client.py:74-82`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: `RetrievedChunk.date`, `RetrievedChunk.status` (Task 1).
- Produces: `RECENCY_RULES: str`, included in `SYSTEM_PROMPT`; `<document ... date="..." status="...">` attributes, present only when non-empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation.py`, and add `RECENCY_RULES` to the existing `from app.generation.claude_client import (...)` block:

```python
RECENCY_PHRASES = [
    "date and status attributes",
    "end of the range",
    "only compare the documents you were given",
    "no dates",
    "Building",
]


def test_documents_carry_date_and_status_attributes_when_present():
    chunk = RetrievedChunk("Body.", "d1", "Doc", "Sec", 0.1, "2026.09-2026.10", "Building")
    _, req = ask("q", [chunk])
    content = req["messages"][0]["content"]
    assert (
        '<document title="Doc" section="Sec" date="2026.09-2026.10" status="Building">'
        in content
    )


def test_missing_date_or_status_adds_no_attribute():
    chunk = RetrievedChunk("Body.", "d1", "Doc", "Sec", 0.1, "2026.01", "")
    _, req = ask("q", [chunk])
    content = req["messages"][0]["content"]
    assert '<document title="Doc" section="Sec" date="2026.01">' in content
    assert "status=" not in content


def test_date_and_status_attributes_cannot_break_out():
    chunk = RetrievedChunk("b", "d", "T", "S", 0.1, '1" x="y', 'a"><b>')
    _, req = ask("q", [chunk])
    content = req["messages"][0]["content"]
    assert 'date="1&quot; x=&quot;y"' in content
    assert 'status="a&quot;&gt;&lt;b&gt;"' in content
    assert content.count("</document>") == 1


def test_recency_rules_are_in_the_system_prompt_after_the_style_rules():
    for phrase in RECENCY_PHRASES:
        assert phrase in RECENCY_RULES
    assert SYSTEM_PROMPT.endswith(RECENCY_RULES)
    assert SYSTEM_PROMPT.index(RECENCY_RULES) > SYSTEM_PROMPT.index("Style:")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generation.py -v`
Expected: collection error `ImportError: cannot import name 'RECENCY_RULES'`.

- [ ] **Step 3: Implement**

In `app/generation/claude_client.py`, after `STYLE_RULES` and replacing the `SYSTEM_PROMPT` line:

```python
RECENCY_RULES = (
    "Dates: a <document> can carry date and status attributes. A date is a single "
    "date or a start-end range; for a range, the end of the range is what counts as "
    "recent. For questions about the latest, newest or most recent item, compare "
    "those attributes, name the newest, and treat status Building as still in "
    "progress rather than finished. You can only compare the documents you were "
    "given, so do not claim something is the newest overall unless a document says "
    "so; if the documents carry no dates, say you cannot tell."
)

SYSTEM_PROMPT = f"{SAFETY_RULES}\n\n{STYLE_RULES}\n\n{RECENCY_RULES}"
```

Note: the test phrase `"only compare the documents you were given"` must appear verbatim in `RECENCY_RULES`. The sentence above splits it across a string-literal line break (`"...You can only compare the documents you were "` then `"given, ..."`), which concatenates to the exact phrase, so keep the trailing space before `"given"`. Likewise `"no dates"` is satisfied by `"carry no dates"`.

Replace `_format_context`:

```python
def _document_tag(chunk: RetrievedChunk) -> str:
    attributes = [("title", chunk.title), ("section", chunk.section_title)]
    # Optional attributes are omitted when empty, so a document without them
    # produces the same tag as before.
    attributes += [
        (name, value)
        for name, value in (("date", chunk.date), ("status", chunk.status))
        if value
    ]
    rendered = " ".join(f'{name}="{escape(value, quote=True)}"' for name, value in attributes)
    return f"<document {rendered}>"


def _format_context(chunks: list[RetrievedChunk]) -> str:
    documents = [
        f"{_document_tag(chunk)}\n"
        f"{_escape_text(chunk.text)}\n"
        "</document>"
        for chunk in chunks
    ]
    return "<context>\n" + "\n".join(documents) + "\n</context>"
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, including the untouched `test_prompt_structure_wraps_context_documents_and_question`, `test_every_safety_rule_is_present_word_for_word` and `test_persona_is_appended_after_all_the_rules_with_a_priority_line`.

- [ ] **Step 5: Confirm the cache key changes by itself**

Run: `.venv/bin/python -m pytest tests/test_cache_key.py -q`
Expected: pass. The system prompt is hashed into the cache key (`app/cache/key.py:42`), so the new rule starts a fresh cache with no code change. Do not bump `KEY_VERSION`.

- [ ] **Step 6: Commit (only if the user has asked for commits)**

```bash
git add app/generation/claude_client.py tests/test_generation.py
git commit -m "feat: show document date and status to the model and add a recency rule"
```

---

### Task 3: README

**Files:**
- Modify: `README.md:29-31`

- [ ] **Step 1: Edit Query step 6**

Replace

```
6. Ask Claude, with the chunks wrapped in escaped `<context>` tags, for a short
   answer. The response contains the answer and its sources. It is stored in the
   cache.
```

with

```
6. Ask Claude, with the chunks wrapped in escaped `<context>` tags (each carrying
   its document's title, section, and, when set, date and status), for a short
   answer. The response contains the answer and its sources. It is stored in the
   cache.
```

- [ ] **Step 2: Verify nothing else claims otherwise**

Run: `grep -n "date\|status" README.md`
Expected: only the metadata example, the response-status table and the new sentence. Confirm none says dates are used to rank or filter results (they are not; that is deferred, see below).

- [ ] **Step 3: Commit (only if the user has asked for commits)**

```bash
git add README.md
git commit -m "docs: note that date and status reach the model"
```

---

## Rollout and verification

1. Merge to `main` and cut a release (`v0.3.2`) only when the user decides; the CI release job then opens the bump PR in the homelab repo. Merging that PR is the rollout.
2. No migration to run first.
3. After rollout, ask the live chatbot: "What is the latest work that he has done?" and "What did he write most recently?". Expected: rag-api (Building, 2026.09-2026.10) and the newest post. If the answer is still wrong, check whether the rag-api project was retrieved at all (the portfolio timeline plan addresses that); do not widen the prompt rule.
4. Hand off to the documenter session to record the change in the vault. Do not edit the vault from the worker session.

## Deferred (Fix 3, only if needed)

Recency-aware retrieval (sort or filter on date) is not in this plan. Revisit if answers are still wrong after both plans ship.

## Self-review

- Spec coverage: date/status reach the model (Tasks 1-2), rule added (Task 2), cache safe (Task 2 Step 5), docs (Task 3). No gaps.
- Placeholders: none; every code step has code.
- Consistency: `RetrievedChunk.date` and `.status` are named identically in Tasks 1 and 2; `RECENCY_RULES` is defined once and imported by the test.
