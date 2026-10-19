# Portfolio Timeline Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export one extra document to rag-api that lists every project and post newest-first with dates and status, so questions like "what is the latest work?" retrieve an explicit, correctly ordered answer.

**Architecture:** The portfolio's `app/rag_export.py` already turns site content into markdown documents and POSTs them to rag-api. Add a generated `timeline:latest-work` document built from the same data (so it cannot go stale), include it in `all_documents()`, and map its id to the work index in `_source_link` so the source chip links somewhere real. No rag-api change.

**Tech Stack:** Python 3.12, Flask, pytest. Repository: `/Users/shawn47/Desktop/Portfolio Website`.

**Spec:** Diagnosis in the Background section below. This is Fix 1 of the "chatbot named the wrong latest work" debug. The companion plan is `2026-10-19-date-and-status-in-prompt.md` (Fix 2, rag-api repo). Ship this plan first; it does not depend on the other.

## Global Constraints

- All work happens in `/Users/shawn47/Desktop/Portfolio Website`, not in the rag-api repo.
- Generated, never hand-written: the timeline must be built from `projects_data.all_projects()` and `writing_data.all_posts()`. Do not hard-code project names or dates.
- Same exclusion rule as projects: skip status in `SKIP_STATUSES` (`{"Planned"}`), so unbuilt work is never listed.
- Do not fabricate: a project with an unparseable date sorts last, and a test must fail if any real entry is unparseable.
- Existing source-link behaviour for `project:` and `post:` ids is unchanged (`tests/test_chat_route.py::test_answer_with_sources_mapped_and_deduplicated` must pass untouched).
- Running the real export writes to the production rag-api and empties its answer cache. The worker session must not run it; the user does (Task 3).
- Commit or push only when the user asks. Branch for working, do not merge.

## Background

The chatbot answered "What is the latest work that he has done?" with Elastic Face Recognition (`2025.10-2026.01`) instead of rag-api (`2026.09-2026.10`, status Building). The retrieved set was five broad chunks (Elastic Face Recognition, Homelab, Tetris, Dementia, one rag-api post) and the rag-api project itself was not among the sources. Retrieval is pure similarity, so "latest" matches nothing in particular. A document that literally says "latest, newest, most recent, newest first" will rank high for these questions and carries the ordering the model needs.

Date formats in the data, which the sort must handle: `2026.03` (single month), `2025.10-2026.01` (start-end range), and posts' ISO dates such as `2026-10-19`. Recency is the END of a range, then the start.

## File Structure

- Modify `app/rag_export.py`: add `_recency_key`, `timeline_markdown`, `timeline_document`; include it in `all_documents()`.
- Modify `app/routes.py:377-384`: `_source_link` handles the `timeline` kind.
- Modify `tests/test_rag_export.py`: unit tests for ordering, exclusions, the document, and the catalog test's id-prefix assertion.
- Modify `tests/test_chat_route.py`: source link test.

---

### Task 1: Recency ordering and the timeline markdown

**Files:**
- Modify: `app/rag_export.py` (add after `project_markdown`, before `project_documents`)
- Test: `tests/test_rag_export.py`

**Interfaces:**
- Produces: `_recency_key(date) -> tuple[int, int, int, int]` returning `(end_year, end_month, start_year, start_month)`, or `(0, 0, 0, 0)` if unparseable; `_exportable_projects(projects) -> list[dict]`; `timeline_markdown(projects, posts) -> str`. Task 2 calls all three.

- [ ] **Step 0: Branch**

```bash
cd "/Users/shawn47/Desktop/Portfolio Website"
git switch -c feat/rag-timeline-document
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rag_export.py` (add `import pytest` at the top of the file with the other imports):

```python
@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2026.03", (2026, 3, 2026, 3)),
        ("2025.10-2026.01", (2026, 1, 2025, 10)),
        ("2026.09-2026.10", (2026, 10, 2026, 9)),
        ("2026-10-19", (2026, 10, 2026, 10)),
        ("", (0, 0, 0, 0)),
        ("ongoing", (0, 0, 0, 0)),
    ],
)
def test_recency_key_uses_the_end_of_a_range(date, expected):
    assert rag_export._recency_key(date) == expected


def test_a_later_end_beats_a_later_start():
    # A long range that ends later outranks a one-month project that started after it did.
    assert rag_export._recency_key("2026.01-2026.09") > rag_export._recency_key("2026.03")


PROJECTS = [
    {"title": "Old", "date": "2026.01", "status": "Completed"},
    {"title": "Newest", "date": "2026.09-2026.10", "status": "Building"},
    {"title": "Mid", "date": "2025.10-2026.04", "status": "Completed"},
    {"title": "Unbuilt", "date": "2026.12", "status": "Planned"},
]
POSTS = [
    {"title": "Post A", "date": "2026-09-01"},
    {"title": "Post B", "date": "2026-10-19"},
]


def test_timeline_lists_newest_first_and_skips_planned():
    md = rag_export.timeline_markdown(PROJECTS, POSTS)
    assert md.startswith("# Timeline of work and writing\n")
    assert "Unbuilt" not in md
    section = md.split("## All projects, newest first")[1].split("## All writing")[0]
    assert section.index("Newest") < section.index("Mid") < section.index("Old")
    assert "- Newest (2026.09-2026.10, Building)" in md
    writing = md.split("## All writing, newest first")[1]
    assert writing.index("Post B") < writing.index("Post A")


def test_timeline_latest_section_names_the_three_newest_of_each():
    many = [{"title": f"P{i}", "date": f"2026.0{i}", "status": "Completed"} for i in range(1, 6)]
    posts = POSTS + [{"title": "Post C", "date": "2026-08-01"}, {"title": "Post D", "date": "2026-07-01"}]
    md = rag_export.timeline_markdown(many, posts)
    latest = md.split("## Latest work")[1].split("## All projects")[0]
    assert all(f"P{i}" in latest for i in (5, 4, 3))
    assert "P2" not in latest and "P1" not in latest
    assert all(f"Post {name}" in latest for name in "BAC")
    assert "Post D" not in latest
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_rag_export.py -q` (use the portfolio's own virtualenv if it has one; see its README)
Expected: FAIL with `AttributeError: module 'app.rag_export' has no attribute '_recency_key'`.

- [ ] **Step 3: Implement**

In `app/rag_export.py`, add `import re` to the imports (after `import os`), then add after `project_markdown`:

```python
_DATE_PART = re.compile(r"(\d{4})[.-](\d{1,2})")
_LATEST_COUNT = 3


def _recency_key(date):
    """Sort key for a project or post date: (end_year, end_month, start_year, start_month).

    Handles "2026.03", "2025.10-2026.01" (start-end) and ISO "2026-10-19". A range is
    as recent as its END. An unparseable date gets (0, 0, 0, 0) and sorts last.
    """
    parts = [(int(year), int(month)) for year, month in _DATE_PART.findall(str(date))]
    if not parts:
        return (0, 0, 0, 0)
    start, end = parts[0], parts[-1]
    return (*end, *start)


def _exportable_projects(projects):
    return [p for p in projects if p.get("status") not in SKIP_STATUSES]


def _project_line(project):
    return f"- {project['title']} ({project.get('date')}, {project.get('status')})"


def _post_line(post):
    return f"- {post['title']} ({post.get('date')})"


def timeline_markdown(projects, posts):
    """One document that answers "what is the latest / most recent / newest work?".

    Built from the same data as the site, newest first, so it is never hand-edited
    and never stale after a re-export.
    """
    ordered_projects = sorted(
        _exportable_projects(projects), key=lambda p: _recency_key(p.get("date")), reverse=True
    )
    ordered_posts = sorted(posts, key=lambda p: _recency_key(p.get("date")), reverse=True)
    parts = [
        "# Timeline of work and writing",
        "Newest first. Check here for the latest, newest or most recent work and writing. "
        "A date is year.month; a range such as 2025.10-2026.01 is start-end, and the end "
        "is what counts as recent. Status says whether a project is still being built.",
        "## Latest work\n\nThe most recent projects:\n"
        + "\n".join(_project_line(p) for p in ordered_projects[:_LATEST_COUNT])
        + "\n\nThe most recent posts:\n"
        + "\n".join(_post_line(p) for p in ordered_posts[:_LATEST_COUNT]),
        "## All projects, newest first\n\n" + "\n".join(_project_line(p) for p in ordered_projects),
        "## All writing, newest first\n\n" + "\n".join(_post_line(p) for p in ordered_posts),
    ]
    return "\n\n".join(parts) + "\n"
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_rag_export.py -q`
Expected: the new tests pass, and `test_real_catalog_exports_with_stable_ids_and_skips_planned` (not yet changed) still passes.

- [ ] **Step 5: Commit (only if the user has asked for commits)**

```bash
git add app/rag_export.py tests/test_rag_export.py
git commit -m "feat: build a newest-first timeline for the RAG export"
```

---

### Task 2: Export the timeline and link its source chip

**Files:**
- Modify: `app/rag_export.py` (new `timeline_document`; replace `all_documents`)
- Modify: `app/routes.py:377-384`
- Test: `tests/test_rag_export.py` (the id-prefix assertion in the catalog test, plus a new test), `tests/test_chat_route.py`

**Interfaces:**
- Consumes: `timeline_markdown`, `_recency_key`, `_exportable_projects` (Task 1).
- Produces: `timeline_document() -> dict` with `document_id == "timeline:latest-work"`; `_source_link("timeline:latest-work") == url_for("work_index")`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_rag_export.py`, change the id-prefix assertion in `test_real_catalog_exports_with_stable_ids_and_skips_planned` to:

```python
    assert all(i.startswith(("project:", "post:", "timeline:")) for i in ids)
```

and append:

```python
def test_real_catalog_has_one_timeline_document_ordered_and_fully_dated():
    docs = rag_export.all_documents()
    timelines = [d for d in docs if d["document_id"] == "timeline:latest-work"]
    assert len(timelines) == 1
    doc = timelines[0]
    assert doc["metadata"]["category"] == "timeline"
    assert doc["metadata"]["status"] == "Published"
    assert doc["metadata"]["date"] and doc["metadata"]["date"] != "n/a"
    assert "## Latest work" in doc["content"]

    # Every real project and post must parse to a real date, or it would sort last
    # and the "latest" answer could be wrong without anyone noticing.
    from app.data import projects as projects_data
    from app.data import writing as writing_data

    for item in list(projects_data.all_projects()) + list(writing_data.all_posts()):
        if item.get("status") in rag_export.SKIP_STATUSES:
            continue
        assert rag_export._recency_key(item.get("date")) != (0, 0, 0, 0), item["title"]
```

In `tests/test_chat_route.py`, append (it already defines `post` and uses the `client`, `chat_on`, `ask` fixtures):

```python
def test_timeline_source_links_to_the_work_index(client, chat_on, ask):
    ask.result = {
        "answer": "Latest is rag-api.",
        "sources": [
            {"document_id": "timeline:latest-work", "title": "Timeline of work and writing",
             "section_title": "Latest work", "distance": 0.1},
        ],
    }
    body = post(client).get_json()
    assert body["sources"] == [{"title": "Timeline of work and writing", "url": "/work"}]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_rag_export.py tests/test_chat_route.py -q`
Expected: FAIL. The catalog test finds no `timeline:latest-work`, and the route test returns `"url": None`.

- [ ] **Step 3: Implement**

In `app/rag_export.py`, add before `all_documents`, and replace `all_documents`:

```python
def timeline_document():
    projects = projects_data.all_projects()
    posts = writing_data.all_posts()
    newest = max(
        _exportable_projects(projects),
        key=lambda p: _recency_key(p.get("date")),
        default={},
    )
    return {
        "document_id": "timeline:latest-work",
        "content": timeline_markdown(projects, posts),
        "metadata": {
            "title": "Timeline of work and writing",
            "category": "timeline",
            "tags": [],
            "featured": False,
            "status": "Published",
            "date": str(newest.get("date") or "n/a"),
        },
    }


def all_documents():
    return [timeline_document()] + list(project_documents()) + list(post_documents())
```

In `app/routes.py`, inside `_source_link`, add before the final `return None`:

```python
    if kind == "timeline":
        return url_for("work_index")
```

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass. If `test_timeline_source_links_to_the_work_index` reports a URL other than `/work`, read the `work_index` route in `app/routes.py` and change the expected string in that test to the real path. Do not change the route.

- [ ] **Step 5: Dry run (read-only; nothing is sent)**

Run: `python -m app.rag_export --dry-run --only timeline:latest-work`
Expected: prints the timeline, with the rag-api project first under "All projects, newest first" and "Observing the RAG API" first under "All writing, newest first". Paste the output back to the user. If any ordering looks wrong, stop and report; do not hand-edit data.

- [ ] **Step 6: Commit (only if the user has asked for commits)**

```bash
git add app/rag_export.py app/routes.py tests/test_rag_export.py tests/test_chat_route.py
git commit -m "feat: export a generated timeline document and link its source"
```

---

### Task 3: Ship it (the user runs this)

**Files:** none.

The worker session must not run these: they write to the production rag-api (which also empties its answer cache via the database trigger).

- [ ] **Step 1: User merges and deploys the portfolio change** through the usual image build and Argo CD flow, so `/api/chat` serves the new `_source_link`.
- [ ] **Step 2: User exports.**

```bash
kubectl -n rag-api port-forward svc/rag-api 8000:8000
RAG_API_URL=http://127.0.0.1:8000 RAG_API_KEY=... python -m app.rag_export --only timeline:latest-work
```

Expected: `timeline:latest-work: created (N chunks)`. Unchanged documents are not re-embedded, so a full `python -m app.rag_export` is also safe.
- [ ] **Step 3: Verify on the live site.** Ask the chatbot: "What is the latest work that he has done?", "What is the most recent thing he wrote?", "What is he building right now?". Expected: rag-api (Building, 2026.09-2026.10) and *Observing the RAG API*. If "latest work" still misses rag-api, record the exact answer and the sources list and report back; that is the signal for Fix 3 (recency-aware retrieval), not a reason to edit this plan's data.

## Rollout notes

- After any content change, re-run the export: the timeline is regenerated from the data, and unchanged documents cost nothing.
- Once the companion rag-api plan ships, Claude also sees `date` and `status` on every chunk, so the two fixes reinforce each other.
- Hand off to the documenter session to record this in the vault. Do not edit the vault from the worker session.

## Self-review

- Spec coverage: ordered timeline from real data (Task 1), exported and linkable (Task 2), shipped by the user (Task 3), Planned excluded (Global Constraints and tests). No gaps.
- Placeholders: none.
- Consistency: `_recency_key`, `_exportable_projects`, `timeline_markdown(projects, posts)`, `timeline_document()` and the id `timeline:latest-work` are spelled identically in every task.
