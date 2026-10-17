import re

import pytest

from app.cache.key import build_cache_key, normalize_question
from app.models.query import QueryFilters, QueryRequest

BASE = dict(
    top_k=5,
    min_similarity=0.0,
    embedding_model="voyage-3",
    claude_model="claude-x",
    max_tokens=250,
    system_prompt="rules",
)


def key(question="What is it?", filters=None, **overrides):
    request = QueryRequest(question=question, filters=filters or QueryFilters())
    return build_cache_key(request, **{**BASE, **overrides})


def test_key_is_a_stable_sha256_hex_digest():
    assert key() == key()
    assert re.fullmatch(r"[0-9a-f]{64}", key())


def test_the_question_text_is_not_in_the_key():
    assert "what" not in key("What is it?").lower()


def test_normalize_question_trims_collapses_whitespace_and_casefolds():
    assert normalize_question("  What   IS\n it? ") == "what is it?"


def test_equivalent_questions_share_a_key():
    assert key("What is it?") == key("  what   IS it?  ")


def test_a_different_question_has_a_different_key():
    assert key("What is it?") != key("What is that?")


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 6},
        {"min_similarity": 0.3},
        {"embedding_model": "voyage-3-large"},
        {"claude_model": "claude-y"},
        {"max_tokens": 400},
        {"system_prompt": "rules plus a persona"},
    ],
)
def test_every_setting_that_shapes_the_answer_changes_the_key(overrides):
    assert key(**overrides) != key()


def test_filters_change_the_key():
    assert key(filters=QueryFilters(category="project")) != key()
    assert key(filters=QueryFilters(featured=True)) != key()
    assert key(filters=QueryFilters(featured=False)) != key(filters=QueryFilters())
    assert key(filters=QueryFilters(tags=["Python"])) != key()


def test_tag_order_case_and_duplicates_do_not_change_the_key():
    a = key(filters=QueryFilters(tags=["Python", "C++"]))
    assert a == key(filters=QueryFilters(tags=["c++", " PYTHON ", "python", ""]))


def test_cpp_and_csharp_tags_stay_distinct():
    assert key(filters=QueryFilters(tags=["C++"])) != key(filters=QueryFilters(tags=["C#"]))
