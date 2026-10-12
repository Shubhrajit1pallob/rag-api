import pytest
from pydantic import ValidationError

from app.models.query import QueryRequest


def test_request_validation():
    with pytest.raises(ValidationError):
        QueryRequest(question="")
    with pytest.raises(ValidationError):
        QueryRequest(question="q", top_k=9)
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 2001)
    assert QueryRequest(question="q", top_k=8).top_k == 8


def test_filters_default_to_none():
    filters = QueryRequest(question="q").filters
    assert filters.category is None and filters.featured is None and filters.tags == []
