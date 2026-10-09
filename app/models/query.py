from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    category: str | None = None
    featured: bool | None = None
    tags: list[str] = Field(default_factory=list)  # all must match


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: QueryFilters = Field(default_factory=QueryFilters)


class Source(BaseModel):
    document_id: str
    title: str
    section_title: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
