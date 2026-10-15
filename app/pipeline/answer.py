from app.generation.claude_client import ClaudeClient, generate_answer
from app.models.document import normalize_tags
from app.models.query import QueryRequest, QueryResponse, Source
from app.pipeline.embedder import EmbeddingClient, embed_query
from app.vectorstore.base import SearchFilters, VectorStore

NO_RESULTS_ANSWER = "I couldn't find anything relevant to that question."


def answer_question(
    request: QueryRequest,
    store: VectorStore,
    embedding_client: EmbeddingClient,
    claude_client: ClaudeClient,
    *,
    embedding_model: str,
    claude_model: str,
    claude_max_tokens: int,
    system_prompt: str,
    default_top_k: int,
    min_similarity: float,
) -> QueryResponse:
    embedding = embed_query(
        request.question, model=embedding_model, client=embedding_client
    )
    hits = store.query(
        embedding,
        top_k=request.top_k or default_top_k,
        filters=SearchFilters(
            category=request.filters.category,
            featured=request.filters.featured,
            tags=normalize_tags(request.filters.tags),
        ),
    )
    if min_similarity > 0:  # 0 means off, even for anti-correlated hits
        hits = [hit for hit in hits if 1 - hit.distance >= min_similarity]
    if not hits:
        return QueryResponse(answer=NO_RESULTS_ANSWER, sources=[])

    answer = generate_answer(
        request.question,
        hits,
        client=claude_client,
        model=claude_model,
        max_tokens=claude_max_tokens,
        system_prompt=system_prompt,
    )
    sources = [
        Source(
            document_id=hit.document_id,
            title=hit.title,
            section_title=hit.section_title,
            distance=hit.distance,
        )
        for hit in hits
    ]
    return QueryResponse(answer=answer, sources=sources)
