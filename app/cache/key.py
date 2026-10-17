import hashlib
import json

from app.models.document import normalize_tags
from app.models.query import QueryRequest

# Bump if the way the key is built changes, so old entries can never match.
KEY_VERSION = 1


def normalize_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def build_cache_key(
    request: QueryRequest,
    *,
    top_k: int,
    min_similarity: float,
    embedding_model: str,
    claude_model: str,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """Fingerprint of everything that decides the answer to a query.

    Only a hash is returned (and stored): the question text itself never reaches the
    database. `top_k` is the effective value (request or default). The full system
    prompt, persona included, is part of the key, so editing it starts a fresh cache.
    """
    parts = {
        "v": KEY_VERSION,
        "question": normalize_question(request.question),
        "category": request.filters.category,
        "featured": request.filters.featured,
        "tags": sorted({tag.lower() for tag in normalize_tags(request.filters.tags)}),
        "top_k": top_k,
        "min_similarity": min_similarity,
        "embedding_model": embedding_model,
        "claude_model": claude_model,
        "max_tokens": max_tokens,
        "system_prompt": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
    }
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
