from typing import Any, Protocol

import anthropic

from app.config import Settings
from app.vectorstore.base import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions using only the context provided in the user's message. "
    "If the context does not contain the answer, say so plainly. "
    "Refer to sources by their title. Do not invent details."
)


class ClaudeClient(Protocol):
    """Anything shaped like anthropic.Anthropic's .messages.create()."""

    messages: Any


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] Title: {chunk.title}\n"
        f"Section: {chunk.section_title}\n"
        f"{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    client: ClaudeClient,
    model: str,
    max_tokens: int = 1024,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n\n{_format_context(chunks)}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def get_claude_client(settings: Settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)
