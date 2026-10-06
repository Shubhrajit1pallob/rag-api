from dataclasses import dataclass

from app.pipeline.parser import Section

DEFAULT_MAX_CHUNK_SIZE = 400


@dataclass
class Chunk:
    """A single embeddable piece of text, with its originating section's title for context."""

    text: str
    title: str
    level: int


def _word_count(text: str) -> int:
    return len(text.split())


def _section_text(section: Section) -> str:
    if section.title and section.body:
        return f"{section.title}\n{section.body}"
    return section.title or section.body


def _split_oversized_section(section: Section, max_chunk_size: int) -> list[Chunk]:
    """Split a single section's body into size-based windows, each prefixed with the title."""
    words = section.body.split()
    chunks: list[Chunk] = []
    for start in range(0, len(words), max_chunk_size):
        window = " ".join(words[start : start + max_chunk_size])
        text = f"{section.title}\n{window}" if section.title else window
        chunks.append(Chunk(text=text, title=section.title, level=section.level))
    return chunks


def chunk_sections(
    sections: list[Section], max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> list[Chunk]:
    """Group cleaned sections into embeddable chunks.

    Structure is the primary boundary: each section is a natural chunk. Adjacent
    sections at the same heading level are merged as long as the combined size
    stays within max_chunk_size, so structurally-related siblings don't end up
    as many tiny chunks. A section whose own body already exceeds max_chunk_size
    is split into size-based windows instead, each carrying the section's title
    so a mid-split chunk still has context on its own.
    """
    chunks: list[Chunk] = []
    buffer_sections: list[Section] = []
    buffer_size = 0

    def flush_buffer() -> None:
        if not buffer_sections:
            return
        text = "\n\n".join(_section_text(s) for s in buffer_sections)
        first = buffer_sections[0]
        chunks.append(Chunk(text=text, title=first.title, level=first.level))
        buffer_sections.clear()

    for section in sections:
        size = _word_count(_section_text(section))

        if size > max_chunk_size:
            flush_buffer()
            buffer_size = 0
            chunks.extend(_split_oversized_section(section, max_chunk_size))
            continue

        same_level_as_buffer = (
            buffer_sections and buffer_sections[-1].level == section.level
        )
        fits_in_buffer = buffer_size + size <= max_chunk_size

        if buffer_sections and same_level_as_buffer and fits_in_buffer:
            buffer_sections.append(section)
            buffer_size += size
        else:
            flush_buffer()
            buffer_sections.append(section)
            buffer_size = size

    flush_buffer()
    return chunks
