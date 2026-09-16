import re

from app.pipeline.parser import Section

_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BOLD_STAR_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
_BOLD_UNDERSCORE_PATTERN = re.compile(r"__([^_]+)__")
_ITALIC_STAR_PATTERN = re.compile(r"\*([^*]+)\*")
_ITALIC_UNDERSCORE_PATTERN = re.compile(r"_([^_]+)_")
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_BLANK_RUN_PATTERN = re.compile(r"\n\s*\n+")


def clean_text(text: str) -> str:
    """Strip markdown formatting noise that doesn't help semantic embedding.

    Links keep their visible text and drop the URL; bold/italic/inline-code
    markers are removed but their inner text is kept; runs of blank lines
    collapse to one. Order matters: bold markers are stripped before italic
    so '**word**' isn't misread as two separate '*word*' italics.
    """
    text = _LINK_PATTERN.sub(r"\1", text)
    text = _BOLD_STAR_PATTERN.sub(r"\1", text)
    text = _BOLD_UNDERSCORE_PATTERN.sub(r"\1", text)
    text = _ITALIC_STAR_PATTERN.sub(r"\1", text)
    text = _ITALIC_UNDERSCORE_PATTERN.sub(r"\1", text)
    text = _INLINE_CODE_PATTERN.sub(r"\1", text)
    text = _BLANK_RUN_PATTERN.sub("\n", text)
    return text.strip()


def clean_sections(sections: list[Section]) -> list[Section]:
    """Clean each section's title and body, dropping any that end up empty."""
    cleaned: list[Section] = []
    for section in sections:
        title = clean_text(section.title)
        body = clean_text(section.body)
        if title or body:
            cleaned.append(Section(level=section.level, title=title, body=body))
    return cleaned
