import re
from dataclasses import dataclass

# Standard markdown only supports heading levels 1-6 ('#' through '######').
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    """One heading and the body text under it, before any cleaning or chunking."""

    level: int
    title: str
    body: str


def parse_markdown(content: str) -> list[Section]:
    """Split markdown content into sections based on '#' headers.

    Content before the first heading becomes a level-0, untitled section,
    but only if it's actually non-empty once stripped -- stray blank lines
    before the first heading don't produce a spurious section. Consecutive
    headings with no body between them still produce a section with an
    empty body rather than being dropped -- later stages decide what to do
    with that, the parser just preserves structure.
    """
    sections: list[Section] = []
    level = 0
    title = ""
    body_lines: list[str] = []
    seen_heading = False

    def flush() -> None:
        body_text = "\n".join(body_lines).strip()
        if seen_heading or body_text:
            sections.append(Section(level=level, title=title, body=body_text))

    for line in content.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match is None:
            body_lines.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        body_lines = []
        seen_heading = True

    flush()
    return sections
