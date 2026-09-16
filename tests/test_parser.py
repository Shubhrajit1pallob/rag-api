from app.pipeline.parser import Section, parse_markdown


def test_empty_content_returns_no_sections():
    assert parse_markdown("") == []


def test_no_headings_returns_single_untitled_section():
    result = parse_markdown("Just some plain text.\nNo headers here.")
    assert result == [Section(level=0, title="", body="Just some plain text.\nNo headers here.")]


def test_single_heading_with_body():
    result = parse_markdown("## Summary\nSome text here.")
    assert result == [Section(level=2, title="Summary", body="Some text here.")]


def test_content_before_first_heading_becomes_untitled_section():
    result = parse_markdown("Intro text.\n\n## Summary\nBody text.")
    assert result == [
        Section(level=0, title="", body="Intro text."),
        Section(level=2, title="Summary", body="Body text."),
    ]


def test_blank_lines_before_first_heading_do_not_create_empty_section():
    result = parse_markdown("\n\n## Summary\nBody text.")
    assert result == [Section(level=2, title="Summary", body="Body text.")]


def test_consecutive_headings_with_no_body_keep_empty_section():
    result = parse_markdown("## A\n## B\nText under B")
    assert result == [
        Section(level=2, title="A", body=""),
        Section(level=2, title="B", body="Text under B"),
    ]


def test_multiple_heading_levels_are_tracked_independently():
    result = parse_markdown("# Title\nIntro.\n## Sub\nDetail.")
    assert result == [
        Section(level=1, title="Title", body="Intro."),
        Section(level=2, title="Sub", body="Detail."),
    ]


def test_body_whitespace_is_stripped():
    result = parse_markdown("## Summary\n\n  Some text.  \n\n")
    assert result == [Section(level=2, title="Summary", body="Some text.")]


def test_seven_hashes_is_not_treated_as_a_heading():
    result = parse_markdown("####### Not a real heading\nBody.")
    assert result == [Section(level=0, title="", body="####### Not a real heading\nBody.")]
