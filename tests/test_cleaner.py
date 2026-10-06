from app.pipeline.cleaner import clean_sections, clean_text
from app.pipeline.parser import Section


def test_plain_text_is_unchanged():
    assert clean_text("Plain text with no markdown.") == "Plain text with no markdown."


def test_link_keeps_visible_text_drops_url():
    assert clean_text("[a link](http://example.com)") == "a link"


def test_bold_star_is_unwrapped():
    assert clean_text("**bold**") == "bold"


def test_bold_underscore_is_unwrapped():
    assert clean_text("__bold__") == "bold"


def test_italic_star_is_unwrapped():
    assert clean_text("*italic*") == "italic"


def test_italic_underscore_is_unwrapped():
    assert clean_text("_italic_") == "italic"


def test_inline_code_is_unwrapped():
    assert clean_text("`code`") == "code"


def test_bold_is_stripped_before_italic_so_double_star_is_not_misread():
    text = "**bold** and *italic* and `code` and [a link](http://example.com)."
    assert clean_text(text) == "bold and italic and code and a link."


def test_blank_line_runs_collapse_to_one():
    assert clean_text("Line one.\n\n\n\nLine two.") == "Line one.\nLine two."


def test_leading_and_trailing_whitespace_is_stripped():
    assert clean_text("  \n  Some text.  \n  ") == "Some text."


def test_clean_sections_keeps_a_section_with_empty_body_but_real_title():
    sections = [Section(level=3, title="Credentials", body="")]
    assert clean_sections(sections) == [Section(level=3, title="Credentials", body="")]


def test_clean_sections_drops_a_section_that_is_whitespace_only_throughout():
    sections = [Section(level=2, title="   ", body="  \n\n  ")]
    assert clean_sections(sections) == []


def test_clean_sections_cleans_and_filters_a_mixed_document():
    sections = [
        Section(level=1, title="Intro", body="This is **bold** text with a [link](http://x.com)."),
        Section(level=2, title="Empty Section", body=""),
        Section(level=2, title="", body="   "),
    ]
    assert clean_sections(sections) == [
        Section(level=1, title="Intro", body="This is bold text with a link."),
        Section(level=2, title="Empty Section", body=""),
    ]
