from app.pipeline.chunker import DEFAULT_MAX_CHUNK_SIZE, Chunk, chunk_sections
from app.pipeline.parser import Section


def test_empty_sections_list_returns_no_chunks():
    assert chunk_sections([]) == []


def test_single_section_becomes_one_chunk():
    sections = [Section(level=1, title="Intro", body="Some words here.")]
    assert chunk_sections(sections, max_chunk_size=400) == [
        Chunk(text="Intro\nSome words here.", title="Intro", level=1)
    ]


def test_adjacent_same_level_sections_merge_when_under_budget():
    sections = [
        Section(level=2, title="A", body="one two three"),
        Section(level=2, title="B", body="four five six"),
    ]
    assert chunk_sections(sections, max_chunk_size=20) == [
        Chunk(text="A\none two three\n\nB\nfour five six", title="A", level=2)
    ]


def test_combined_size_exactly_at_budget_still_merges():
    sections = [
        Section(level=2, title="A", body="one two"),
        Section(level=2, title="B", body="three four"),
    ]
    assert chunk_sections(sections, max_chunk_size=6) == [
        Chunk(text="A\none two\n\nB\nthree four", title="A", level=2)
    ]


def test_sections_at_different_levels_do_not_merge():
    sections = [
        Section(level=1, title="A", body="one two"),
        Section(level=2, title="B", body="three four"),
    ]
    assert chunk_sections(sections, max_chunk_size=400) == [
        Chunk(text="A\none two", title="A", level=1),
        Chunk(text="B\nthree four", title="B", level=2),
    ]


def test_merge_stops_once_budget_would_be_exceeded():
    sections = [
        Section(level=2, title="A", body="one two three"),
        Section(level=2, title="B", body="four five six"),
        Section(level=2, title="C", body="seven eight nine"),
    ]
    assert chunk_sections(sections, max_chunk_size=8) == [
        Chunk(text="A\none two three\n\nB\nfour five six", title="A", level=2),
        Chunk(text="C\nseven eight nine", title="C", level=2),
    ]


def test_oversized_section_splits_into_size_based_windows_with_title_prefix():
    body = " ".join(f"word{i}" for i in range(10))
    sections = [Section(level=1, title="Big", body=body)]
    expected_windows = [
        "word0 word1 word2 word3",
        "word4 word5 word6 word7",
        "word8 word9",
    ]
    assert chunk_sections(sections, max_chunk_size=4) == [
        Chunk(text=f"Big\n{window}", title="Big", level=1) for window in expected_windows
    ]


def test_oversized_section_without_title_has_no_prefix():
    body = " ".join(f"w{i}" for i in range(6))
    sections = [Section(level=0, title="", body=body)]
    expected_windows = ["w0 w1 w2", "w3 w4 w5"]
    assert chunk_sections(sections, max_chunk_size=3) == [
        Chunk(text=window, title="", level=0) for window in expected_windows
    ]


def test_oversized_section_flushes_pending_buffer_first():
    sections = [
        Section(level=2, title="A", body="one"),
        Section(level=2, title="B", body=" ".join(f"w{i}" for i in range(6))),
    ]
    assert chunk_sections(sections, max_chunk_size=3) == [
        Chunk(text="A\none", title="A", level=2),
        Chunk(text="B\nw0 w1 w2", title="B", level=2),
        Chunk(text="B\nw3 w4 w5", title="B", level=2),
    ]


def test_default_max_chunk_size_matches_explicit_default():
    sections = [Section(level=1, title="T", body="a b c")]
    assert chunk_sections(sections) == chunk_sections(
        sections, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE
    )
