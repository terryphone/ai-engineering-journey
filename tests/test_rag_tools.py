import pytest

from analyst.build_database import build_database
from analyst.make_reports import REPORTS, write_reports
from analyst.rag_tools import build_index, load_chunks, search_reports, split_report
from analyst.sql_tools import run_sql_query

SAMPLE = """# Test Review (synthetic)

First paragraph.

Second paragraph.
"""


def test_long_reports_are_split_between_paragraphs():
    chunks = split_report(SAMPLE, "test.md", max_chars=20)

    assert [c["id"] for c in chunks] == ["test.md#0", "test.md#1"]
    assert chunks[0]["text"] == "Test Review (synthetic)\nFirst paragraph."
    assert chunks[1]["source"] == "test.md"
    assert chunks[1]["title"] == "Test Review (synthetic)"


def test_short_reports_stay_in_one_chunk():
    chunks = split_report(SAMPLE, "test.md")

    assert len(chunks) == 1
    assert chunks[0]["text"] == (
        "Test Review (synthetic)\nFirst paragraph.\n\nSecond paragraph."
    )


def test_split_report_ignores_extra_blank_lines():
    messy = "# Title\n\n\n\nOnly paragraph.\n\n\n"
    assert len(split_report(messy, "a.md")) == 1


def test_load_chunks_reads_markdown_files_only(tmp_path):
    (tmp_path / "a.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not a report", encoding="utf-8")

    chunks = load_chunks(tmp_path)

    assert len(chunks) == 2
    assert {c["source"] for c in chunks} == {"a.md", "b.md"}


def test_write_reports_creates_every_report(tmp_path):
    write_reports(tmp_path / "reports")
    assert len(load_chunks(tmp_path / "reports")) >= len(REPORTS)


def test_search_with_empty_query_returns_error(tmp_path):
    assert "error" in search_reports("  ", index_dir=tmp_path)


def test_search_without_index_returns_error(tmp_path):
    result = search_reports("anything", index_dir=tmp_path / "missing")
    assert "error" in result


@pytest.mark.parametrize("report, quarter, column", [
    ("2020Q2_review.md", "2020Q2", "qoq_growth_pct"),
    ("2021Q2_review.md", "2021Q2", "yoy_growth_pct"),
    ("2026Q3_review.md", "2026Q3", "yoy_growth_pct"),
    ("2026Q3_review.md", "2026Q2", "yoy_growth_pct"),
])
def test_report_numbers_match_the_database(tmp_path, report, quarter, column):
    """The reports must not drift away from the data."""
    db = tmp_path / "test.db"
    build_database(db)

    result = run_sql_query(
        f"SELECT {column} FROM gdp_growth WHERE quarter = '{quarter}'", db
    )
    value = result["rows"][0][0]

    assert f"{abs(value):.2f}%" in REPORTS[report]

@pytest.fixture(scope="module")
def report_index(tmp_path_factory):
    """A real index of every report, built in a temp folder."""
    root = tmp_path_factory.mktemp("rag")
    write_reports(root / "reports")
    build_index(root / "reports", root / "index")
    return root / "index"


def sources(result):
    return [p["source"] for p in result["passages"]]


def test_search_puts_the_named_quarter_first(report_index):
    # Real agent query where pure embedding search ranked 2020Q2 fifth (cut off)
    result = search_reports(
        "2020Q2 quarter-on-quarter GDP growth cause", index_dir=report_index
    )
    assert sources(result)[0] == "2020Q2_review.md"


def test_search_quarter_match_ignores_case(report_index):
    result = search_reports("why did output fall in 2020q2", index_dir=report_index)
    assert sources(result)[0] == "2020Q2_review.md"


def test_search_has_no_duplicates_and_keeps_the_limit(report_index):
    result = search_reports("2020Q2 contraction cause", index_dir=report_index)
    texts = [p["text"] for p in result["passages"]]
    assert len(texts) == len(set(texts))
    assert len(texts) <= 4


def test_search_for_a_quarter_without_a_report_still_works(report_index):
    # No 2022Q1 review exists: fall back to normal search, never invent it
    result = search_reports("2022Q1 review", index_dir=report_index)
    assert result["passages"]
    assert "2022Q1_review.md" not in sources(result)


def test_search_without_a_quarter_is_unchanged(report_index):
    result = search_reports("how is growth measured", index_dir=report_index)
    assert result["passages"]
