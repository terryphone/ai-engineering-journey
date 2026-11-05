import pytest

from build_database import build_database
from make_reports import REPORTS, write_reports
from rag_tools import load_chunks, search_reports, split_report
from sql_tools import run_sql_query

SAMPLE = """# Test Review (synthetic)

First paragraph.

Second paragraph.
"""


def test_split_report_makes_one_chunk_per_paragraph():
    chunks = split_report(SAMPLE, "test.md")

    assert [c["id"] for c in chunks] == ["test.md#0", "test.md#1"]
    assert chunks[0]["text"] == "Test Review (synthetic)\nFirst paragraph."
    assert chunks[1]["source"] == "test.md"
    assert chunks[1]["title"] == "Test Review (synthetic)"


def test_split_report_ignores_extra_blank_lines():
    messy = "# Title\n\n\n\nOnly paragraph.\n\n\n"
    assert len(split_report(messy, "a.md")) == 1


def test_load_chunks_reads_markdown_files_only(tmp_path):
    (tmp_path / "a.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "b.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("not a report", encoding="utf-8")

    chunks = load_chunks(tmp_path)

    assert len(chunks) == 4
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