import sqlite3

import pytest

import sql_tools
from build_database import build_database
from sql_tools import run_sql_query


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    build_database(path)
    return path


def count_rows(db):
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM sector_output").fetchone()[0]
    conn.close()
    return n


def test_select_returns_rows(db):
    result = run_sql_query("SELECT COUNT(*) FROM gdp_quarterly", db)
    assert result["rows"] == [[27]]
    assert result["columns"] == ["COUNT(*)"]


def test_yoy_is_empty_for_first_four_quarters(db):
    result = run_sql_query(
        "SELECT yoy_growth_pct FROM gdp_growth WHERE quarter = '2020Q1'", db
    )
    assert result["rows"] == [[None]]


def test_yoy_matches_manual_calculation(db):
    result = run_sql_query(
        "SELECT gdp FROM gdp_quarterly "
        "WHERE quarter IN ('2025Q3', '2026Q3') ORDER BY quarter", db
    )
    old, new = (row[0] for row in result["rows"])
    expected = round((new / old - 1) * 100, 2)

    growth = run_sql_query(
        "SELECT yoy_growth_pct FROM gdp_growth WHERE quarter = '2026Q3'", db
    )
    assert growth["rows"] == [[expected]]


@pytest.mark.parametrize("query", [
    "DROP TABLE sector_output",
    "DELETE FROM sector_output",
    "INSERT INTO sector_output VALUES ('2030Q1', 'X', 1)",
    "UPDATE sector_output SET output = 0",
    "PRAGMA table_info(sector_output)",
    "ATTACH DATABASE 'other.db' AS other",
    "SELECT 1; DROP TABLE sector_output",
    "WITH t AS (SELECT 1) DELETE FROM sector_output",
    "WITH t AS (SELECT 1) INSERT INTO sector_output VALUES ('a','b',1)",
])
def test_dangerous_queries_are_blocked(db, query):
    result = run_sql_query(query, db)
    assert "error" in result
    assert count_rows(db) == 135  # data untouched


def test_row_limit(db):
    result = run_sql_query("SELECT * FROM sector_output", db)
    assert result["row_count"] == 50
    assert result["truncated"] is True


def test_unknown_table_returns_error_not_exception(db):
    assert "error" in run_sql_query("SELECT * FROM nope", db)


def test_empty_query(db):
    assert "error" in run_sql_query("   ", db)


def test_runaway_query_is_stopped(db, monkeypatch):
    monkeypatch.setattr(sql_tools, "MAX_SECONDS", 0.2)
    result = run_sql_query(
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) "
        "SELECT COUNT(*) FROM r", db
    )
    assert "interrupted" in result["error"].lower()

def test_qoq_growth_shows_the_2020_dip(db):
    result = run_sql_query(
        "SELECT quarter FROM gdp_growth "
        "WHERE qoq_growth_pct IS NOT NULL "
        "ORDER BY qoq_growth_pct ASC LIMIT 1", db
    )
    assert result["rows"] == [["2020Q2"]]