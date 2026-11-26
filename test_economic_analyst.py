"""Offline tests for loading and validating the dashboard's GDP data."""
import sqlite3

import pandas as pd
import pytest

from build_database import build_database
from economic_analyst import calculate_growth, load_data, validate_data
from sql_tools import run_sql_query


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    build_database(path)
    return path


def test_loads_every_quarter_in_order(db):
    df = load_data(db)
    assert len(df) == 27
    assert str(df["Quarter"].iloc[0]) == "2020Q1"
    assert str(df["Quarter"].iloc[-1]) == "2026Q3"


def test_dashboard_growth_matches_the_sql_view(db):
    # The dashboard and the SQL agents must report the same figures
    stats = calculate_growth(load_data(db))
    rows = run_sql_query(
        "SELECT quarter, yoy_growth_pct FROM gdp_growth "
        "ORDER BY quarter DESC LIMIT 2", db
    )["rows"]
    assert stats["quarter"] == rows[0][0]
    assert stats["growth"] == pytest.approx(rows[0][1])
    assert stats["previous_growth"] == pytest.approx(rows[1][1])


def test_missing_database_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_data(tmp_path / "missing.db")


def test_database_without_the_view_raises_value_error(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match="Could not read"):
        load_data(path)


def frame(quarters, gdp):
    return pd.DataFrame({"Quarter": quarters, "GDP": gdp})


QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]


@pytest.mark.parametrize("df, message", [
    (frame(QUARTERS[:5] + ["2026Q3"], [1] * 6), "Missing quarters"),
    (frame(QUARTERS[:5] + ["2026Q1"], [1] * 6), "Duplicate"),
    (frame(QUARTERS, [1, 1, 1, 1, 1, 0]), "positive"),
    (frame(QUARTERS[:5], [1] * 5), "At least 6"),
    (pd.DataFrame({"Quarter": QUARTERS}), "Quarter and GDP"),
])
def test_validation_rejects_bad_data(df, message):
    with pytest.raises(ValueError, match=message):
        validate_data(df)
