import pandas as pd
import pytest

from analyst_tools import build_tools


@pytest.fixture
def tools():
    df = pd.DataFrame({
        "Quarter": pd.PeriodIndex(
            ["2025Q1", "2025Q2", "2025Q3", "2025Q4",
             "2026Q1", "2026Q2", "2026Q3"], freq="Q"
        ),
        "GDP": [145000, 148000, 151000, 154000,
                150000, 155000, 160000],
    })
    return build_tools(df)


def test_weakest_growth(tools):
    assert tools["get_extreme_growth"]("lowest") == {
        "quarter": "2026Q1", "yoy_growth_pct": 3.45}


def test_highest_growth(tools):
    assert tools["get_extreme_growth"]("highest") == {
        "quarter": "2026Q3", "yoy_growth_pct": 5.96}


def test_compare_quarters(tools):
    result = tools["compare_quarters"]("2025Q1", "2026Q3")
    assert result["pct_change"] == 10.34


def test_not_enough_history(tools):
    assert "error" in tools["get_yoy_growth"]("2025Q2")


def test_unknown_quarter(tools):
    assert "error" in tools["get_gdp"]("2030Q1")


def test_bad_kind(tools):
    assert "error" in tools["get_extreme_growth"]("median")


def test_average_growth(tools):
    result = tools["average_yoy_growth"]("2026Q1", "2026Q3")
    assert result["average_yoy_growth_pct"] == 4.71
    assert result["quarters_used"] == 3


def test_average_growth_wrong_order(tools):
    assert "error" in tools["average_yoy_growth"]("2026Q3", "2026Q1")
    