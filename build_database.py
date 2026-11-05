import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "economy.db"

# sector: (starting output, average quarterly growth rate)
SECTORS = {
    "Manufacturing": (30000, 0.006),
    "Services": (60000, 0.009),
    "Construction": (8000, 0.004),
    "Finance": (20000, 0.010),
    "Trade": (15000, 0.005),
}

# One-off dips that create "unusual movements" to investigate
SHOCKS = {"2020Q2": 0.92, "2020Q3": 0.96}

SCHEMA = """
CREATE TABLE sector_output (
    quarter TEXT NOT NULL,
    sector  TEXT NOT NULL,
    output  REAL NOT NULL,
    PRIMARY KEY (quarter, sector)
);

CREATE VIEW gdp_quarterly AS
SELECT quarter, SUM(output) AS gdp
FROM sector_output
GROUP BY quarter;

CREATE VIEW gdp_growth AS
SELECT
    quarter,
    gdp,
    ROUND((gdp / LAG(gdp, 4) OVER (ORDER BY quarter) - 1) * 100, 2)
        AS yoy_growth_pct,
    ROUND((gdp / LAG(gdp, 1) OVER (ORDER BY quarter) - 1) * 100, 2)
        AS qoq_growth_pct
FROM gdp_quarterly;
"""


def quarter_labels():
    labels = []
    for year in range(2020, 2027):
        for q in range(1, 5):
            labels.append(f"{year}Q{q}")
    return labels[:27]  # 2020Q1 to 2026Q3


def build_database(db_path=DB_PATH):
    """Create the synthetic economy database from scratch."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(42)  # fixed seed: same data every time
    rows = []

    for t, quarter in enumerate(quarter_labels()):
        for sector, (start, growth) in SECTORS.items():
            noise = 1 + rng.uniform(-0.01, 0.01)
            output = (
                start * (1 + growth) ** t
                * SHOCKS.get(quarter, 1.0)
                * noise
            )
            rows.append((quarter, sector, round(output, 1)))

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO sector_output VALUES (?, ?, ?)", rows
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    build_database()
    print(f"Created {DB_PATH}")