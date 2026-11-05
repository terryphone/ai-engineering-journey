from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Fictional reports written to match the synthetic economy database.
REPORTS = {
    "2020Q2_review.md": """# Quarterly Economic Review: 2020Q2 (synthetic)

Output contracted sharply in 2020Q2. GDP fell 6.66% from the previous quarter to 123,619 (synthetic units), the largest quarterly decline in the series. All five sectors contracted. Trade fell the most (8.84%) and Services the least (5.55%).

The review attributes the contraction to a temporary nationwide disruption to logistics and port operations, which halted shipments and delayed construction projects. The disruption is described as short-lived, with operations expected to normalise within two quarters.

Services held up best because it depends less on the physical movement of goods.
""",
    "2020Q4_review.md": """# Quarterly Economic Review: 2020Q4 (synthetic)

The recovery continued in 2020Q4. GDP grew 4.97% from the previous quarter to 135,781 (synthetic units), following a 4.64% rise in 2020Q3. With this, output returned above its pre-disruption level of 132,438 in 2020Q1.

The review credits the reopening of ports and the clearing of delayed shipments. Construction projects that were postponed in the second quarter resumed, adding to growth.
""",
    "2021Q2_review.md": """# Quarterly Economic Review: 2021Q2 (synthetic)

Year-on-year growth reached 11.41% in 2021Q2, the highest in the series, followed by 8.18% in 2021Q3. These figures look exceptional but largely reflect a base effect: they compare against 2020Q2 and 2020Q3, when output was temporarily depressed.

Underlying momentum was ordinary. Quarter-on-quarter growth in 2021Q2 was only 0.11%. Year-on-year growth normalised to 3.86% by 2021Q4, once the comparison base no longer included the disruption.
""",
    "2026Q3_review.md": """# Quarterly Economic Review: 2026Q3 (synthetic)

GDP rose 0.67% on the previous quarter and 2.82% year on year in 2026Q3, down from 3.10% in 2026Q2. Growth therefore decelerated by 0.28 percentage points.

The slowdown was concentrated in Trade and Manufacturing. Trade growth slowed to 0.34% year on year from 1.77%, and Manufacturing to 1.27% from 1.77%. The review attributes this to softer external demand for goods.

Finance (5.67% year on year) and Services (3.22%) remained the strongest sectors. Services accounted for 46.3% of GDP, the largest share of any sector.
""",
    "methodology_note.md": """# Methodology Note: How Growth Is Measured (synthetic)

Year-on-year (YoY) growth compares a quarter with the same quarter one year earlier. It removes seasonal patterns but can be distorted by base effects, when the earlier quarter was unusually low or high.

Quarter-on-quarter (QoQ) growth compares a quarter with the one immediately before it. It reacts faster to turning points but is more volatile.

Unusual movements should be assessed on growth rates, not GDP levels, because levels trend upward over time. A common screen flags growth rates more than two standard deviations from their average.
""",
}


def write_reports(reports_dir=REPORTS_DIR):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(exist_ok=True)
    for name, text in REPORTS.items():
        (reports_dir / name).write_text(text, encoding="utf-8")
    return len(REPORTS)


if __name__ == "__main__":
    print(f"Wrote {write_reports()} reports to {REPORTS_DIR}")