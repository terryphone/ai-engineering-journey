
import os
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel


# =====================================
# 1. DEFINE AI OUTPUT STRUCTURE
# =====================================

class EconomicReport(BaseModel):
    quarter: str
    growth_rate: float
    previous_growth: float
    commentary: str
    key_observation: str


# =====================================
# 2. LOAD AND VALIDATE DATA
# =====================================

def load_data(filename):
    df = pd.read_csv(filename)

    required = {"Quarter", "GDP"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain Quarter and GDP columns"
        )

    if df.empty:
        raise ValueError("Dataset is empty")

    df["Quarter"] = pd.PeriodIndex(
        df["Quarter"], freq="Q"
    )

    df["GDP"] = pd.to_numeric(
        df["GDP"], errors="raise"
    )

    if df["GDP"].isna().any():
        raise ValueError("GDP contains missing values")

    if not (df["GDP"] > 0).all():
        raise ValueError("GDP must be positive")

    if df["Quarter"].duplicated().any():
        raise ValueError("Duplicate quarters found")

    df = df.sort_values("Quarter").reset_index(
        drop=True
    )

    expected = pd.period_range(
        start=df["Quarter"].iloc[0],
        end=df["Quarter"].iloc[-1],
        freq="Q"
    )

    if len(df) != len(expected):
        raise ValueError(
            "Missing quarters in the dataset"
        )

    if len(df) < 6:
        raise ValueError(
            "At least 6 quarters are required"
        )

    return df


# =====================================
# 3. CALCULATE ECONOMIC STATISTICS
# =====================================

def calculate_growth(df):
    df = df.copy()

    df["YoY_Growth"] = (
        df["GDP"].pct_change(periods=4) * 100
    ).round(2)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    return {
        "quarter": str(latest["Quarter"]),
        "gdp": float(latest["GDP"]),
        "growth": float(latest["YoY_Growth"]),
        "previous_quarter": str(previous["Quarter"]),
        "previous_growth": float(
            previous["YoY_Growth"]
        ),
        "growth_change": round(
            float(latest["YoY_Growth"])
            - float(previous["YoY_Growth"]),
            2
        )
    }


# =====================================
# 4. GENERATE AI COMMENTARY
# =====================================

def generate_commentary(stats):
    client = genai.Client()

    prompt = f"""
    You are an economic analyst.

    The following figures are synthetic.

    Latest quarter: {stats['quarter']}
    GDP level: {stats['gdp']}
    Latest YoY growth: {stats['growth']}%

    Previous quarter: {stats['previous_quarter']}
    Previous YoY growth:
    {stats['previous_growth']}%

    Change in YoY growth:
    {stats['growth_change']} percentage points.

    Write concise economic commentary.

    Rules:
    - Use only the figures provided.
    - Compare the latest and previous growth.
    - Explain whether growth accelerated
      or decelerated.
    - Do not invent economic causes.
    - Clearly state that data is synthetic.
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EconomicReport,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned no text")

    report = EconomicReport.model_validate_json(
        response.text
    )

    # Validate structured numerical fields
    if report.quarter != stats["quarter"]:
        raise ValueError("AI returned wrong quarter")

    if abs(
        report.growth_rate - stats["growth"]
    ) > 0.01:
        raise ValueError("AI returned wrong growth")

    if abs(
        report.previous_growth
        - stats["previous_growth"]
    ) > 0.01:
        raise ValueError(
            "AI returned wrong previous growth"
        )

    return report


# =====================================
# 5. SAVE REPORT
# =====================================

def save_report(report, stats, filename):
    with open(
        filename, "w", encoding="utf-8"
    ) as file:
        file.write("AI ECONOMIC REPORT\n")
        file.write("==================\n\n")
        file.write("Synthetic data - educational use\n\n")

        file.write(
            f"Quarter: {stats['quarter']}\n"
        )
        file.write(
            f"GDP: {stats['gdp']:,.0f}\n"
        )
        file.write(
            f"YoY Growth: {stats['growth']:.2f}%\n"
        )
        file.write(
            "Previous YoY Growth: "
            f"{stats['previous_growth']:.2f}%\n"
        )
        file.write(
            "Growth Change: "
            f"{stats['growth_change']:+.2f} pp\n\n"
        )

        file.write("COMMENTARY\n")
        file.write(report.commentary + "\n\n")

        file.write("KEY OBSERVATION\n")
        file.write(report.key_observation + "\n")


# =====================================
# 6. MAIN PROGRAM
# =====================================

def main():
    input_file = "economic_data.csv"
    output_file = "economic_report.txt"

    print("Loading economic data...")
    df = load_data(input_file)

    print("Calculating GDP growth...")
    stats = calculate_growth(df)

    print("\nCalculated statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nGenerating AI commentary...")
    report = generate_commentary(stats)

    print("\n=== AI COMMENTARY ===")
    print(report.commentary)

    print("\n=== KEY OBSERVATION ===")
    print(report.key_observation)

    save_report(report, stats, output_file)

    print(f"\nReport saved to {output_file}")


if __name__ == "__main__":
    main()
