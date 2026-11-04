
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel

# =====================================
# STEP 1: Create synthetic GDP data
# =====================================

data = {
    "Quarter": [
        "2025Q1", "2025Q2", "2025Q3", "2025Q4",
        "2026Q1", "2026Q2", "2026Q3"
    ],
    "GDP": [
        145000, 148000, 151000, 154000,
        150000, 155000, 160000
    ]
}

df = pd.DataFrame(data)

print("=== ORIGINAL GDP DATA ===")
print(df)

# =====================================
# STEP 2: Calculate YoY growth
# =====================================

df["YoY_Growth"] = (
    df["GDP"].pct_change(periods=4) * 100
)

df["YoY_Growth"] = df["YoY_Growth"].round(2)

print("\n=== GDP WITH YOY GROWTH ===")
print(df)

# =====================================
# STEP 3: Extract latest quarter
# =====================================

latest = df.iloc[-1]

quarter = latest["Quarter"]
gdp = float(latest["GDP"])
growth = float(latest["YoY_Growth"])

previous = df.iloc[-2]

previous_quarter = previous["Quarter"]
previous_growth = float(previous["YoY_Growth"])

growth_change = round(
    growth - previous_growth, 2
)

print("\nLatest quarter:", quarter)
print("GDP:", gdp)
print("YoY Growth:", growth, "%")

# =====================================
# STEP 4: Define structured output
# =====================================

class GDPAnalysis(BaseModel):
    quarter: str
    growth_rate: float
    commentary: str
    key_observation: str

# =====================================
# STEP 5: Send data to Gemini
# =====================================

client = genai.Client()

prompt = f"""
You are an economic analyst.

The following GDP data is synthetic.

Latest quarter: {quarter}
Latest GDP: {gdp}
Latest YoY growth: {growth}%

Previous quarter: {previous_quarter}
Previous YoY growth: {previous_growth}%

Change in YoY growth: {growth_change}
percentage points.

Write a concise economic commentary.

Requirements:
1. State the latest GDP growth rate.
2. Compare it with the previous quarter's
   YoY growth rate.
3. Explain whether growth accelerated,
   decelerated, or remained unchanged.
4. Do not invent economic explanations.
5. Clearly identify the data as synthetic.
"""

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GDPAnalysis,
    ),
)

analysis = GDPAnalysis.model_validate_json(
    response.text
)

# =====================================
# STEP 6: Validate numerical output
# =====================================

if analysis.quarter != quarter:
    raise ValueError("Incorrect quarter returned")

if abs(analysis.growth_rate - growth) > 0.01:
    raise ValueError("Incorrect growth rate returned")

# =====================================
# STEP 7: Display the commentary
# =====================================

print("\n=== AI GDP ANALYSIS ===")
print("Quarter:", analysis.quarter)
print("Growth Rate:", analysis.growth_rate, "%")

print("\nCommentary:")
print(analysis.commentary)

print("\nKey Observation:")
print(analysis.key_observation)

# =====================================
# STEP 8: Save the commentary
# =====================================

with open(
    "gdp_commentary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(
        f"Quarter: {analysis.quarter}\n"
        f"Growth: {analysis.growth_rate}%\n\n"
        f"Commentary:\n{analysis.commentary}\n\n"
        f"Key Observation:\n"
        f"{analysis.key_observation}\n"
    )

print("\nCommentary saved to gdp_commentary.txt")
