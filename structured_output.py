
from google import genai
from google.genai import types
from pydantic import BaseModel

# Step 1: Define the structure of our economic data
class EconomicIndicator(BaseModel):
    indicator: str
    country: str
    definition: str
    components: list[str]
    importance: str
    example: str

# Step 2: Connect to Gemini
client = genai.Client()

# Step 3: Ask Gemini about Singapore's GDP
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=(
        "Explain Singapore's Gross Domestic Product (GDP). "
        "Include its definition, main components under "
        "the expenditure approach, economic importance, "
        "and a simple illustrative example."
    ),
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EconomicIndicator,
    ),
)

# Step 4: Convert the response into a Python object
economic_data = EconomicIndicator.model_validate_json(
    response.text
)

# Step 5: Display the structured information
print("\n=== ECONOMIC INDICATOR ===")
print("Indicator:", economic_data.indicator)
print("Country:", economic_data.country)

print("\nDefinition:")
print(economic_data.definition)

print("\nMain Components:")
for component in economic_data.components:
    print("-", component)

print("\nEconomic Importance:")
print(economic_data.importance)

print("\nExample:")
print(economic_data.example)
