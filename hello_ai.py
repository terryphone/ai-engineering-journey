
from google import genai

# Create Gemini client
client = genai.Client()

# Send a request to the AI model
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is Psyduck? or Ya Ya?"
)

# Display the response
print(response.text)
