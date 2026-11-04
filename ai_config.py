
import os
from openai import OpenAI

MODEL_NAME = "gpt-oss-120b"


def get_ai_client():
    api_key = os.getenv("CEREBRAS_API_KEY")

    if not api_key:
        raise ValueError(
            "CEREBRAS_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
        timeout=60.0,
    )
