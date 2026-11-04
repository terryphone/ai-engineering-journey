import json

import pandas as pd

from ai_config import get_ai_client, MODEL_NAME
from analyst_tools import build_tools, CHAT_TOOLS

MAX_ROUNDS = 6      # tool-calling rounds before we stop the agent
MEMORY_TURNS = 6    # remember the last 6 question/answer pairs

SYSTEM_PROMPT = """
You are an economic data analyst.

The dataset contains synthetic quarterly GDP data.

Rules:
- Always use the provided tools for numerical data.
- For multi-part questions, work step by step: call the tools
  you need, then combine the results in your answer.
- Use the earlier conversation to resolve references such as
  "that quarter" or "compare it with the previous one".
- Never invent figures or economic explanations.
- Do not forecast or predict future values. Explain that the
  tools only describe historical data.
- Clearly state that data is synthetic.
- Distinguish changes in GDP levels from changes in
  year-on-year growth rates.
- If a tool returns an error, explain it.
- If available tools cannot answer a question,
  explain the limitation.
"""


def _memory(history):
    """Turn the chat history into model-ready messages."""
    usable = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if not m.get("is_error")        # never remember error messages
    ]
    return usable[-MEMORY_TURNS * 2:]   # keep only the recent past


def run_agent(df: pd.DataFrame, question: str, history=None):
    """
    Answer a question, using earlier messages as memory.
    Returns (answer, trace), where trace lists every tool call.
    """
    client = get_ai_client().with_options(max_retries=5)
    functions = build_tools(df)

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + _memory(history or [])
        + [{"role": "user", "content": question}]
    )

    trace = []

    for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=CHAT_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        # No tool requested: this is the final answer
        if not message.tool_calls:
            if not message.content:
                raise ValueError("Cerebras returned no text response")
            return message.content, trace

        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            arguments = {}

            if name not in functions:
                result = {"error": "Unknown tool requested"}
            else:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = functions[name](**arguments)
                except (ValueError, TypeError, KeyError) as error:
                    result = {"error": str(error)}

            trace.append({
                "tool": name,
                "arguments": arguments,
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    # Out of rounds: force a best-effort answer instead of crashing
    messages.append({
        "role": "user",
        "content": (
            "You have used all available tool calls. Answer now "
            "using only the results above, and say what is missing."
        ),
    })
    final = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=CHAT_TOOLS,
        tool_choice="none",
        temperature=0,
    )
    return final.choices[0].message.content or "No answer produced.", trace