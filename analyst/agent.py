import pandas as pd

from analyst.agent_core import run_tool_agent
from analyst.analyst_tools import build_tools, CHAT_TOOLS

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


def run_agent(df: pd.DataFrame, question: str, history=None):
    """
    Answer a question, using earlier messages as memory.
    Returns (answer, trace), where trace lists every tool call.
    """
    return run_tool_agent(
        SYSTEM_PROMPT,
        CHAT_TOOLS,
        build_tools(df),
        question,
        history,
        max_rounds=MAX_ROUNDS,
        memory_turns=MEMORY_TURNS,
    )
