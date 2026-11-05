import json

from ai_config import get_ai_client, MODEL_NAME
from sql_tools import run_sql_query

MAX_ROUNDS = 6      # tool-calling rounds before we stop the agent
MEMORY_TURNS = 6    # remember the last 6 question/answer pairs

SCHEMA_DESCRIPTION = """
Database: SQLite. Synthetic quarterly economic data for 27 quarters
(2020Q1 to 2026Q3). Output values are in synthetic units.

TABLE sector_output(quarter TEXT, sector TEXT, output REAL)
  - One row per quarter and sector.
  - quarter looks like '2026Q3' and sorts correctly as text.
  - sector is one of: Manufacturing, Services, Construction,
    Finance, Trade.

VIEW gdp_quarterly(quarter, gdp)
  - Total output per quarter (sum of all sectors).

VIEW gdp_growth(quarter, gdp, yoy_growth_pct, qoq_growth_pct)
  - yoy_growth_pct: year-on-year growth in percent
    (NULL for the first four quarters).
  - qoq_growth_pct: quarter-on-quarter growth in percent
    (NULL for the first quarter).are).
"""

SYSTEM_PROMPT = f"""
- You are a data analyst assistant for a synthetic economic database.
- For outliers or unusual movements, analyse growth rates
  (qoq_growth_pct and yoy_growth_pct), not GDP levels, because
  levels trend upward. Use a z-score, state the threshold, and
  name the method you used.
- When sorting for the smallest or largest value, exclude NULLs
  with WHERE column IS NOT NULL.
- Do the arithmetic inside SQL rather than retyping numbers from
  earlier results.
- Never use the pipe character in tables or text; write
  "absolute z" instead.

{SCHEMA_DESCRIPTION}

Rules:
- Use the run_sql_query tool for every number. Never guess figures.
- Write a single read-only SQLite SELECT statement per call.
- Prefer the views gdp_quarterly and gdp_growth over recomputing
  growth yourself.
- For quarter-on-quarter change, use
  LAG(gdp) OVER (ORDER BY quarter) on gdp_quarterly.
- If a query returns an error, read the message, fix the SQL and
  try again.
- If a result is marked truncated, refine the query with
  aggregation or filters and say so.
- Never invent causes: the data contains no explanations.
- Do not forecast future values.
- You cannot modify the database. If asked to change or delete
  data, explain that you only have read access.
- Report numbers to 2 decimal places and name the quarters used.
- Clearly state that the data is synthetic.
- Do not use LaTeX; write formulas in plain text.
"""

SQL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": (
                "Run one read-only SQLite SELECT query against the "
                "economy database and return the rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SQLite SELECT statement",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

FUNCTIONS = {"run_sql_query": run_sql_query}


def _memory(history):
    """Turn the chat history into model-ready messages."""
    usable = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if not m.get("is_error")
    ]
    return usable[-MEMORY_TURNS * 2:]


def run_sql_agent(question: str, history=None):
    """
    Answer a question by writing and running SQL.
    Returns (answer, trace), where trace lists every query run.
    """
    client = get_ai_client().with_options(max_retries=5)

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
            tools=SQL_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            if not message.content:
                raise ValueError("Cerebras returned no text response")
            return message.content, trace

        messages.append(message)

        for call in message.tool_calls:
            name = call.function.name
            arguments = {}

            if name not in FUNCTIONS:
                result = {"error": "Unknown tool requested"}
            else:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = FUNCTIONS[name](**arguments)
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
            "You have used all available queries. Answer now using "
            "only the results above, and say what is missing."
        ),
    })
    final = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=SQL_TOOLS,
        tool_choice="none",
        temperature=0,
    )
    return final.choices[0].message.content or "No answer produced.", trace