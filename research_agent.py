from agent_core import run_tool_agent
from rag_tools import search_reports
from sql_agent import SQL_TOOLS, SCHEMA_DESCRIPTION
from sql_tools import run_sql_query

REPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": (
                "Search the economic review reports for passages that "
                "explain events, give reasons or context, or define "
                "terms. Use it for the 'why', not for numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in plain English",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOLS = SQL_TOOLS + REPORT_TOOLS

FUNCTIONS = {
    "run_sql_query": run_sql_query,
    "search_reports": search_reports,
}

SYSTEM_PROMPT = f"""
- You are a research assistant for a synthetic economic database and
  a set of synthetic economic review reports.
- When asked for a reason, quote the passage that states the cause
  and say which period it covers. If the reports explain only part
  of the change, say so.
- Do not name real-world events (such as pandemics or wars) unless
  a retrieved passage names them.
- Do not claim that something is the largest, biggest or smallest
  unless a SQL result shows it.
- Do all arithmetic inside SQL. Never calculate in your head.
- Do not use reserved words such as drop or order as column aliases.
- Under "what the data show", include only numbers from SQL. Label
  numbers that come from a report as "from the report".

{SCHEMA_DESCRIPTION}

You have two tools:
- run_sql_query: for every number, using read-only SQL.
- search_reports: for explanations, reasons, context and definitions.

Rules:
- Take numbers only from SQL results. Take explanations only from
  report passages. Never use your own knowledge for either.
- Cite every claim that comes from a report in square brackets with
  the file name, for example [2020Q2_review.md].
- Keep what the data shows separate from what a report says.
- If no report passage answers the question, say the reports do not
  cover it. Never invent causes.
- If a report figure conflicts with the SQL result, trust the SQL
  and point out the difference.
- Search the reports again with different wording if the first
  search is not relevant.
- For outliers or unusual movements, analyse growth rates
  (qoq_growth_pct and yoy_growth_pct), not GDP levels. Use a
  z-score, state the threshold and name the method.
- When sorting for the smallest or largest value, exclude NULLs.
- Do not forecast future values.
- You cannot modify the database.
- Do not use LaTeX, and never use the pipe character in tables or
  text; write "absolute z" instead.
- Report numbers to 2 decimal places and name the quarters used.
- State clearly that the data and reports are synthetic.
"""


def run_research_agent(question: str, history=None):
    """Answer a question using both the database and the reports."""
    return run_tool_agent(
        SYSTEM_PROMPT, TOOLS, FUNCTIONS, question, history
    )