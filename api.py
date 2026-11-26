"""REST API for the three agents and the GDP metrics.

Run locally:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive documentation.

The /metrics endpoint is pure pandas and needs no API key.
The /ask endpoints call the LLM and need CEREBRAS_API_KEY.
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analyst import agent
from analyst import research_agent
from analyst import sql_agent
from analyst.economic_analyst import calculate_growth, load_data
from analyst.rag_tools import INDEX_DIR
from analyst.sql_tools import DB_PATH

MAX_QUESTION_CHARS = 1000
MAX_HISTORY_MESSAGES = 20  # the agents only use the last few anyway

app = FastAPI(
    title="AI Economic Analyst API",
    description=(
        "Tool-calling LLM agents over synthetic economic data. Every "
        "number comes from pandas or SQL, never from the model. "
        "All data is synthetic and for educational use."
    ),
    version="1.0.0",
)


# =====================================
# 1. REQUEST AND RESPONSE MODELS
# =====================================

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=10_000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "Which quarter had the weakest growth?",
                "history": [],
            }]
        }
    }


class ToolStep(BaseModel):
    tool: str
    arguments: dict
    result: dict


class Answer(BaseModel):
    answer: str
    trace: list[ToolStep]


class Metrics(BaseModel):
    quarter: str
    gdp: float
    growth: float = Field(description="Latest YoY growth, %")
    previous_quarter: str
    previous_growth: float = Field(description="Previous YoY growth, %")
    growth_change: float = Field(description="Change in YoY growth, pp")


# =====================================
# 2. ENDPOINTS WITHOUT THE LLM
# =====================================

@app.get("/health")
def health() -> dict:
    """Liveness check, plus which data files are available."""
    return {
        "status": "ok",
        "database": DB_PATH.exists(),
        "report_index": INDEX_DIR.exists(),
    }


@app.get("/metrics", response_model=Metrics)
def metrics() -> dict:
    """Latest GDP and YoY growth from the database (no LLM)."""
    try:
        return calculate_growth(load_data())
    except ValueError as error:
        raise HTTPException(500, f"Invalid dataset: {error}")


# =====================================
# 3. AGENT ENDPOINTS
# =====================================

def _run(run_agent, question: Question, *extra) -> dict:
    """Call an agent and map its failures to HTTP errors."""
    history = [m.model_dump() for m in question.history]
    try:
        answer, trace = run_agent(*extra, question.question, history)
    except ValueError as error:
        # get_ai_client raises ValueError when the key is missing
        if "CEREBRAS_API_KEY" in str(error):
            raise HTTPException(503, "LLM API key is not configured")
        raise HTTPException(502, f"Agent failed: {error}")
    except Exception as error:
        raise HTTPException(502, f"Agent failed: {type(error).__name__}")
    return {"answer": answer, "trace": trace}


@app.post("/ask/dashboard", response_model=Answer)
def ask_dashboard(question: Question) -> dict:
    """Ask the dashboard agent (pandas tools over quarterly GDP)."""
    try:
        df = load_data()
    except ValueError as error:
        raise HTTPException(500, f"Invalid dataset: {error}")
    return _run(agent.run_agent, question, df)


@app.post("/ask/sql", response_model=Answer)
def ask_sql(question: Question) -> dict:
    """Ask the SQL agent (read-only text-to-SQL over 27 quarters)."""
    if not DB_PATH.exists():
        raise HTTPException(503, "Database not built: run build_database.py")
    return _run(sql_agent.run_sql_agent, question)


@app.post("/ask/research", response_model=Answer)
def ask_research(question: Question) -> dict:
    """Ask the research agent (SQL + report search, with source checks)."""
    if not DB_PATH.exists():
        raise HTTPException(503, "Database not built: run build_database.py")
    if not INDEX_DIR.exists():
        raise HTTPException(503, "Report index not built: run build_index.py")
    return _run(research_agent.run_research_agent, question)
