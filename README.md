# AI Economic Dashboard

A Streamlit dashboard that analyses quarterly GDP data and lets you ask questions about it in plain English. An LLM agent answers by calling Python functions, so every number comes from pandas and not from the model.

> **Note:** all data is synthetic and for educational use.

## Features

- **Data validation:** checks the CSV for missing quarters, duplicates, non-positive values and too little history.
- **GDP metrics and chart:** latest GDP, year-on-year (YoY) growth, change in growth, and a growth trend chart.
- **AI commentary:** generates a short economic commentary and key observation. The app checks that the model's quarter and growth figures match the calculated ones, and rejects the output if they don't.
- **Chat with your data:** an agent that answers questions using tool calling.
  - Remembers the recent conversation, so follow-ups like "how does 2026Q2 compare with that average?" work.
  - Chains several tool calls for multi-part questions.
  - Declines to forecast or invent causes that the data can't support.
  - Shows a "Tools used" panel under each answer, so you can see exactly what it called.
  - Stops after a fixed number of tool rounds and gives a best-effort answer instead of looping.

## How it works

```
User question
     |
     v
  agent.py  --- conversation memory + system rules
     |
     |  1. LLM picks a tool           (get_yoy_growth, average_yoy_growth, ...)
     |  2. Python runs it on the data (pandas, analyst_tools.py)
     |  3. Result goes back to the LLM
     |  4. Repeat until it can answer
     v
 Final answer + tool trace
```

The model never calculates anything itself. It only chooses which function to call and explains the result.

### Available tools

| Tool | What it does |
|---|---|
| `list_quarters` | Lists the quarters in the dataset |
| `get_gdp` | GDP level for one quarter |
| `get_yoy_growth` | YoY growth for one quarter |
| `get_gdp_series` | GDP and YoY growth for every quarter |
| `compare_quarters` | % change in GDP level between two quarters |
| `average_yoy_growth` | Average YoY growth over a range of quarters |
| `get_extreme_growth` | Quarter with the highest or lowest YoY growth |

## Project structure

```
app.py                  Streamlit dashboard and chat interface
agent.py                Agent loop: memory, multi-step tool calling, trace
analyst_tools.py        Tool functions and their schemas
economic_analyst.py     Data loading, validation, growth stats, AI commentary
ai_config.py            LLM client and model settings
economic_data.csv       Synthetic quarterly GDP data
test_analyst_tools.py   Offline unit tests (no API calls)
archives/               Earlier learning exercises
```

## Setup

Requires Python 3 (developed on 3.14) and an API key from [Cerebras](https://cloud.cerebras.ai).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export CEREBRAS_API_KEY="your-key-here"
streamlit run app.py
```

The key is read from the environment and is never stored in the repo. The LLM client uses an OpenAI-compatible API, so you can switch providers by editing `ai_config.py`.

## Example questions

- What was the average YoY growth from 2026Q1 to 2026Q3, and is the latest quarter above or below it?
- Which quarter had the weakest growth?
- Compare GDP levels in 2026Q1 and 2026Q3.
- *Will growth improve next quarter?* The agent declines, because the tools only describe historical data.

## Tests

```bash
pip install pytest
pytest test_analyst_tools.py -q
```

These tests cover the tool functions directly and run without an API key.

## Limitations

- The dataset has only 7 quarters, so YoY growth exists for just the last three.
- Conversation memory keeps the final answers, not the raw tool results, so an earlier mistake could carry into later turns.
- There is no forecasting, by design.

## Roadmap

- [x] LLM APIs and structured outputs
- [x] AI-powered data analysis with pandas and Streamlit
- [x] Function calling and tools
- [x] Conversational agent with memory
- [ ] SQL-powered agent (SQLite, safe text-to-SQL)
- [ ] RAG over economic reports (embeddings, ChromaDB)
- [ ] Agent evaluation and regression tests
- [ ] Power BI integration
- [ ] Deployment (FastAPI, Docker)