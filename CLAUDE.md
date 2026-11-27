# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project: LLM agents that answer questions about **synthetic** economic data. The core design rule across every agent is that the model never computes or invents numbers. It only chooses tools; pandas or SQL produce every figure, and report passages supply every explanation. System prompts state this rule, and code enforces it.

## Rules for Claude Code

- Do not run `scripts/try_*.py`, `scripts/evaluate.py` or anything else that calls the Cerebras API unless asked; it uses API quota.
- Run `pytest -q` after every change. All tests must pass before committing.
- Never edit `data/reports/*.md` by hand. Change `analyst/make_reports.py` and re-run it.
- Never commit API keys, `.env` or `data/chroma_db/`.
- Take file locations from `analyst/paths.py`; do not build paths from `__file__` elsewhere.

## Commands

```bash
source .venv/bin/activate
export CEREBRAS_API_KEY=...          # needed only for agents/app, not tests

# Generated data (rebuild after changing the generators)
python -m analyst.build_database     # -> data/economy.db (SQLite, fixed seed 42)
python -m analyst.make_reports       # -> data/reports/*.md (from REPORTS dict)
python -m scripts.build_index        # -> data/chroma_db/ (gitignored)

streamlit run app.py                 # main dashboard; pages/ adds SQL + Research agents
uvicorn api:app --reload             # REST API; docs at /docs
docker build -t economic-analyst-api .

pytest -q                            # all offline tests
pytest tests/test_grounding.py -q          # one file
pytest tests/test_rag_tools.py -k split -q # one test by name
```

## Layout

`app.py` and `api.py` are the entry points and stay at the root (Streamlit needs `pages/` next to `app.py`). All library code is the `analyst/` package and is imported as `from analyst.x import y`. Data is in `data/`, tests in `tests/` (`pytest.ini` sets `pythonpath = .`), and live-API scripts in `scripts/`. Run scripts and generators as modules from the root (`python -m scripts.try_research_agent`) so `analyst` is importable.

## Architecture

**LLM client:** `ai_config.py` returns an OpenAI-SDK client pointed at Cerebras (`gpt-oss-120b`). Switching providers happens only here.

**Agent loop:** all agents use the same loop. The model calls tools until it answers, with up to `max_rounds` rounds. Then a final call with `tool_choice="none"` forces a best-effort answer. Each agent returns `(answer, trace)`, and the UI shows the trace as "Tools used". Memory keeps only the last N user/assistant turns, skipping any message flagged `is_error`. The loop lives only in `agent_core.run_tool_agent`; `agent.py`, `sql_agent.py` and `research_agent.py` are thin wrappers that pass their own prompt, tools and limits. `test_agent_core.py` covers it with a fake client.

**Three agents, three data sources:**
- `agent.py` + `analyst_tools.py`: pandas tools over quarterly GDP that `economic_analyst.load_data` reads from the `gdp_quarterly` view. Used by `app.py` (dashboard and chat) and `/ask/dashboard`. Everything reads `economy.db`, so there is one source of truth; `test_economic_analyst.py` checks the dashboard's growth matches the `gdp_growth` view.
- `sql_agent.py` + `sql_tools.py`: text-to-SQL over `economy.db` (27 quarters, `sector_output` table plus `gdp_quarterly` / `gdp_growth` views). `run_sql_query` is safe in layers: SELECT/WITH only, a read-only URI connection, a SQLite authorizer allowlist, a 2s progress-handler timeout and a 50-row cap with a `truncated` flag. Keep all of these layers.
- `research_agent.py`: SQL tools plus `search_reports` (`rag_tools.py`, ChromaDB with its built-in local embeddings). Afterwards, `grounding.check_grounding` verifies that every `[file.md]` citation was actually retrieved, every quotation of 6+ words matches retrieved text (3-word shingles, ≥70% coverage), and every percentage matches a number in an SQL result or retrieved passage. On failure the agent retries once with the problems listed; the retry is checked only against its own trace, since it runs in a fresh conversation. If the retry also fails, it refuses rather than show an unverified answer.

**REST API:** `api.py` (FastAPI) wraps the same three agents plus a no-LLM `/metrics` endpoint. It calls agents through their modules (`sql_agent.run_sql_agent`) so `test_api.py` can monkeypatch them. A missing API key maps to 503, other agent failures to 502.

**Tool error contract:** tool functions return `{"error": ...}` dicts instead of raising, and the loop converts `ValueError`/`TypeError`/`KeyError` into the same shape. The model is expected to read errors and retry.

**Data consistency:** `make_reports.py` holds fictional reports whose figures must match `economy.db`. `test_rag_tools.py::test_report_numbers_match_the_database` checks this, so changing `build_database.py` (seed, growth rates, shocks) means updating the report text too.

**Chunking:** `split_report` keeps whole paragraphs together up to 1000 chars and prefixes every chunk with the report title, so short reports stay as one chunk. Rebuild the index after changing reports or chunking.

## Conventions

- Prompt rules for agents (no forecasting, no invented causes, state that the data is synthetic, 2 d.p., no LaTeX, no pipe characters) are deliberate fixes for observed model behaviour. Keep them when editing prompts.
- Tests must stay offline (no API key). They build temp databases with `build_database(tmp_path / ...)` and fake traces for grounding.
