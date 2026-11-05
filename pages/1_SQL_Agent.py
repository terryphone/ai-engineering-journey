import pandas as pd
import streamlit as st

from sql_agent import run_sql_agent
from sql_tools import DB_PATH

st.set_page_config(
    page_title="SQL Agent",
    page_icon="🗄️",
    layout="wide"
)

st.title("SQL Agent")
st.caption(
    "Ask questions in plain English. The AI writes read-only SQL "
    "against a synthetic economy database | Synthetic data"
)

if not DB_PATH.exists():
    st.error(
        "economy.db not found. Run `python build_database.py` first."
    )
    st.stop()

if "sql_chat_history" not in st.session_state:
    st.session_state.sql_chat_history = []


def show_trace(trace):
    """Show every SQL query the agent ran, with its result."""
    if not trace:
        return

    with st.expander(f"SQL used ({len(trace)})"):
        for step in trace:
            st.code(step["arguments"].get("query", ""), language="sql")

            result = step["result"]

            if "error" in result:
                st.error(result["error"])
            else:
                st.dataframe(
                    pd.DataFrame(
                        result["rows"], columns=result["columns"]
                    ),
                    hide_index=True
                )
                if result.get("truncated"):
                    st.caption("Result truncated to the first rows.")


if st.button("Clear conversation"):
    st.session_state.sql_chat_history = []
    st.rerun()

for msg in st.session_state.sql_chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        show_trace(msg.get("trace", []))

question = st.chat_input(
    "e.g. Which sector grew fastest since 2020Q1?"
)

if question:

    previous_messages = list(st.session_state.sql_chat_history)

    st.session_state.sql_chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Writing and running SQL..."):
            trace, is_error = [], False
            try:
                answer, trace = run_sql_agent(
                    question, previous_messages
                )
            except Exception as error:
                answer = (
                    "Unable to answer the question. "
                    f"Details: {error}"
                )
                is_error = True

        st.write(answer)
        show_trace(trace)

    st.session_state.sql_chat_history.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "is_error": is_error,
    })