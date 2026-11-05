import pandas as pd
import streamlit as st

from rag_tools import INDEX_DIR
from research_agent import run_research_agent
from sql_tools import DB_PATH

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔎",
    layout="wide"
)

st.title("Research Agent")
st.caption(
    "Combines SQL on the economy database with retrieval from "
    "economic review reports | All data and reports are synthetic"
)

if not DB_PATH.exists():
    st.error("economy.db not found. Run `python build_database.py`.")
    st.stop()

if not INDEX_DIR.exists():
    st.error(
        "Report index not found. Run `python make_reports.py` "
        "and then `python build_index.py`."
    )
    st.stop()

if "research_chat_history" not in st.session_state:
    st.session_state.research_chat_history = []


def show_trace(trace):
    """Show every SQL query and report passage the agent used."""
    if not trace:
        return

    with st.expander(f"Queries and sources ({len(trace)})"):
        for step in trace:
            result = step["result"]

            if step["tool"] == "run_sql_query":
                st.code(step["arguments"].get("query", ""), language="sql")

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.dataframe(
                        pd.DataFrame(
                            result["rows"], columns=result["columns"]
                        ),
                        hide_index=True
                    )

            else:
                query = step["arguments"].get("query", "")
                st.markdown(f"**Report search:** {query}")

                if "error" in result:
                    st.error(result["error"])
                else:
                    for passage in result["passages"]:
                        st.markdown(
                            f"`{passage['source']}` "
                            f"(distance {passage['distance']})"
                        )
                        st.caption(passage["text"])


if st.button("Clear conversation"):
    st.session_state.research_chat_history = []
    st.rerun()

for msg in st.session_state.research_chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        show_trace(msg.get("trace", []))

question = st.chat_input(
    "e.g. Why did GDP fall in 2020Q2, and by how much?"
)

if question:

    previous_messages = list(st.session_state.research_chat_history)

    st.session_state.research_chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the data and the reports..."):
            trace, is_error = [], False
            try:
                answer, trace = run_research_agent(
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

    st.session_state.research_chat_history.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "is_error": is_error,
    })