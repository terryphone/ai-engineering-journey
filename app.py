
from pathlib import Path

import pandas as pd
import streamlit as st

from economic_analyst import (
    load_data,
    calculate_growth,
    generate_commentary,
)

# AI-powered chatbot
from analyst_tools import ask_data_question
from agent import run_agent

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Economic Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("AI Economic Dashboard")

st.caption(
    "AI-powered economic analysis | "
    "Synthetic data for educational use"
)

# Initialise session state
if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "data_signature" not in st.session_state:
    st.session_state.data_signature = None


# =====================================
# 2. LOAD SAMPLE DATASET
# =====================================

st.header("1. Select Economic Data")

data_file = (
    Path(__file__).resolve().parent
    / "economic_data.csv"
)

st.info(
    "Using synthetic GDP data stored "
    "in the project directory."
)

if not data_file.exists():
    st.error("economic_data.csv not found.")
    st.stop()


# =====================================
# 3. LOAD AND VALIDATE DATA
# =====================================

try:
    df = load_data(data_file)
    stats = calculate_growth(df)

except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Invalid dataset: {error}")
    st.stop()

st.success("Dataset loaded successfully!")

# Reset previous AI results if data changes
data_signature = pd.util.hash_pandas_object(
    df, index=True
).sum()

if st.session_state.data_signature != data_signature:
    st.session_state.ai_report = None
    st.session_state.chat_history = []
    st.session_state.data_signature = data_signature


# =====================================
# 4. DISPLAY ECONOMIC DATA
# =====================================

st.header("2. Economic Data")

display_df = df.copy()

display_df["Quarter"] = (
    display_df["Quarter"].astype(str)
)

display_df["YoY_Growth"] = (
    display_df["GDP"].pct_change(4) * 100
).round(2)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# =====================================
# 5. DISPLAY GDP METRICS
# =====================================

st.header("3. GDP Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Latest GDP",
        value=f"{stats['gdp']:,.0f}"
    )

with col2:
    st.metric(
        label="YoY GDP Growth",
        value=f"{stats['growth']:.2f}%"
    )

with col3:
    st.metric(
        label="Change in YoY Growth",
        value=f"{stats['growth_change']:+.2f} pp"
    )


# =====================================
# 6. GDP GROWTH CHART
# =====================================

st.header("4. GDP Growth Trend")

chart_df = display_df.dropna(
    subset=["YoY_Growth"]
).set_index("Quarter")

st.line_chart(
    chart_df["YoY_Growth"],
    y_label="YoY Growth (%)",
    x_label="Quarter"
)


# =====================================
# 7. AI-GENERATED COMMENTARY
# =====================================

st.header("5. AI Economic Commentary")

if st.button(
    "Generate AI Commentary",
    type="primary"
):

    with st.spinner(
        "AI is analysing your GDP data..."
    ):

        try:
            report = generate_commentary(stats)

            # Save result between reruns
            st.session_state.ai_report = report

        except Exception as error:
            st.error(
                "Could not generate AI commentary. "
                f"Details: {error}"
            )


# Display saved commentary
if st.session_state.ai_report is not None:

    report = st.session_state.ai_report

    st.subheader("Economic Commentary")
    st.write(report.commentary)

    st.subheader("Key Observation")
    st.write(report.key_observation)

    # Prepare downloadable report
    report_text = (
        "AI ECONOMIC REPORT\n"
        "==================\n\n"
        "Synthetic data - educational use\n\n"
        f"Quarter: {stats['quarter']}\n"
        f"GDP: {stats['gdp']:,.0f}\n"
        f"YoY Growth: {stats['growth']:.2f}%\n"
        f"Previous YoY Growth: "
        f"{stats['previous_growth']:.2f}%\n"
        f"Growth Change: "
        f"{stats['growth_change']:+.2f} pp\n\n"
        f"Commentary:\n{report.commentary}\n\n"
        f"Key Observation:\n"
        f"{report.key_observation}\n"
    )

    st.download_button(
        label="Download Economic Report",
        data=report_text,
        file_name="economic_report.txt",
        mime="text/plain",
        on_click="ignore"
    )


# =====================================
# 8. AI ECONOMIC CHATBOT
# =====================================

st.header("6. Ask Questions About the Data")

st.caption(
    "Ask questions about your synthetic GDP dataset. "
    "The assistant remembers the recent conversation."
)


def show_trace(trace):
    """Show which tools the agent used, for transparency."""
    if trace:
        with st.expander(f"Tools used ({len(trace)})"):
            for step in trace:
                st.code(
                    f"{step['tool']}({step['arguments']})\n"
                    f"-> {step['result']}",
                    language="text",
                )


if st.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        show_trace(msg.get("trace", []))

question = st.chat_input(
    "e.g. What was the average YoY growth from 2026Q1 to 2026Q3?"
)

if question:

    # Memory = everything said before this question
    previous_messages = list(st.session_state.chat_history)

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("AI is analysing the data..."):
            trace, is_error = [], False
            try:
                answer, trace = run_agent(
                    df, question, previous_messages
                )
            except Exception as error:
                answer = (
                    "Unable to answer the question. "
                    f"Details: {error}"
                )
                is_error = True

        st.write(answer)
        show_trace(trace)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "is_error": is_error,
    })