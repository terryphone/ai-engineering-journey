
import pandas as pd
import streamlit as st

# Reuse functions from Lesson 4
from economic_analyst import (
    load_data,
    calculate_growth,
    generate_commentary,
)

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

# Store AI results between Streamlit reruns
if "ai_report" not in st.session_state:
    st.session_state.ai_report = None
    
# =====================================
# 2. LOAD SAMPLE DATASET
# =====================================

from pathlib import Path

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
    use_container_width=True,
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

# Generate commentary when button is clicked
if st.button(
    "Generate AI Commentary",
    type="primary"
):
    with st.spinner(
        "Gemini is analysing your GDP data..."
    ):
        try:
            report = generate_commentary(stats)

            # Save the response in session state
            st.session_state.ai_report = report

        except Exception as error:
            st.error(
                "Could not generate AI commentary. "
                f"Details: {error}"
            )

# Display saved commentary
# This remains visible after Streamlit reruns
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
        f"Growth Change: "
        f"{stats['growth_change']:+.2f} pp\n\n"
        f"Commentary:\n{report.commentary}\n\n"
        f"Key Observation:\n"
        f"{report.key_observation}"
    )

    st.download_button(
        label="Download Economic Report",
        data=report_text,
        file_name="economic_report.txt",
        mime="text/plain",
        on_click="ignore"
    )