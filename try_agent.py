from economic_analyst import load_data
from agent import run_agent

df = load_data("economic_data.csv")

questions = [
    "What was the average YoY growth from 2026Q1 to 2026Q3, "
    "and is the latest quarter above or below that average?",
    "And how does 2026Q2 compare with that average?",      # tests memory
    "Will growth improve in 2026Q4?",                      # should decline
]

history = []

for q in questions:
    print("\n" + "=" * 50)
    print("QUESTION:", q)

    answer, trace = run_agent(df, q, history)

    print("TOOLS USED:", [step["tool"] for step in trace])
    print("ANSWER:", answer)

    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})