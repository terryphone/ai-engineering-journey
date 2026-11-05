from research_agent import run_research_agent

questions = [
    "Why did GDP fall in 2020Q2, and by how much?",
    "Is the 11.41% YoY growth in 2021Q2 a genuine boom?",
    "Why did growth slow in 2026Q3, and which sectors drove it?",
    "What will growth be in 2027?",
    "What did the reports say about the 2022Q1 slowdown?",
    "What was the average YoY growth over the last four quarters, "
    "how does it compare with the four quarters before that, and "
    "does the latest report give a reason for the change?",
]

history = []

for q in questions:
    print("\n" + "=" * 50)
    print("QUESTION:", q)

    answer, trace = run_research_agent(q, history)

    for step in trace:
        print(f"\n{step['tool']}: {step['arguments']}")

    print("\nANSWER:", answer)

    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})