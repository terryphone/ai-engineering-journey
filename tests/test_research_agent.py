"""Offline tests for the research agent's source check, retry and refusal."""
from analyst import research_agent
from analyst.make_reports import REPORTS
from analyst.rag_tools import split_report
from analyst.research_agent import UNVERIFIED_ANSWER, run_research_agent


def search_step(name):
    """A trace step in which this report was retrieved."""
    passages = [
        {"source": name, "text": chunk["text"]}
        for chunk in split_report(REPORTS[name], name)
    ]
    return {
        "tool": "search_reports",
        "arguments": {"query": "x"},
        "result": {"passages": passages},
    }


def scripted(monkeypatch, attempts):
    """Make run_tool_agent return these (answer, trace) pairs in order."""
    calls = []

    def fake(system_prompt, tools, functions, question, history=None, **_):
        calls.append(question)
        return attempts[len(calls) - 1]

    monkeypatch.setattr(research_agent, "run_tool_agent", fake)
    return calls


GOOD = ("See [2020Q2_review.md].", [search_step("2020Q2_review.md")])
BAD_CITATION = ("See [2021Q2_review.md].", [search_step("2020Q2_review.md")])


def test_grounded_answer_is_returned_without_retry(monkeypatch):
    calls = scripted(monkeypatch, [GOOD])
    answer, _ = run_research_agent("q")
    assert answer == GOOD[0]
    assert len(calls) == 1


def test_failed_check_retries_once_with_the_problems(monkeypatch):
    calls = scripted(monkeypatch, [BAD_CITATION, GOOD])
    answer, trace = run_research_agent("q")
    assert answer == GOOD[0]
    assert len(calls) == 2
    assert "2021Q2_review.md, which was not retrieved" in calls[1]
    assert len(trace) == 2  # both attempts are shown to the user


def test_refuses_when_retry_also_fails(monkeypatch):
    scripted(monkeypatch, [BAD_CITATION, BAD_CITATION])
    answer, _ = run_research_agent("q")
    assert answer == UNVERIFIED_ANSWER


def test_retry_cannot_rely_on_first_attempt_sources(monkeypatch):
    # The retry cites a report that only the first attempt retrieved.
    # The retry runs in a fresh conversation, so this is not grounded.
    first = ("See [2021Q2_review.md].", [search_step("2020Q2_review.md"),
                                          search_step("2026Q3_review.md")])
    retry = ("See [2026Q3_review.md].", [search_step("2020Q2_review.md")])
    scripted(monkeypatch, [first, retry])
    answer, _ = run_research_agent("q")
    assert answer == UNVERIFIED_ANSWER
