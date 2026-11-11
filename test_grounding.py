from grounding import check_grounding
from make_reports import REPORTS
from rag_tools import split_report


def trace_for(*report_names):
    """A fake agent trace in which these reports were retrieved."""
    passages = []
    for name in report_names:
        for chunk in split_report(REPORTS[name], name):
            passages.append({"source": name, "text": chunk["text"]})
    return [{
        "tool": "search_reports",
        "arguments": {"query": "x"},
        "result": {"passages": passages},
    }]


def test_real_quote_with_formatting_differences_passes():
    # Spacing before %, a [s] edit, a non-breaking hyphen and bold text,
    # all seen in real model output
    answer = (
        "The review says the 11.41 % YoY increase “largely reflect[s] a "
        "base effect: they compare against 2020Q2 and 2020Q3, when "
        "output was temporarily depressed” [2021Q2_review.md]. "
        "It adds that “underlying momentum was ordinary. Quarter‑on‑"
        "quarter growth in 2021Q2 was only 0.11 %” [2021Q2_review.md]."
    )
    assert check_grounding(answer, trace_for("2021Q2_review.md")) == []


def test_quote_with_ellipsis_passes():
    answer = (
        "“Output contracted sharply in 2020Q2. GDP fell 6.66 % from the "
        "previous quarter … All five sectors contracted. Trade fell the "
        "most (8.84 %) and Services the least (5.55 %).”【2020Q2_review.md】"
    )
    assert check_grounding(answer, trace_for("2020Q2_review.md")) == []


def test_citation_of_a_report_that_was_not_retrieved_fails():
    # The 2022Q1 report does not exist; the model invented it
    answer = (
        "The 2022Q1 review states that “the slowdown was driven "
        "primarily by weaker Manufacturing and Construction activity, "
        "which together fell short of their prior-quarter growth "
        "rates.” [2022Q1_review.md]"
    )
    problems = check_grounding(answer, trace_for("2026Q3_review.md"))

    assert any("2022Q1_review.md" in p for p in problems)
    assert any("quoted text" in p for p in problems)


def test_invented_quote_with_a_real_file_name_fails():
    answer = (
        "The report says “growth slowed because of a sudden collapse in "
        "consumer confidence across every sector” [2026Q3_review.md]."
    )
    problems = check_grounding(answer, trace_for("2026Q3_review.md"))
    assert any("quoted text" in p for p in problems)


def test_answer_without_citations_or_quotes_passes():
    assert check_grounding("The reports do not cover 2022Q1.", []) == []


def test_short_quotes_are_ignored():
    answer = 'Growth was "ordinary" and the term "base effect" applies.'
    assert check_grounding(answer, []) == []