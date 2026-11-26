from analyst.grounding import check_grounding
from analyst.make_reports import REPORTS
from analyst.rag_tools import split_report


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

# --- Percentages must come from SQL results or retrieved passages ---

def sql_step(*values):
    """A fake run_sql_query step whose result holds these values."""
    return {
        "tool": "run_sql_query",
        "arguments": {"query": "SELECT ..."},
        "result": {
            "columns": ["quarter", "value"],
            "rows": [["2026Q3", v] for v in values],
            "row_count": len(values),
            "truncated": False,
        },
    }


def test_percentage_computed_outside_sql_fails():
    # Real failure: 0.28 / 0.30 worked out by the model, not by SQL
    trace = [sql_step(3.21, 2.91, -0.30)] + trace_for("2026Q3_review.md")
    answer = (
        "The average fell from 3.21 % to 2.91 %. The reports explain "
        "about 93 % of the drop; the rest (≈7 %) is unexplained."
    )
    problems = check_grounding(answer, trace)

    assert any("93%" in p for p in problems)
    assert any("7%" in p for p in problems)
    assert not any("3.21%" in p or "2.91%" in p for p in problems)


def test_percentage_rounded_from_sql_passes():
    # SQL returns 3.0125; the prompt asks for 2 decimal places
    assert check_grounding("Average growth was 3.01 %.", [sql_step(3.0125)]) == []


def test_percentage_sign_and_spacing_differences_pass():
    # Non-breaking hyphen, narrow no-break space, and "fell 6.66 %" for -6.66
    answer = "Growth was ‑6.66 %, so GDP fell 6.66% in 2020Q2."
    assert check_grounding(answer, [sql_step(-6.66)]) == []


def test_percentage_from_a_retrieved_report_passes():
    answer = (
        "From the report: Trade growth slowed to 0.34 % from 1.77 % "
        "[2026Q3_review.md]."
    )
    assert check_grounding(answer, trace_for("2026Q3_review.md")) == []


def test_percentage_with_no_tools_called_fails():
    problems = check_grounding("GDP will grow 2.50 % next year.", [])
    assert any("2.50%" in p for p in problems)


def test_failed_sql_step_is_ignored():
    trace = [{
        "tool": "run_sql_query",
        "arguments": {"query": "SELECT nope"},
        "result": {"error": "no such column: nope"},
    }, sql_step(2.82)]
    assert check_grounding("Growth was 2.82 %.", trace) == []
