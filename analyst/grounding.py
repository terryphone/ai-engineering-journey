import re

# Matches [file.md] and the full-width 【file.md】 the model sometimes uses
CITATION = re.compile(r"[\[【]\s*([\w.\-]+\.md)\s*[\]】]")

# Matches text inside curly or straight double quotes
QUOTE = re.compile(r"[“\"]([^”\"]+)[”\"]")

# Matches a percentage such as '6.66 %', '6.66%' or '6.66\u202f%' (the sign is ignored)
PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")

# Matches any number in report text
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

MIN_QUOTE_WORDS = 6    # ignore very short quotes
MIN_COVERAGE = 0.7     # share of a quote that must appear in the sources


def _words(text: str) -> list[str]:
    """Lowercase words and numbers, ignoring formatting differences."""
    text = text.lower()
    text = re.sub(r"\[\w{1,3}\]", "", text)      # editing marks: reflect[s]
    text = re.sub(r"[\u2010-\u2015-]", " ", text)  # hyphens of every kind
    text = re.sub(r"\s*%", "%", text)             # '6.66 %' -> '6.66%'
    text = text.replace(",", "")                  # '123,619' -> '123619'
    return re.findall(r"\d+(?:\.\d+)?%?|[a-z]+", text)


def _shingles(words: list[str], size: int = 3) -> set:
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def _retrieved(trace) -> tuple[set, list[str]]:
    """Return (file names, passage texts) from every report search."""
    sources, texts = set(), []

    for step in trace:
        if step["tool"] != "search_reports":
            continue
        for passage in step["result"].get("passages", []):
            sources.add(passage["source"])
            texts.append(passage["text"])

    return sources, texts


def _known_numbers(trace, texts: list[str]) -> list[float]:
    """Every number in a successful SQL result or a retrieved passage."""
    numbers = []

    for step in trace:
        if step["tool"] != "run_sql_query":
            continue
        for row in step["result"].get("rows", []):
            for value in row:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numbers.append(float(value))

    for text in texts:
        numbers += [float(n.replace(",", "")) for n in NUMBER.findall(text)]

    return numbers


def _matches(written: str, known: list[float]) -> bool:
    """True if some known number rounds to the figure as written (ignoring sign)."""
    value = float(written.replace(",", ""))
    decimals = len(written.split(".")[1]) if "." in written else 0
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(abs(n) - value) <= tolerance for n in known)


def check_grounding(answer: str, trace) -> list[str]:
    """
    Check that every cited file and every quotation in the answer
    really comes from a passage the agent retrieved, and that every
    percentage comes from an SQL result or a retrieved passage.
    Returns a list of problems (empty means all checks passed).
    """
    sources, texts = _retrieved(trace)
    problems = []

    # 1. Citations must point to a report that was actually retrieved
    for name in sorted(set(CITATION.findall(answer))):
        if name not in sources:
            problems.append(f"cited {name}, which was not retrieved")

    # 2. Quotations must appear in the retrieved text
    known = set()
    for text in texts:
        known |= _shingles(_words(text))

    for quote in QUOTE.findall(answer):
        # Quotes with "..." or an ellipsis are checked piece by piece
        for piece in re.split(r"…|\.\.\.", quote):
            words = _words(piece)
            if len(words) < MIN_QUOTE_WORDS:
                continue

            pieces = _shingles(words)
            coverage = len(pieces & known) / len(pieces)

            if coverage < MIN_COVERAGE:
                problems.append(
                    "quoted text not found in the retrieved passages: "
                    f"'{piece.strip()[:60]}...'"
                )

    # 3. Percentages must come from SQL results or report passages,
    #    never from the model's own arithmetic
    numbers = _known_numbers(trace, texts)
    seen = set()
    for written in PERCENT.findall(answer):
        if written in seen:
            continue
        seen.add(written)
        if not _matches(written, numbers):
            problems.append(
                f"percentage {written}% not found in any SQL result or "
                "retrieved passage; compute it with run_sql_query"
            )

    return problems