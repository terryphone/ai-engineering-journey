import re

# Matches [file.md] and the full-width 【file.md】 the model sometimes uses
CITATION = re.compile(r"[\[【]\s*([\w.\-]+\.md)\s*[\]】]")

# Matches text inside curly or straight double quotes
QUOTE = re.compile(r"[“\"]([^”\"]+)[”\"]")

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


def check_grounding(answer: str, trace) -> list[str]:
    """
    Check that every cited file and every quotation in the answer
    really comes from a passage the agent retrieved.
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

    return problems