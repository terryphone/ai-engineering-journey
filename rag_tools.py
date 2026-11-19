import re
from functools import lru_cache
from pathlib import Path

import chromadb

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
INDEX_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "economic_reports"
MAX_RESULTS = 4
MAX_CHUNK_CHARS = 1000  # short reports stay in one piece

# A quarter named in a query, e.g. '2020Q2' or '2020 q2'
QUARTER = re.compile(r"\b(\d{4})\s*q([1-4])\b", re.IGNORECASE)

# =====================================
# 1. SPLIT REPORTS INTO CHUNKS
# =====================================

def split_report(
    text: str,
    source: str,
    max_chars: int = MAX_CHUNK_CHARS
) -> list[dict]:
    """
    Split one report into chunks. Paragraphs are kept together until
    a chunk would pass max_chars, so a short report stays whole and
    a description is never separated from its cause.
    """
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:])

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    pieces, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)

    return [
        {
            "id": f"{source}#{i}",
            # The title is repeated in every chunk so it keeps its context
            "text": f"{title}\n{piece}",
            "source": source,
            "title": title,
            "chunk": i,
        }
        for i, piece in enumerate(pieces)
    ]


def load_chunks(reports_dir=REPORTS_DIR) -> list[dict]:
    chunks = []
    for path in sorted(Path(reports_dir).glob("*.md")):
        chunks.extend(split_report(path.read_text(encoding="utf-8"), path.name))
    return chunks


# =====================================
# 2. BUILD THE VECTOR INDEX
# =====================================

def build_index(reports_dir=REPORTS_DIR, index_dir=INDEX_DIR) -> int:
    """Embed every chunk and store it in ChromaDB. Returns the count."""
    chunks = load_chunks(reports_dir)

    if not chunks:
        raise ValueError(f"No .md reports found in {reports_dir}")

    client = chromadb.PersistentClient(path=str(index_dir))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete

    # Chroma embeds the documents with its built-in local model
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "title": c["title"], "chunk": c["chunk"]}
            for c in chunks
        ],
    )

    _get_collection.cache_clear()
    return len(chunks)


# =====================================
# 3. SEARCH (THE AGENT'S TOOL)
# =====================================

@lru_cache(maxsize=2)
def _get_collection(index_dir: str):
    client = chromadb.PersistentClient(path=index_dir)
    return client.get_collection(COLLECTION_NAME)


def _to_passages(found) -> list[dict]:
    """Turn a Chroma query result into the tool's passage dicts."""
    return [
        {
            "source": meta["source"],
            "text": text,
            "distance": round(distance, 3),
        }
        for text, meta, distance in zip(
            found["documents"][0],
            found["metadatas"][0],
            found["distances"][0],
        )
    ]


def search_reports(
    query: str,
    n_results: int = MAX_RESULTS,
    index_dir=None
) -> dict:
    """Find the report passages most relevant to a question."""

    if not (query or "").strip():
        return {"error": "Empty query"}

    index_dir = Path(index_dir or INDEX_DIR)

    if not index_dir.exists():
        return {"error": "Report index not built. Run: python build_index.py"}

    limit = max(1, min(int(n_results), 5))

    try:
        collection = _get_collection(str(index_dir))
        found = collection.query(query_texts=[query], n_results=limit)
    except Exception as error:
        return {"error": f"Search failed: {error}"}

    # Embedding search gives little weight to an exact quarter such as
    # '2020Q2', so passages from a review of a named quarter go first.
    period_passages = []
    for year, number in dict.fromkeys(QUARTER.findall(query)):
        try:
            period_passages += _to_passages(collection.query(
                query_texts=[query],
                n_results=limit,
                where={"source": f"{year}Q{number}_review.md"},
            ))
        except Exception:
            pass  # no review for that quarter: keep the normal results

    passages, seen = [], set()
    for passage in period_passages + _to_passages(found):
        key = (passage["source"], passage["text"])
        if key not in seen:
            seen.add(key)
            passages.append(passage)

    passages = passages[:limit]

    return {"passages": passages}