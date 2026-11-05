from functools import lru_cache
from pathlib import Path

import chromadb

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
INDEX_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "economic_reports"
MAX_RESULTS = 4


# =====================================
# 1. SPLIT REPORTS INTO CHUNKS
# =====================================

def split_report(text: str, source: str) -> list[dict]:
    """Split one report into chunks, one per paragraph."""
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:])

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    return [
        {
            "id": f"{source}#{i}",
            # The title is repeated in every chunk so it keeps its context
            "text": f"{title}\n{paragraph}",
            "source": source,
            "title": title,
            "chunk": i,
        }
        for i, paragraph in enumerate(paragraphs)
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

    try:
        collection = _get_collection(str(index_dir))
        found = collection.query(
            query_texts=[query],
            n_results=max(1, min(int(n_results), 5)),
        )
    except Exception as error:
        return {"error": f"Search failed: {error}"}

    passages = [
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

    return {"passages": passages}