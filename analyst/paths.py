"""Where the project's data lives. Every module takes its paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DB_PATH = DATA_DIR / "economy.db"        # SQLite database (committed)
REPORTS_DIR = DATA_DIR / "reports"       # synthetic reports (committed)
INDEX_DIR = DATA_DIR / "chroma_db"       # vector index (generated, gitignored)
