import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "economy.db"

MAX_ROWS = 50        # never send more than this to the model
MAX_SECONDS = 2.0    # abort queries that run longer

# The only things a query is allowed to do
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _authorizer(action, arg1, arg2, db_name, source):
    """SQLite calls this for every operation a query tries to do."""
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def run_sql_query(query: str, db_path=None) -> dict:
    """Run one read-only SELECT query and return the rows."""

    db_path = Path(db_path or DB_PATH)

    query = (query or "").strip().rstrip(";").strip()

    if not query:
        return {"error": "Empty query"}

    # Layer 1: only SELECT / WITH statements
    first_word = query.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return {"error": "Only SELECT queries are allowed"}

    # Layer 2: database opened read-only
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)

    # Layer 3: SQLite itself rejects anything except reading
    conn.set_authorizer(_authorizer)

    # Layer 4: stop runaway queries
    start = time.monotonic()

    def _too_slow():
        return 1 if time.monotonic() - start > MAX_SECONDS else 0

    conn.set_progress_handler(_too_slow, 1000)

    try:
        # sqlite3 refuses to run more than one statement at a time
        cursor = conn.execute(query)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as error:
        return {"error": str(error)}
    finally:
        conn.close()

    # Layer 5: cap the result size
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]

    return {
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }