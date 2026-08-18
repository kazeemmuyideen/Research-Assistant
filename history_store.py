import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

DB_PATH = Path(__file__).parent / "research_history.db"


def _get_secret(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    try:
        import streamlit as st

        for name in names:
            val = st.secrets.get(name)
            if val:
                return val
    except Exception:
        pass
    return None


SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY")
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

_client = None
if USE_SUPABASE:
    from supabase import create_client

    _client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE = "research_history"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=False)
def _supabase_call_with_retry(fn):
    return fn()


def _supabase_call(fn):
    """
    Runs a Supabase call with retries for transient network drops
    (e.g. httpx.RemoteProtocolError: Server disconnected). Returns None
    on total failure instead of raising, so a network hiccup degrades to
    an empty/failed result rather than crashing the whole Streamlit app.
    """
    try:
        return _supabase_call_with_retry(fn)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Supabase backend
# ---------------------------------------------------------------------------
def _supabase_init_db():
    # Table must already exist — Supabase's Python client can't run DDL.
    # Run this once in the Supabase SQL editor:
    #
    # create table research_history (
    #   id bigint generated always as identity primary key,
    #   timestamp timestamptz not null default now(),
    #   query text not null,
    #   topic text,
    #   summary text,
    #   full_report text,
    #   sources jsonb,
    #   tools_used jsonb,
    #   error text
    # );
    #
    # If you already created the table before full_report existed, add it with:
    #   alter table research_history add column if not exists full_report text;
    pass


def _supabase_save_entry(query, structured=None, error=None) -> int:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "topic": structured.topic if structured else None,
        "summary": structured.summary if structured else None,
        "full_report": structured.full_report if structured else None,
        "sources": structured.sources if structured else None,
        "tools_used": structured.tools_used if structured else None,
        "error": error,
    }
    resp = _supabase_call(lambda: _client.table(TABLE).insert(row).execute())
    return resp.data[0]["id"] if resp and resp.data else None


def _supabase_load_history(limit=50):
    resp = _supabase_call(
        lambda: _client.table(TABLE).select("*").order("id", desc=True).limit(limit).execute()
    )
    if resp is None:
        return []
    entries = []
    for row in resp.data:
        entries.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "query": row["query"],
                "topic": row.get("topic"),
                "summary": row.get("summary"),
                "full_report": row.get("full_report"),
                "sources": row.get("sources") or [],
                "tools_used": row.get("tools_used") or [],
                "error": row.get("error"),
            }
        )
    return entries


def _supabase_clear_history():
    _supabase_call(lambda: _client.table(TABLE).delete().gte("id", 0).execute())


def _supabase_delete_entry(entry_id: int):
    _supabase_call(lambda: _client.table(TABLE).delete().eq("id", entry_id).execute())


# ---------------------------------------------------------------------------
# SQLite backend (local fallback — used automatically if Supabase isn't configured)
# ---------------------------------------------------------------------------
def _sqlite_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_init_db():
    with _sqlite_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                topic TEXT,
                summary TEXT,
                full_report TEXT,
                sources TEXT,
                tools_used TEXT,
                error TEXT
            )
            """
        )
        # Migration for databases created before full_report existed —
        # CREATE TABLE IF NOT EXISTS won't retroactively add a column to an
        # existing table, so add it explicitly and ignore if it's already there.
        try:
            conn.execute("ALTER TABLE research_history ADD COLUMN full_report TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()


def _sqlite_save_entry(query, structured=None, error=None) -> int:
    with _sqlite_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO research_history (timestamp, query, topic, summary, full_report, sources, tools_used, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                query,
                structured.topic if structured else None,
                structured.summary if structured else None,
                structured.full_report if structured else None,
                json.dumps(structured.sources) if structured else None,
                json.dumps(structured.tools_used) if structured else None,
                error,
            ),
        )
        conn.commit()
        return cur.lastrowid


def _sqlite_load_history(limit=50):
    with _sqlite_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM research_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for row in rows:
        entries.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "query": row["query"],
                "topic": row["topic"],
                "summary": row["summary"],
                "full_report": row["full_report"] if "full_report" in row.keys() else None,
                "sources": json.loads(row["sources"]) if row["sources"] else [],
                "tools_used": json.loads(row["tools_used"]) if row["tools_used"] else [],
                "error": row["error"],
            }
        )
    return entries


def _sqlite_clear_history():
    with _sqlite_connect() as conn:
        conn.execute("DELETE FROM research_history")
        conn.commit()


def _sqlite_delete_entry(entry_id: int):
    with _sqlite_connect() as conn:
        conn.execute("DELETE FROM research_history WHERE id = ?", (entry_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Public interface — dispatches to whichever backend is active.
# app.py (and everything else) only ever calls these.
# ---------------------------------------------------------------------------
def backend_name() -> str:
    return "supabase" if USE_SUPABASE else "sqlite (local)"


def init_db():
    (_supabase_init_db if USE_SUPABASE else _sqlite_init_db)()


def save_entry(query: str, structured=None, error: str | None = None) -> int:
    return (_supabase_save_entry if USE_SUPABASE else _sqlite_save_entry)(query, structured, error)


def load_history(limit: int = 50):
    return (_supabase_load_history if USE_SUPABASE else _sqlite_load_history)(limit)


def clear_history():
    (_supabase_clear_history if USE_SUPABASE else _sqlite_clear_history)()


def delete_entry(entry_id: int):
    (_supabase_delete_entry if USE_SUPABASE else _sqlite_delete_entry)(entry_id)