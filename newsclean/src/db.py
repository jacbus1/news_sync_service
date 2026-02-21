import sqlite3
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone


def init_db(sqlite_path: str, schema_path: str) -> None:
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            conn.executescript(f.read())
        # Lightweight migrations for existing DBs (SQLite doesn't support ALTER TABLE ... IF NOT EXISTS).
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_digests)").fetchall()}
            if "draft_chat_id" not in cols:
                conn.execute("ALTER TABLE daily_digests ADD COLUMN draft_chat_id TEXT")
            if "draft_message_id" not in cols:
                conn.execute("ALTER TABLE daily_digests ADD COLUMN draft_message_id INTEGER")
        except Exception:
            pass
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS bot_state (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()


def insert_events(sqlite_path: str, rows: List[Dict]) -> int:
    if not rows:
        return 0
    conn = sqlite3.connect(sqlite_path)
    inserted = 0
    try:
        for r in rows:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO events_raw (source,title,url,published_at,ingested_at,content) VALUES (?,?,?,?,?,?)",
                    (r.get('source'), r.get('title'), r.get('url'), r.get('published_at'), r.get('ingested_at'), r.get('content'))
                )
                inserted += conn.total_changes
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()
    return inserted


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_sent_keys(sqlite_path: str, keys: List[str]) -> set:
    if not keys:
        return set()
    conn = sqlite3.connect(sqlite_path)
    try:
        qmarks = ",".join(["?"] * len(keys))
        cur = conn.execute(f"SELECT key FROM sent_items WHERE key IN ({qmarks})", tuple(keys))
        return {r[0] for r in cur.fetchall() if r and r[0]}
    finally:
        conn.close()


def mark_sent(sqlite_path: str, rows: List[Dict]) -> None:
    """
    rows: [{key,url,title}]
    """
    if not rows:
        return
    now = _utc_now_iso()
    conn = sqlite3.connect(sqlite_path)
    try:
        for r in rows:
            k = r.get("key")
            if not k:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO sent_items (key,url,title,first_seen_at,sent_at) VALUES (?,?,?,?,?)",
                (k, r.get("url"), r.get("title"), now, now),
            )
        conn.commit()
    finally:
        conn.close()


def insert_summary(sqlite_path: str, created_at: str, item_count: int, summary_text: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "INSERT INTO summaries (created_at, item_count, summary_text) VALUES (?,?,?)",
            (created_at, item_count, summary_text)
        )
        conn.commit()
    finally:
        conn.close()


def get_last_summary(sqlite_path: str):
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.execute(
            "SELECT created_at, summary_text FROM summaries ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row
    finally:
        conn.close()


def insert_daily_report(sqlite_path: str, report_date: str, fgi_value: int, report_text: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "INSERT INTO daily_reports (report_date, fgi_value, report_text) VALUES (?,?,?)",
            (report_date, fgi_value, report_text)
        )
        conn.commit()
    finally:
        conn.close()


def get_last_daily_report_date(sqlite_path: str):
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.execute(
            "SELECT report_date FROM daily_reports ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def upsert_daily_digest_draft(sqlite_path: str, digest_date_et: str, created_at: str, draft_text: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "INSERT INTO daily_digests (digest_date_et, status, created_at, draft_text, draft_chat_id, draft_message_id, sent_at) "
            "VALUES (?,?,?,?,NULL,NULL,NULL) "
            "ON CONFLICT(digest_date_et) DO UPDATE SET draft_text=excluded.draft_text, status='draft'",
            (digest_date_et, "draft", created_at, draft_text),
        )
        conn.commit()
    finally:
        conn.close()


def get_daily_digest(sqlite_path: str, digest_date_et: str):
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.execute(
            "SELECT digest_date_et, status, created_at, draft_text, sent_at FROM daily_digests WHERE digest_date_et = ?",
            (digest_date_et,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def mark_daily_digest_sent(sqlite_path: str, digest_date_et: str, sent_at: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "UPDATE daily_digests SET status='sent', sent_at=? WHERE digest_date_et=?",
            (sent_at, digest_date_et),
        )
        conn.commit()
    finally:
        conn.close()


def set_daily_digest_draft_message(sqlite_path: str, digest_date_et: str, chat_id: str, message_id: int) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "UPDATE daily_digests SET draft_chat_id=?, draft_message_id=? WHERE digest_date_et=?",
            (str(chat_id), int(message_id), digest_date_et),
        )
        conn.commit()
    finally:
        conn.close()


def state_get(sqlite_path: str, k: str):
    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute("SELECT v FROM bot_state WHERE k=?", (k,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def state_set(sqlite_path: str, k: str, v: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            "INSERT INTO bot_state (k,v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (k, str(v)),
        )
        conn.commit()
    finally:
        conn.close()
