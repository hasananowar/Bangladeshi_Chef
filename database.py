import sqlite3
from datetime import datetime

DB_PATH = "./chat_history.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def create_session() -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (created_at) VALUES (?)",
            (datetime.utcnow().isoformat(),)
        )
        conn.commit()
        return cursor.lastrowid


def save_message(session_id: int, role: str, content: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat())
        )
        conn.commit()


def load_messages(session_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def load_all_sessions() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY id DESC"
        ).fetchall()
        return [{"id": row["id"], "created_at": row["created_at"]} for row in rows]
