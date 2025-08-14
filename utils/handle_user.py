from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "user_history.db"

_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
_cur = _conn.cursor()

_cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    password TEXT,
    email TEXT,
    firm TEXT,
    unit TEXT,
    location TEXT,
    created_at TEXT
)
""")
_conn.commit()

def create_user(user_id: str, name: str, password: str, email: str, firm: str, unit: str, location: str) -> None:
    now = datetime.utcnow().isoformat()
    _cur.execute(
        """
        INSERT OR IGNORE INTO users (user_id, name, password, email, firm, unit, location, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, name, password, email, firm, unit, location, now),
    )
    _conn.commit()

def get_user(user_id: str):
    _cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    return _cur.fetchone()

def get_all_users():
    _cur.execute("SELECT user_id, name, created_at FROM users ORDER BY created_at DESC")
    return _cur.fetchall()

def update_user_name(user_id: str, new_name: str) -> None:
    _cur.execute("UPDATE users SET name = ? WHERE user_id = ?", (new_name, user_id))
    _conn.commit()

def validate_login(user_id: str, password: str) -> bool:
    _cur.execute("SELECT 1 FROM users WHERE user_id = ? AND password = ?", (user_id, password))
    return _cur.fetchone() is not None