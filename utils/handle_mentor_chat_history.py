from __future__ import annotations
import datetime as _dt, json, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "user_history.db"

def _connect():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with _connect() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                mentor_topics TEXT,
                current_topic TEXT,
                completed_topics TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, title)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                learning_goal TEXT,
                skills TEXT,
                difficulty TEXT,
                role TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

def save_chat(*, user_id: str, title: str, messages_json: str, mentor_topics: List[str], current_topic: Optional[str], completed_topics: List[str]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chats (user_id, title, messages_json, mentor_topics, current_topic, completed_topics, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, title) DO UPDATE SET
              messages_json=excluded.messages_json,
              mentor_topics=excluded.mentor_topics,
              current_topic=excluded.current_topic,
              completed_topics=excluded.completed_topics,
              updated_at=excluded.updated_at
            """,
            (
                user_id,
                title,
                messages_json,
                json.dumps(mentor_topics or []),
                current_topic,
                json.dumps(completed_topics or []),
                _dt.datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()

def get_chats(user_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT title, updated_at FROM chats WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
        return [{"title": r["title"], "updated_at": r["updated_at"]} for r in rows]

def get_chat_messages_with_state(user_id: str, title: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    with _connect() as conn:
        row = conn.execute("SELECT messages_json, mentor_topics, current_topic, completed_topics FROM chats WHERE user_id=? AND title=?", (user_id, title)).fetchone()
        if not row:
            return None
        try:
            messages = json.loads(row["messages_json"] or "[]")
        except Exception:
            messages = []
        try:
            mentor_topics = json.loads(row["mentor_topics"] or "[]")
        except Exception:
            mentor_topics = []
        try:
            completed_topics = json.loads(row["completed_topics"] or "[]")
        except Exception:
            completed_topics = []
        state = {"mentor_topics": mentor_topics, "current_topic": row["current_topic"], "completed_topics": completed_topics}
        return messages, state

def save_user_preferences(user_id: str, learning_goal: Optional[str], skills: List[str], difficulty: str, role: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_preferences (user_id, learning_goal, skills, difficulty, role, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              learning_goal=excluded.learning_goal,
              skills=excluded.skills,
              difficulty=excluded.difficulty,
              role=excluded.role,
              updated_at=excluded.updated_at
            """,
            (user_id, learning_goal, json.dumps(skills or []), difficulty, role, _dt.datetime.utcnow().isoformat()),
        )
        conn.commit()

def get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT learning_goal, skills, difficulty, role FROM user_preferences WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return None
        try:
            skills = json.loads(row["skills"] or "[]")
        except Exception:
            skills = []
        return {"learning_goal": row["learning_goal"], "skills": skills, "difficulty": row["difficulty"], "role": row["role"]}