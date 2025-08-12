# shared/storage/handle_mentor_chat_history.py
from typing import Optional, Tuple, List, Dict, Any
import json
import os
import sqlite3
import datetime

DATABASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DATABASE_NAME = 'mentor_data.db'
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

def _get_db_connection():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db():
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        # Table for chat messages (history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                mentor_topics TEXT,         -- JSON string of mentor's suggested topics
                current_topic TEXT,         -- Current topic being discussed
                completed_topics TEXT,      -- JSON string of completed topics
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, title) -- Ensures a user has unique chat titles
            )
        ''')
        # Table for user general preferences (not tied to a specific chat session)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                learning_goal TEXT,
                skills TEXT,                -- JSON string of skills
                difficulty TEXT,
                role TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


def save_chat(user_id: str, title: str, messages_json: str, mentor_topics: Optional[List[str]] = None, current_topic: Optional[str] = None, completed_topics: Optional[List[str]] = None):
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        mentor_topics_json = json.dumps(mentor_topics) if mentor_topics is not None else None
        completed_topics_json = json.dumps(completed_topics) if completed_topics is not None else None

        cursor.execute('''
            INSERT OR REPLACE INTO chats (user_id, title, messages_json, mentor_topics, current_topic, completed_topics)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, title, messages_json, mentor_topics_json, current_topic, completed_topics_json))
        conn.commit()

def get_chats(user_id: str):
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at FROM chats WHERE user_id = ?", (user_id,))
        return cursor.fetchall()

def get_chat_messages_with_state(user_id: str, chat_title: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Retrieves chat messages and the mentor session state for a given user and chat title.
    Returns a tuple of (messages, state) or None if not found.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT messages_json, mentor_topics, current_topic, completed_topics
            FROM chats
            WHERE user_id = ? AND title = ?
        ''', (user_id, chat_title))
        row = cursor.fetchone()

        if row:
            # Parse the JSON string of messages into a list of dictionaries
            messages = json.loads(row["messages_json"]) if row["messages_json"] else []
            
            # Prepare the state dictionary
            mentor_topics = json.loads(row["mentor_topics"]) if row["mentor_topics"] else []
            completed_topics = json.loads(row["completed_topics"]) if row["completed_topics"] else []
            state = {
                "mentor_topics": mentor_topics,
                "current_topic": row["current_topic"],
                "completed_topics": completed_topics
            }
            
            # Return a tuple of (messages, state)
            return messages, state
            
    # If no row is found, return None
    return None

def save_user_preferences(user_id: str, learning_goal: Optional[str], skills: List[str], difficulty: str, role: str):
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        skills_json = json.dumps(skills)
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences (user_id, learning_goal, skills, difficulty, role, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, learning_goal, skills_json, difficulty, role, datetime.datetime.now().isoformat()))
        conn.commit()

def get_user_preferences(user_id: str) -> Optional[Dict[str, Any]]:
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT learning_goal, skills, difficulty, role FROM user_preferences WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            skills = json.loads(row["skills"]) if row["skills"] else []
            return {
                "learning_goal": row["learning_goal"],
                "skills": skills,
                "difficulty": row["difficulty"],
                "role": row["role"]
            }
        return None

# Call init_db once when the module is imported to ensure tables exist
init_db()