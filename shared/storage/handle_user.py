import sqlite3
from datetime import datetime
import os

DATABASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DATABASE_NAME = 'user_history.db'
DB_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# === Core User Functions ===

def create_user(user_id, name, password, email, firm, unit, location):
    now = datetime.utcnow().isoformat()
    c.execute('''
        INSERT OR IGNORE INTO users (
            user_id, name, password, email, firm, unit, location, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, name, password, email, firm, unit, location, now))
    conn.commit()

def get_user(user_id):
    c.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
    return c.fetchone()

def get_all_users():
    c.execute('SELECT user_id, name, created_at FROM users ORDER BY created_at DESC')
    return c.fetchall()

def update_user_name(user_id, new_name):
    c.execute('UPDATE users SET name = ? WHERE user_id = ?', (new_name, user_id))
    conn.commit()

def validate_login(user_id, password):
    c.execute('SELECT * FROM users WHERE user_id = ? AND password = ?', (user_id, password))
    return c.fetchone() is not None
