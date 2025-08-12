import sqlite3
from datetime import datetime
import os



DATABASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DATABASE_NAME = 'user_history.db'
DB_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Drop and create users table
c.execute('DROP TABLE IF EXISTS users')
conn.commit()

c.execute('''
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    password TEXT NOT NULL,
    email TEXT NOT NULL,
    firm TEXT,
    unit TEXT,
    location TEXT,
    created_at TEXT NOT NULL
)
''')
conn.commit()

# Seed dummy users
dummy_users = [
    {
        "user_id": "harish",
        "name": "Harish",
        "password": "harish@123",
        "email": "harish@email.com",
        "firm": "Independent",
        "unit": "Gen AI",
        "location": "Bangalore"
    },
    {
        "user_id": "melvin",
        "name": "Melvin",
        "password": "Melvin@123",
        "email": "melvin@uk.com",
        "firm": "UK",
        "unit": "Gen AI",
        "location": "UK"
    }
]

now = datetime.utcnow().isoformat()

for user in dummy_users:
    c.execute('''
        INSERT INTO users (user_id, name, password, email, firm, unit, location, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user["user_id"], user["name"], user["password"], user["email"],
          user["firm"], user["unit"], user["location"], now))

conn.commit()
print("✅ Database initialized and dummy users created.")
