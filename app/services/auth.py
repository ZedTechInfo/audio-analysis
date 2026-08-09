from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _check_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt_hex, digest_hex = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


class AuthStore:
    def __init__(self, database: Path):
        self.database = database
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )""")

    def _connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def create_user(self, email: str, username: str, password: str) -> int:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users (email, username, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (email, username, _hash_password(password), now, now),
            )
            return int(cursor.lastrowid)

    def authenticate(self, identifier: str, password: str) -> dict | None:
        identifier = identifier.strip().lower()
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            user = connection.execute(
                "SELECT * FROM users WHERE (email = ? OR username = ?) AND is_active = 1",
                (identifier, identifier),
            ).fetchone()
        return dict(user) if user and _check_password(password, user["password_hash"]) else None

    def create_session(self, user_id: int, days: int = 7) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (_hash_token(token), user_id, now.isoformat(), (now + timedelta(days=days)).isoformat()),
            )
        return token

    def get_session_user(self, token: str) -> dict | None:
        if not token:
            return None
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token_hash = ? AND users.is_active = 1 AND sessions.expires_at > ?",
                (_hash_token(token), _now()),
            ).fetchone()
            return dict(row) if row else None

    def delete_session(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))

    def create_reset_token(self, user_id: int, minutes: int = 30) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL", (_now(), user_id))
            connection.execute(
                "INSERT INTO password_reset_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (_hash_token(token), user_id, now.isoformat(), (now + timedelta(minutes=minutes)).isoformat()),
            )
        return token

    def reset_password(self, token: str, password: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id FROM password_reset_tokens WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?",
                (_hash_token(token), _now()),
            ).fetchone()
            if not row:
                return False
            user_id = row[0]
            now = _now()
            connection.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", (_hash_password(password), now, user_id))
            connection.execute("UPDATE password_reset_tokens SET used_at = ? WHERE token_hash = ?", (now, _hash_token(token)))
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return True
