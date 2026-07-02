"""SQLite persistence for the three memory tiers.

Local-first by design: everything lives in one SQLite file (or ``:memory:``),
with no external services and no extra dependencies. Full-text retrieval uses
SQLite's built-in FTS5, so the archival tier works with zero embedding models.

Scoping is ``user_id`` + ``session_id``:
- messages / summaries are session-scoped (a conversation),
- facts are user-scoped (durable, span sessions).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Optional

from .models import Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    tokens     INTEGER,
    created_at REAL NOT NULL,
    metadata   TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_scope ON messages(user_id, session_id, id);

CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,
    covers_from INTEGER,
    covers_to   INTEGER,
    tokens      INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_scope ON summaries(user_id, session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(user_id, key)
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    message_id UNINDEXED,
    user_id    UNINDEXED,
    session_id UNINDEXED,
    tokenize = 'porter'
);
"""


def _fts_query(text: str) -> Optional[str]:
    """Turn arbitrary user text into a safe FTS5 MATCH expression."""
    terms = re.findall(r"\w+", text.lower())
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


class SQLiteStore:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.fts_enabled = True
        try:
            self.conn.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            # SQLite build without FTS5 — archival keyword search is disabled,
            # everything else still works.
            self.fts_enabled = False
        self.conn.commit()

    # -- messages (working / archival) -----------------------------------

    def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> Message:
        created_at = time.time()
        cur = self.conn.execute(
            "INSERT INTO messages (user_id, session_id, role, content, tokens, created_at, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, session_id, role, content, tokens, created_at, json.dumps(metadata or {})),
        )
        msg_id = cur.lastrowid
        if self.fts_enabled:
            self.conn.execute(
                "INSERT INTO messages_fts (content, message_id, user_id, session_id) VALUES (?, ?, ?, ?)",
                (content, msg_id, user_id, session_id),
            )
        self.conn.commit()
        return Message(
            role=role,
            content=content,
            id=msg_id,
            created_at=created_at,
            tokens=tokens,
            metadata=metadata or {},
        )

    def recent_messages(
        self, user_id: str, session_id: str, limit: Optional[int] = None
    ) -> list[Message]:
        """Messages newest-first (most recent first), optionally capped."""
        sql = (
            "SELECT * FROM messages WHERE user_id = ? AND session_id = ? ORDER BY id DESC"
        )
        params: tuple = (user_id, session_id)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        return [self._row_to_message(r) for r in self.conn.execute(sql, params)]

    def search(
        self, user_id: str, session_id: str, query: str, k: int = 5
    ) -> list[Message]:
        """Archival retrieval via FTS5 keyword match, best matches first."""
        if not self.fts_enabled:
            return []
        match = _fts_query(query)
        if match is None:
            return []
        rows = self.conn.execute(
            "SELECT m.*, bm25(messages_fts) AS score "
            "FROM messages_fts f JOIN messages m ON m.id = f.message_id "
            "WHERE messages_fts MATCH ? AND m.user_id = ? AND m.session_id = ? "
            "ORDER BY score LIMIT ?",
            (match, user_id, session_id, k),
        )
        return [self._row_to_message(r) for r in rows]

    def count_messages(self, user_id: str, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchone()
        return int(row["n"])

    # -- facts (core) ----------------------------------------------------

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO facts (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (user_id, key, value, time.time()),
        )
        self.conn.commit()

    def get_facts(self, user_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM facts WHERE user_id = ? ORDER BY key", (user_id,)
        )
        return {r["key"]: r["value"] for r in rows}

    # -- helpers ---------------------------------------------------------

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            role=row["role"],
            content=row["content"],
            id=row["id"],
            created_at=row["created_at"],
            tokens=row["tokens"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        self.conn.close()
