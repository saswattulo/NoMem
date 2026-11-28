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

CREATE TABLE IF NOT EXISTS annotations (
    message_id INTEGER PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0,
    norm_hash  TEXT,
    simhash    INTEGER,
    keyphrases TEXT
);
CREATE INDEX IF NOT EXISTS idx_ann_hash ON annotations(user_id, session_id, norm_hash);

CREATE TABLE IF NOT EXISTS entities (
    message_id INTEGER NOT NULL,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    entity     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities ON entities(user_id, session_id, entity);

CREATE TABLE IF NOT EXISTS times (
    message_id INTEGER NOT NULL,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    t_start    REAL NOT NULL,
    t_end      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_times ON times(user_id, session_id, t_start);

CREATE TABLE IF NOT EXISTS slots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT NOT NULL,
    subject           TEXT NOT NULL,
    attribute         TEXT NOT NULL,
    value             TEXT NOT NULL,
    multi             INTEGER NOT NULL DEFAULT 0,
    valid_from        REAL NOT NULL,
    valid_to          REAL,
    source_message_id INTEGER,
    closed_by         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_slots ON slots(user_id, subject, attribute, valid_to);

CREATE TABLE IF NOT EXISTS vectors (
    message_id INTEGER PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    vec        BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors ON vectors(user_id, session_id);

CREATE TABLE IF NOT EXISTS frames (
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    frame_hash TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    speaker    TEXT NOT NULL,
    sentence   TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames ON frames(user_id, frame_hash, speaker);

CREATE TABLE IF NOT EXISTS conflicts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    older_id   INTEGER NOT NULL,
    newer_id   INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    attribute  TEXT,
    detail     TEXT NOT NULL,
    confidence REAL NOT NULL,
    UNIQUE(older_id, newer_id)
);
CREATE INDEX IF NOT EXISTS idx_conflicts_older ON conflicts(user_id, older_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_newer ON conflicts(user_id, newer_id);

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


# A small English stop-word set so keyword retrieval matches on meaningful
# terms, not filler like "what" or "my". Kept intentionally short.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "get", "give", "had", "has", "have", "he", "her", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them", "then",
    "there", "they", "this", "to", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def _fts_query(text: str, max_terms: int = 64) -> Optional[str]:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Stop words are dropped so retrieval keys on meaningful terms; if a query is
    *only* stop words we fall back to using them so it still returns something.
    Repeats are dropped, and a very long query (a pasted document) keeps only
    its ``max_terms`` longest words, so one read can't turn into a scan of
    thousands of OR terms.
    """
    terms = list(dict.fromkeys(re.findall(r"\w+", text.lower())))
    meaningful = [t for t in terms if t not in _STOPWORDS]
    terms = meaningful or terms
    if not terms:
        return None
    if max_terms and len(terms) > max_terms:
        keep = set(sorted(terms, key=lambda t: (-len(t), t))[:max_terms])
        terms = [t for t in terms if t in keep]
    return " OR ".join(f'"{t}"' for t in terms)


def _chunks(ids: list[int], n: int = 500):
    ids = list(ids)
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


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
        created_at: Optional[float] = None,
        commit: bool = True,
    ) -> Message:
        created_at = time.time() if created_at is None else created_at
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
        if commit:
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

    def fts_ranked(
        self, user_id: str, session_id: str, query: str, k: int
    ) -> list[int]:
        """Message ids by BM25 (best first)."""
        if not self.fts_enabled:
            return []
        match = _fts_query(query)
        if match is None:
            return []
        rows = self.conn.execute(
            "SELECT f.message_id AS id FROM messages_fts f "
            "WHERE messages_fts MATCH ? AND f.user_id = ? AND f.session_id = ? "
            "ORDER BY bm25(messages_fts) LIMIT ?",
            (match, user_id, session_id, k),
        )
        return [int(r["id"]) for r in rows]

    def fts_scored(
        self, user_id: str, session_id: str, query: str, k: int, max_terms: int = 64
    ) -> list[tuple[int, float]]:
        """``(message_id, bm25)`` best first; higher is better (FTS5's sign flipped)."""
        if not self.fts_enabled or k <= 0:
            return []
        match = _fts_query(query, max_terms)
        if match is None:
            return []
        rows = self.conn.execute(
            "SELECT f.message_id AS id, bm25(messages_fts) AS s FROM messages_fts f "
            "WHERE messages_fts MATCH ? AND f.user_id = ? AND f.session_id = ? "
            "ORDER BY s LIMIT ?",
            (match, user_id, session_id, k),
        )
        return [(int(r["id"]), -float(r["s"])) for r in rows]

    def in_session(self, user_id: str, session_id: str, ids) -> set[int]:
        """The subset of ``ids`` that are messages of this user's session."""
        out: set[int] = set()
        for chunk in _chunks(list(dict.fromkeys(ids))):
            q = (f"SELECT id FROM messages WHERE user_id = ? AND session_id = ? "
                 f"AND id IN ({','.join('?' * len(chunk))})")
            out.update(int(r["id"]) for r in self.conn.execute(q, (user_id, session_id, *chunk)))
        return out

    def get_messages(self, ids: list[int]) -> dict[int, Message]:
        out: dict[int, Message] = {}
        for chunk in _chunks(ids):
            q = f"SELECT * FROM messages WHERE id IN ({','.join('?' * len(chunk))})"
            for r in self.conn.execute(q, chunk):
                out[r["id"]] = self._row_to_message(r)
        return out

    # -- annotations ------------------------------------------------------

    def add_annotation(self, msg: Message, user_id: str, session_id: str, ann, commit: bool = True) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO annotations (message_id, user_id, session_id, importance, norm_hash, simhash, keyphrases)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg.id, user_id, session_id, ann.importance, ann.norm_hash, ann.simhash, json.dumps(ann.keyphrases)),
        )
        self.conn.executemany(
            "INSERT INTO entities (message_id, user_id, session_id, entity) VALUES (?, ?, ?, ?)",
            [(msg.id, user_id, session_id, e) for e in ann.entities],
        )
        self.conn.executemany(
            "INSERT INTO times (message_id, user_id, session_id, t_start, t_end) VALUES (?, ?, ?, ?, ?)",
            [(msg.id, user_id, session_id, a, b) for a, b in ann.times],
        )
        if commit:
            self.conn.commit()

    def annotations(self, ids: list[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for chunk in _chunks(ids):
            q = f"SELECT * FROM annotations WHERE message_id IN ({','.join('?' * len(chunk))})"
            for r in self.conn.execute(q, chunk):
                out[r["message_id"]] = dict(r)
        return out

    def find_by_hash(self, user_id: str, session_id: str, norm_hash: str, before_id: int) -> Optional[int]:
        row = self.conn.execute(
            "SELECT message_id FROM annotations WHERE user_id = ? AND session_id = ? AND norm_hash = ?"
            " AND message_id < ? ORDER BY message_id LIMIT 1",
            (user_id, session_id, norm_hash, before_id),
        ).fetchone()
        return int(row["message_id"]) if row else None

    def entity_ranked(self, user_id: str, session_id: str, ents: list[str], k: int,
                      max_df: Optional[int] = None) -> list[int]:
        """Messages mentioning the most query entities (rarest entities weigh most).

        Entities found in more than ``max_df`` messages (e.g. the user's own
        name in a transcript) say nothing about relevance and are ignored.
        """
        if not ents:
            return []
        having = f" HAVING COUNT(*) <= {int(max_df)}" if max_df else ""
        q = (
            "SELECT e.message_id AS id, SUM(1.0 / df.n) AS score FROM entities e "
            "JOIN (SELECT entity, COUNT(*) AS n FROM entities WHERE user_id = ? AND session_id = ? "
            f"AND entity IN ({','.join('?' * len(ents))}) GROUP BY entity{having}) df ON df.entity = e.entity "
            "WHERE e.user_id = ? AND e.session_id = ? "
            "GROUP BY e.message_id ORDER BY score DESC, e.message_id DESC LIMIT ?"
        )
        rows = self.conn.execute(q, (user_id, session_id, *ents, user_id, session_id, k))
        return [int(r["id"]) for r in rows]

    def time_ranked(self, user_id: str, session_id: str, ranges: list[tuple[float, float]], k: int) -> list[int]:
        """Messages sent in, or talking about, any of the given time ranges."""
        ids: list[int] = []
        for a, b in ranges:
            rows = self.conn.execute(
                "SELECT id FROM (SELECT id FROM messages WHERE user_id = ? AND session_id = ? "
                "AND created_at BETWEEN ? AND ? UNION SELECT message_id AS id FROM times "
                "WHERE user_id = ? AND session_id = ? AND t_start <= ? AND t_end >= ?) "
                "ORDER BY id DESC LIMIT ?",
                (user_id, session_id, a, b, user_id, session_id, b, a, k),
            )
            ids += [int(r["id"]) for r in rows]
        seen: set[int] = set()
        return [i for i in ids if not (i in seen or seen.add(i))][:k]

    def neighbors(self, user_id: str, session_id: str, ids: list[int], span: int = 1) -> dict[int, list[int]]:
        """For each id, the ids of the ``span`` turns before and after it."""
        out: dict[int, list[int]] = {}
        for i in ids:
            rows = self.conn.execute(
                "SELECT id FROM (SELECT id FROM messages WHERE user_id = ? AND session_id = ? AND id < ? "
                "ORDER BY id DESC LIMIT ?) UNION SELECT id FROM (SELECT id FROM messages WHERE "
                "user_id = ? AND session_id = ? AND id > ? ORDER BY id LIMIT ?)",
                (user_id, session_id, i, span, user_id, session_id, i, span),
            )
            out[i] = [int(r["id"]) for r in rows]
        return out

    # -- vectors -----------------------------------------------------------

    def add_vector(self, msg_id: int, user_id: str, session_id: str, vec: bytes, commit: bool = True) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors (message_id, user_id, session_id, vec) VALUES (?, ?, ?, ?)",
            (msg_id, user_id, session_id, vec),
        )
        if commit:
            self.conn.commit()

    def vectors(self, user_id: str, session_id: str) -> list[tuple[int, bytes]]:
        rows = self.conn.execute(
            "SELECT message_id, vec FROM vectors WHERE user_id = ? AND session_id = ? ORDER BY message_id",
            (user_id, session_id),
        )
        return [(int(r[0]), r[1]) for r in rows]

    # -- slots (structured facts with validity windows) --------------------

    def current_slots(self, user_id: str, subject: Optional[str] = None,
                      attribute: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM slots WHERE user_id = ? AND valid_to IS NULL"
        params: list = [user_id]
        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject)
        if attribute is not None:
            sql += " AND attribute = ?"
            params.append(attribute)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY id", params)]

    def slot_history(self, user_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM slots WHERE user_id = ? ORDER BY id", (user_id,))]

    def add_slot(self, user_id: str, subject: str, attribute: str, value: str, multi: bool,
                 valid_from: float, source_message_id: Optional[int], commit: bool = True) -> int:
        cur = self.conn.execute(
            "INSERT INTO slots (user_id, subject, attribute, value, multi, valid_from, source_message_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, subject, attribute, value, int(multi), valid_from, source_message_id),
        )
        if commit:
            self.conn.commit()
        return int(cur.lastrowid)

    def close_slot(self, slot_id: int, valid_to: float, closed_by: Optional[int], commit: bool = True) -> None:
        self.conn.execute("UPDATE slots SET valid_to = ?, closed_by = ? WHERE id = ?",
                          (valid_to, closed_by, slot_id))
        if commit:
            self.conn.commit()

    def slots_from_messages(self, user_id: str, ids: list[int]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for chunk in _chunks(ids):
            q = (f"SELECT * FROM slots WHERE user_id = ? AND source_message_id IN "
                 f"({','.join('?' * len(chunk))})")
            for r in self.conn.execute(q, (user_id, *chunk)):
                out.setdefault(r["source_message_id"], []).append(dict(r))
        return out

    # -- conflicts (contradicting memories, both kept) ----------------------

    def add_frame(self, user_id: str, session_id: str, frame_hash: str, message_id: int,
                  speaker: str, sentence: str, created_at: float) -> None:
        self.conn.execute(
            "INSERT INTO frames (user_id, session_id, frame_hash, message_id, speaker, sentence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, session_id, frame_hash, message_id, speaker, sentence, created_at),
        )

    def frames_like(self, user_id: str, frame_hash: str, speaker: str, before_id: int,
                    limit: int = 5) -> list[dict]:
        """Earlier sentences with the same frame and speaker, latest first."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM frames WHERE user_id = ? AND frame_hash = ? AND speaker = ? AND message_id < ?"
            " ORDER BY created_at DESC, message_id DESC LIMIT ?",
            (user_id, frame_hash, speaker, before_id, limit))]

    def add_conflict(self, user_id: str, older_id: int, newer_id: int, kind: str, detail: str,
                     confidence: float, attribute: Optional[str] = None) -> None:
        """Link two contradicting memories. The stronger evidence wins a repeat."""
        self.conn.execute(
            "INSERT INTO conflicts (user_id, older_id, newer_id, kind, attribute, detail, confidence)"
            " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(older_id, newer_id) DO UPDATE SET"
            " kind = excluded.kind, attribute = excluded.attribute, detail = excluded.detail,"
            " confidence = excluded.confidence WHERE excluded.confidence > conflicts.confidence",
            (user_id, older_id, newer_id, kind, attribute, detail, confidence),
        )

    def conflicts(self, user_id: str, ids: Optional[list[int]] = None) -> list[dict]:
        """Conflicts touching any of ``ids`` (all of the user's when ``None``)."""
        if ids is None:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM conflicts WHERE user_id = ? ORDER BY id", (user_id,))]
        out: dict[int, dict] = {}
        for chunk in _chunks(ids):
            ph = ",".join("?" * len(chunk))
            for r in self.conn.execute(
                f"SELECT * FROM conflicts WHERE user_id = ? AND (older_id IN ({ph}) OR newer_id IN ({ph}))",
                (user_id, *chunk, *chunk)):
                out[r["id"]] = dict(r)
        return [out[k] for k in sorted(out)]

    def commit(self) -> None:
        self.conn.commit()

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
