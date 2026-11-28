"""A shared, thread-safe NoMem for servers.

:class:`MemoryService` keeps one SQLite file and hands out a :class:`NoMem`
per ``(user_id, session_id)``, so the HTTP API and the MCP server can serve
many users and sessions from one process. Every call takes one lock: SQLite
writes are serialised anyway, and each call is milliseconds.

Everything returned is plain JSON-ready data (dicts, lists, strings, numbers).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .core import NoMem
from .store import SQLiteStore
from .tokens import default_counter

VALID_ROLES = ("user", "assistant", "system", "tool")


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _message(m) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "created_at": _iso(m.created_at),
        "tokens": m.tokens,
        "metadata": m.metadata or {},
    }


class MemoryService:
    """Many users and sessions over one SQLite file.

    Args:
        db: SQLite path (or ``":memory:"``). Created if missing.
        max_tokens: Default context budget; each :meth:`context` call may
            pass its own.
        core: Default persona/rules placed first in every context.
        embedder: ``"none"`` (default, model-free), ``"model2vec"``
            (``pip install nomem[embed]``), ``"hashing"``, or an
            :class:`~nomem.embed.Embedder` instance.
        **nomem_kwargs: Passed to every :class:`NoMem` (``retrieval_top_k``,
            ``archival_ratio``, ``facts_ratio`` ...).
    """

    def __init__(self, db: str = "nomem.db", *, max_tokens: int = 3000, core: str = "",
                 embedder: Any = "none", **nomem_kwargs):
        self.store = SQLiteStore(db)
        self.max_tokens = max_tokens
        self.core = core
        self.embedder = _make_embedder(embedder)
        nomem_kwargs.setdefault("counter", default_counter())  # one counter, one warning
        self.kwargs = nomem_kwargs
        self._lock = threading.RLock()
        self._mems: dict[tuple[str, str], NoMem] = {}

    def _mem(self, user_id: str, session_id: str) -> NoMem:
        key = (user_id or "default", session_id or "default")
        mem = self._mems.get(key)
        if mem is None:
            mem = NoMem(self.max_tokens, self.core, user_id=key[0], session_id=key[1],
                        db=self.store, embedder=self.embedder, **self.kwargs)
            self._mems[key] = mem
        return mem

    # -- write -----------------------------------------------------------

    def add(self, content: str, role: str = "user", *, user_id: str = "default",
            session_id: str = "default", timestamp: Optional[float] = None,
            metadata: Optional[dict] = None) -> dict:
        """Store one message verbatim. Returns it with its id and conflicts."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {', '.join(VALID_ROLES)}")
        with self._lock:
            mem = self._mem(user_id, session_id)
            msg = mem.add(role, content, metadata=metadata, timestamp=timestamp)
            conflicts = [c for c in mem.conflicts(msg.id) if c["newer_id"] == msg.id]
            return _message(msg) | {"conflicts": [_conflict(c) for c in conflicts]}

    def add_many(self, messages: list[dict], *, user_id: str = "default",
                 session_id: str = "default") -> list[dict]:
        """Store several ``{"role", "content", "timestamp"?, "metadata"?}`` messages in order."""
        out = []
        for m in messages:
            if not isinstance(m, dict):
                raise ValueError("each message must be an object with role and content")
            out.append(self.add(m.get("content", ""), m.get("role", "user"), user_id=user_id,
                                session_id=session_id, timestamp=m.get("timestamp"),
                                metadata=m.get("metadata")))
        return out

    def remember(self, key: str, value: str, *, user_id: str = "default") -> dict:
        """Pin a durable fact for a user (shown on the Known-facts card)."""
        if not key or not value:
            raise ValueError("key and value are required")
        with self._lock:
            self._mem(user_id, "default").remember(key, value)
            return {"key": key, "value": value}

    # -- read ------------------------------------------------------------

    def context(self, query: Optional[str] = None, *, user_id: str = "default",
                session_id: str = "default", max_tokens: Optional[int] = None,
                explain: bool = False) -> dict:
        """Build a token-budgeted context, optionally with the decision log."""
        with self._lock:
            mem = self._mem(user_id, session_id)
            mem.max_tokens = int(max_tokens or self.max_tokens)
            if mem.max_tokens < 1:
                raise ValueError("max_tokens must be >= 1")
            res = mem.build_context(query=query or None)
            out = {"text": res.text, "messages": res.messages,
                   "tokens_used": res.tokens_used, "max_tokens": res.max_tokens,
                   "usage": res.usage()}
            if explain:
                out["explain"] = res.explain()
                out["log"] = [{"action": e.action, "tier": e.tier, "detail": e.detail,
                               "tokens": e.tokens, "budget_remaining": e.budget_remaining}
                              for e in res.log]
            return out

    def search(self, query: str, *, user_id: str = "default", session_id: str = "default",
               k: int = 5) -> list[dict]:
        """Best ``k`` stored messages for ``query``, with scores and the rankers that found them."""
        if not query:
            raise ValueError("query is required")
        with self._lock:
            hits = self._mem(user_id, session_id).search_hits(query, int(k))
            return [_message(h.message) | {"score": round(h.score, 6), "reasons": h.reasons}
                    for h in hits]

    def facts(self, *, user_id: str = "default", history: bool = False) -> dict:
        """Current slot facts and pinned facts; with ``history``, superseded ones too."""
        with self._lock:
            mem = self._mem(user_id, "default")
            rows = mem.fact_history() if history else mem.facts()
            return {
                "facts": [_slot(r) for r in rows],
                "pinned": self.store.get_facts(mem.user_id),
            }

    def conflicts(self, *, user_id: str = "default") -> list[dict]:
        """Every pair of memories where a newer one may change an older one."""
        with self._lock:
            return [_conflict(c) for c in self._mem(user_id, "default").conflicts()]

    def stats(self, *, user_id: str = "default", session_id: str = "default") -> dict:
        with self._lock:
            return {"user_id": user_id, "session_id": session_id,
                    "messages": self._mem(user_id, session_id).message_count()}

    def close(self) -> None:
        with self._lock:
            self.store.close()


def _slot(r: dict) -> dict:
    return {
        "subject": r.get("subject"),
        "attribute": r.get("attribute"),
        "value": r.get("value"),
        "valid_from": _iso(r.get("valid_from")),
        "valid_to": _iso(r.get("valid_to")),
        "source_message_id": r.get("source_message_id"),
    }


def _conflict(c: dict) -> dict:
    return {
        "older_id": c["older_id"], "newer_id": c["newer_id"],
        "older": c["older"], "newer": c["newer"],
        "older_at": _iso(c["older_at"]), "newer_at": _iso(c["newer_at"]),
        "kind": c.get("kind"), "detail": c.get("detail"),
        "confidence": c.get("confidence"),
    }


def _make_embedder(spec):
    if spec is None or spec == "none":
        return None
    if not isinstance(spec, str):
        return spec
    from .embed import HashingEmbedder, Model2VecEmbedder

    if spec == "model2vec":
        return Model2VecEmbedder()
    if spec == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {spec!r}: use none, model2vec or hashing")
