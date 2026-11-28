"""NoMem for LangGraph. ``pip install "nomem-ai[langgraph]"``.

Two pieces, for the two kinds of memory LangGraph has:

* :class:`NoMemStore` — a ``BaseStore`` for long-term memory
  (``graph.compile(store=...)``, ``create_react_agent(..., store=...)``).
  ``put`` / ``get`` / ``search`` / ``delete`` / ``list_namespaces`` work as
  with ``InMemoryStore``, and ``search(query=...)`` uses NoMem's hybrid
  retrieval, no embedding model required.
* :class:`NoMemGraphMemory` — conversation memory for a graph: a ``record``
  node that stores new turns, and a ``prompt`` that adds dated memories and
  known facts to the model input. Its ``tools()`` work with ``ToolNode``.

Short-term state (the thread's messages) stays with LangGraph's checkpointer;
NoMem adds what the checkpointer does not have: earlier threads, facts and
conflicts, within a token budget.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Union

from ..core import NoMem
from ..service import MemoryService
from ._toolkit import Backend, Memory, content_text, memory_text

try:
    from langchain_core.messages import BaseMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.base import (BaseStore, GetOp, Item, ListNamespacesOp, PutOp,
                                      SearchItem, SearchOp)
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('nomem.integrations.langgraph needs langgraph: '
                      'pip install "nomem-ai[langgraph]"') from e

from .langchain import memory_tools, message_role

_NS_PREFIX = "lg:"
_KEY = "lg_key"


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _value_text(value: dict, fields: Optional[list[str]]) -> str:
    """The text NoMem indexes for a stored value."""
    if fields:
        parts = []
        for f in fields:
            v = value
            for part in f.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if v is not None:
                parts.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        if parts:
            return "\n".join(parts)
    for f in ("content", "text", "memory", "data"):
        if isinstance(value.get(f), str) and value[f].strip():
            return value[f]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


_OPS = {
    "$eq": lambda a, b: a == b, "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b, "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b, "$lte": lambda a, b: a is not None and a <= b,
}


def _matches(value: dict, flt: Optional[dict]) -> bool:
    for k, want in (flt or {}).items():
        have = value.get(k)
        if isinstance(want, dict) and want and all(op in _OPS for op in want):
            try:
                if not all(_OPS[op](have, v) for op, v in want.items()):
                    return False
            except TypeError:
                return False
        elif have != want:
            return False
    return True


def _ns_match(ns: tuple, path: tuple, suffix: bool) -> bool:
    if len(path) > len(ns):
        return False
    part = ns[-len(path):] if suffix else ns[:len(path)]
    return all(p == "*" or p == n for p, n in zip(path, part)) if path else True


class NoMemStore(BaseStore):
    """LangGraph long-term memory store on NoMem.

    Each namespace is its own NoMem user (so facts and conflicts never mix
    across namespaces); each item is one stored message whose
    text is the value's ``content`` (or ``text`` / ``memory`` / ``data``, or
    the fields named in ``index``, or the JSON of the value) and whose
    metadata keeps the full value. ``put`` on an existing key replaces it.

    Search ranks with NoMem's keyword, entity, time and slot rankers (plus
    vectors when the memory has an embedder). Scores are fusion scores: only
    their order is meaningful.

    Args:
        memory: A :class:`~nomem.MemoryService` or a SQLite path.
        user_id: Prefix for the NoMem user ids the namespaces map to, so
            several stores can share one database.
        **service_kwargs: Passed to ``MemoryService`` when ``memory`` is a path.
    """

    supports_ttl = False

    def __init__(self, memory: Union[MemoryService, str] = ":memory:", *,
                 user_id: str = "default", **service_kwargs):
        self.service = memory if isinstance(memory, MemoryService) else MemoryService(
            memory, **service_kwargs)
        self.user_id = user_id

    # -- helpers ------------------------------------------------------------

    def _uid(self, namespace: tuple) -> str:
        if not namespace or not all(isinstance(p, str) and p for p in namespace):
            raise ValueError(f"namespace must be a non-empty tuple of non-empty strings, got {namespace!r}")
        return self._prefix + json.dumps(list(namespace), ensure_ascii=False)

    @property
    def _prefix(self) -> str:
        return f"{_NS_PREFIX}{self.user_id}:"

    def _backend(self, namespace: tuple) -> Backend:
        return Backend(self.service, self._uid(namespace), "default")

    def _namespaces(self) -> list[tuple]:
        out = []
        for uid in self.service.store.user_ids(self._prefix):
            try:
                out.append(tuple(json.loads(uid[len(self._prefix):])))
            except ValueError:
                continue
        return out

    def _rows(self, namespace: tuple, key: Optional[str] = None):
        return self.service.store.messages_with_meta(self._uid(namespace), "default", _KEY, key)

    @staticmethod
    def _item(namespace: tuple, msg, score: Optional[float] = None, search: bool = False):
        meta = msg.metadata or {}
        created = _dt(meta.get("lg_created", msg.created_at))
        updated = _dt(msg.created_at)
        value = meta.get("lg_value") or {}
        if search:
            return SearchItem(namespace, meta[_KEY], value, created, updated, score)
        return Item(value=value, key=meta[_KEY], namespace=namespace,
                    created_at=created, updated_at=updated)

    # -- operations ---------------------------------------------------------

    def _get(self, op: GetOp):
        rows = self._rows(op.namespace, op.key)
        return self._item(op.namespace, rows[-1]) if rows else None

    def _put(self, op: PutOp) -> None:
        if not isinstance(op.key, str) or not op.key:
            raise ValueError("key must be a non-empty string")
        old = self._rows(op.namespace, op.key)
        backend = self._backend(op.namespace)
        created = old[0].metadata.get("lg_created", old[0].created_at) if old else time.time()
        with backend.lock:
            if old:
                backend.forget([m.id for m in old])
            if op.value is None:
                return
            if not isinstance(op.value, dict):
                raise ValueError("value must be a dict")
            fields = op.index if isinstance(op.index, list) else None
            backend.add(_value_text(op.value, fields), "user",
                        metadata={_KEY: op.key, "lg_value": op.value, "lg_created": created})

    def _search(self, op: SearchOp) -> list:
        prefix = tuple(op.namespace_prefix)
        spaces = [ns for ns in self._namespaces() if ns[:len(prefix)] == prefix]
        want = max(0, op.offset) + max(0, op.limit)
        found = []
        for ns in spaces:
            if op.query:
                hits = self._backend(ns).search(op.query, max(want * 3, 25))
                ids = [h["id"] for h in hits if _KEY in (h.get("metadata") or {})]
                msgs = self.service.store.get_messages(ids)
                score = {h["id"]: h["score"] for h in hits}
                for mid in ids:
                    m = msgs.get(mid)
                    if m and _matches(m.metadata.get("lg_value") or {}, op.filter):
                        found.append((score[mid], m.created_at, ns, m))
            else:
                for m in self._rows(ns):
                    if _matches(m.metadata.get("lg_value") or {}, op.filter):
                        found.append((None, m.created_at, ns, m))
        if op.query:
            found.sort(key=lambda t: (-t[0], -t[1]))
        else:
            found.sort(key=lambda t: -t[1])
        return [self._item(ns, m, s, search=True) for s, _, ns, m in found[op.offset:want]]

    def _list(self, op: ListNamespacesOp) -> list[tuple]:
        spaces = self._namespaces()
        for cond in op.match_conditions or ():
            spaces = [ns for ns in spaces
                      if _ns_match(ns, tuple(cond.path), cond.match_type == "suffix")]
        if op.max_depth is not None:
            spaces = [ns[:op.max_depth] for ns in spaces]
        spaces = sorted(set(spaces))
        return spaces[op.offset:op.offset + op.limit]

    def batch(self, ops: Iterable) -> list:
        out = []
        with self.service._lock:
            for op in ops:
                if isinstance(op, GetOp):
                    out.append(self._get(op))
                elif isinstance(op, PutOp):
                    out.append(self._put(op))
                elif isinstance(op, SearchOp):
                    out.append(self._search(op))
                elif isinstance(op, ListNamespacesOp):
                    out.append(self._list(op))
                else:
                    raise ValueError(f"unknown store operation: {op!r}")
        return out

    async def abatch(self, ops: Iterable) -> list:
        return await asyncio.to_thread(self.batch, list(ops))


class NoMemGraphMemory:
    """Conversation memory for a LangGraph graph.

    Add :meth:`record` as a node (it stores the turns of ``state["messages"]``
    that are not stored yet, so it is safe to run every step), and use
    :meth:`prompt` to build the model input, e.g.
    ``create_react_agent(model, tools, prompt=mem.prompt)``.

    With a :class:`~nomem.MemoryService`, the memory is chosen per run from
    ``config["configurable"]``: ``user_id`` picks the user, and the session is
    ``session_id`` if given, else one shared session per user (so memories
    carry across threads). Set ``per_thread=True`` to key it on ``thread_id``.

    Args:
        memory: A :class:`~nomem.NoMem` or :class:`~nomem.MemoryService`.
        system: Your own instructions, put first in the system message.
        max_tokens: Budget for the memory part of the prompt.
        messages_key: The state key holding the messages.
    """

    def __init__(self, memory: Memory, *, system: str = "", max_tokens: Optional[int] = None,
                 messages_key: str = "messages", per_thread: bool = False,
                 user_id: Optional[str] = None, session_id: Optional[str] = None):
        self.memory = memory
        self.system = system
        self.max_tokens = max_tokens
        self.messages_key = messages_key
        self.per_thread = per_thread
        self.user_id = user_id
        self.session_id = session_id

    def backend(self, config: Optional[RunnableConfig] = None) -> Backend:
        if isinstance(self.memory, NoMem):
            return Backend(self.memory)
        conf = (config or {}).get("configurable", {}) or {}
        user = conf.get("user_id") or self.user_id or "default"
        session = (conf.get("session_id") or self.session_id
                   or (conf.get("thread_id") if self.per_thread else None) or "default")
        return Backend(self.memory, str(user), str(session))

    def _messages(self, state: Any) -> list:
        if isinstance(state, dict):
            return list(state.get(self.messages_key) or [])
        if isinstance(state, list):
            return state
        return list(getattr(state, self.messages_key, []) or [])

    # -- writing ------------------------------------------------------------

    def record(self, state: Any, config: Optional[RunnableConfig] = None) -> dict:
        """Graph node: store human and AI text turns not stored yet. Returns ``{}``."""
        backend = self.backend(config)
        store = backend.nomem.store
        for m in self._messages(state):
            role = message_role(m)
            text = content_text(m.content) if role else ""
            if not role or not text.strip():
                continue
            mid = getattr(m, "id", None)
            if mid and store.messages_with_meta(backend.user_id, backend.session_id, "lc_id", str(mid)):
                continue
            backend.add(text, role, metadata={"lc_id": str(mid)} if mid else None)
        return {}

    async def arecord(self, state: Any, config: Optional[RunnableConfig] = None) -> dict:
        return await asyncio.to_thread(self.record, state, config)

    # -- reading ------------------------------------------------------------

    def context(self, query: Optional[str], config: Optional[RunnableConfig] = None,
                exclude: Iterable[str] = ()) -> str:
        """Memory text for ``query``: facts, retrieved memories and earlier turns.

        Turns whose text is in ``exclude`` (those already in the thread) are
        left out, so the model does not see them twice.
        """
        res = self.backend(config).context(query, self.max_tokens)
        return memory_text(res.messages, exclude)

    def prompt(self, state: Any, config: Optional[RunnableConfig] = None) -> list[BaseMessage]:
        """Model input: one system message with memory, then the thread's messages."""
        msgs = self._messages(state)
        texts = [content_text(m.content) for m in msgs if message_role(m)]
        query = next((content_text(m.content) for m in reversed(msgs)
                      if message_role(m) == "user"), None)
        memory = self.context(query, config, exclude=texts)
        system = "\n\n".join(p for p in (self.system.strip(), memory) if p)
        rest = [m for m in msgs if getattr(m, "type", "") != "system"]
        return ([SystemMessage(system)] if system else []) + rest

    async def aprompt(self, state: Any, config: Optional[RunnableConfig] = None) -> list[BaseMessage]:
        return await asyncio.to_thread(self.prompt, state, config)

    def tools(self, config: Optional[RunnableConfig] = None, **kwargs) -> list:
        """NoMem tools (LangChain ``StructuredTool``) for ``ToolNode`` or ``bind_tools``."""
        return memory_tools(self.backend(config), **kwargs)
