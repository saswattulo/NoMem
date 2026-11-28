"""Framework-neutral pieces every adapter is built from.

:class:`MemoryToolkit` turns a :class:`~nomem.NoMem` or a
:class:`~nomem.MemoryService` into a small set of tools (name, description,
JSON Schema, Python function). Each framework adapter only translates those
specs into its own tool type, so every framework gets the same tools with the
same names, arguments and behaviour as the MCP server.

Tools never raise into the agent loop: bad arguments or a failing store come
back as ``{"error": "..."}`` so the model can read the problem and retry.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

from ..core import NoMem
from ..service import VALID_ROLES, MemoryService, _conflict, _message, _slot

Memory = Union[NoMem, MemoryService]

TOOL_NAMES = ("remember", "recall", "search_memory", "list_facts", "pin_fact",
              "list_conflicts", "forget")
DEFAULT_TOOLS = ("remember", "recall", "search_memory", "pin_fact")

RECALL_HINT = (
    "Memories are shown with their dates. A [CONFLICT: ...] marker means a later "
    "memory changed an earlier one: prefer the later one unless the user says otherwise."
)


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model sees (name, description, schema) and what runs."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


# -- backends: one interface over NoMem and MemoryService ---------------------


class Backend:
    """Thread-safe access to one user's memory, bound to one session."""

    def __init__(self, memory: Memory, user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        if isinstance(memory, NoMem):
            if user_id not in (None, memory.user_id) or session_id not in (None, memory.session_id):
                raise ValueError("a NoMem is bound to one user_id/session_id; pass a "
                                 "MemoryService to address several")
            self.mem: Optional[NoMem] = memory
            self.service: Optional[MemoryService] = None
            self.user_id, self.session_id = memory.user_id, memory.session_id
            self.lock = threading.RLock()
        elif isinstance(memory, MemoryService):
            self.mem = None
            self.service = memory
            self.user_id, self.session_id = user_id or "default", session_id or "default"
            self.lock = memory._lock
        else:
            raise TypeError(f"expected NoMem or MemoryService, got {type(memory).__name__}")

    def with_session(self, session_id: str) -> "Backend":
        if self.service is None:
            if session_id != self.session_id:
                raise ValueError("a NoMem is bound to one session; pass a MemoryService "
                                 "to use several sessions")
            return self
        return Backend(self.service, self.user_id, session_id)

    @property
    def nomem(self) -> NoMem:
        """The underlying :class:`NoMem` for this user and session."""
        if self.mem is not None:
            return self.mem
        with self.lock:
            return self.service._mem(self.user_id, self.session_id)

    @property
    def default_budget(self) -> int:
        return self.mem.max_tokens if self.mem is not None else self.service.max_tokens

    def add(self, content: str, role: str = "user", *, metadata: Optional[dict] = None,
            timestamp: Optional[float] = None) -> dict:
        if self.service is not None:
            return self.service.add(content, role, user_id=self.user_id, session_id=self.session_id,
                                    metadata=metadata, timestamp=timestamp)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {', '.join(VALID_ROLES)}")
        with self.lock:
            msg = self.mem.add(role, content, metadata=metadata, timestamp=timestamp)
            conflicts = [c for c in self.mem.conflicts(msg.id) if c["newer_id"] == msg.id]
            return _message(msg) | {"conflicts": [_conflict(c) for c in conflicts]}

    def context(self, query: Optional[str] = None, max_tokens: Optional[int] = None):
        """A :class:`~nomem.ContextResult`, with ``max_tokens`` applied for this call only."""
        with self.lock:
            mem = self.nomem
            saved = mem.max_tokens
            mem.max_tokens = int(max_tokens or saved)
            try:
                return mem.build_context(query=query or None)
            finally:
                mem.max_tokens = saved

    def search(self, query: str, k: int = 5) -> list[dict]:
        with self.lock:
            return [_message(h.message) | {"score": round(h.score, 6), "reasons": h.reasons}
                    for h in self.nomem.search_hits(query, int(k))]

    def facts(self, history: bool = False) -> dict:
        with self.lock:
            mem = self.nomem
            rows = mem.fact_history() if history else mem.facts()
            return {"facts": [_slot(r) for r in rows], "pinned": mem.store.get_facts(mem.user_id)}

    def pin(self, key: str, value: str) -> dict:
        with self.lock:
            self.nomem.remember(key, value)
        return {"key": key, "value": value}

    def conflicts(self) -> list[dict]:
        with self.lock:
            return [_conflict(c) for c in self.nomem.conflicts()]

    def forget(self, ids: Iterable[int]) -> int:
        with self.lock:
            return self.nomem.forget([int(i) for i in ids])

    def clear(self) -> int:
        with self.lock:
            return self.nomem.forget_session()

    def recent(self, limit: Optional[int] = None):
        """Stored messages oldest first (the newest ``limit`` when given)."""
        with self.lock:
            mem = self.nomem
            return list(reversed(mem.store.recent_messages(mem.user_id, mem.session_id, limit)))


# -- argument coercion: models send "5" for 5 and "false" for False -----------


def _int(value: Any, name: str, lo: int, hi: int, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number, got {value!r}") from None
    return max(lo, min(hi, n))


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


# -- the toolkit ---------------------------------------------------------------


class MemoryToolkit:
    """The NoMem tools, ready to hand to any agent framework.

    Args:
        memory: A :class:`~nomem.NoMem` (one user and session) or a
            :class:`~nomem.MemoryService` (many; pick with ``user_id`` and
            ``session_id``).
        user_id / session_id: Which memory a ``MemoryService`` serves.
        tools: Tool names to expose, from :data:`TOOL_NAMES`. Defaults to
            ``remember``, ``recall``, ``search_memory`` and ``pin_fact``.
            ``forget`` is opt-in because it deletes data.
        recall_tokens: Default size of a ``recall`` answer, in tokens.
        max_recall_tokens: Largest ``recall`` a model may ask for.
        max_k: Largest ``k`` a model may ask ``search_memory`` for.
        remember_role: Role stored by ``remember`` when the model gives none.
    """

    def __init__(self, memory: Memory, *, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, tools: Optional[Iterable[str]] = None,
                 recall_tokens: int = 1500, max_recall_tokens: int = 8000, max_k: int = 25,
                 remember_role: str = "user"):
        self.backend = memory if isinstance(memory, Backend) else Backend(memory, user_id, session_id)
        names = tuple(tools) if tools is not None else DEFAULT_TOOLS
        unknown = [n for n in names if n not in TOOL_NAMES]
        if unknown:
            raise ValueError(f"unknown tool(s) {unknown}; choose from {list(TOOL_NAMES)}")
        if remember_role not in VALID_ROLES:
            raise ValueError(f"remember_role must be one of {', '.join(VALID_ROLES)}")
        self.recall_tokens = recall_tokens
        self.max_recall_tokens = max_recall_tokens
        self.max_k = max_k
        self.remember_role = remember_role
        all_specs = self._all_specs()
        self.specs: list[ToolSpec] = [all_specs[n] for n in names]
        self._by_name = {s.name: s for s in self.specs}

    # -- the tool functions --------------------------------------------------

    def remember(self, content: str, role: Optional[str] = None) -> dict:
        role = (role or self.remember_role).strip().lower()
        if role not in ("user", "assistant"):
            raise ValueError('role must be "user" or "assistant"')
        out = self.backend.add(_text(content, "content"), role)
        return {"id": out["id"], "stored": True, "conflicts": out["conflicts"]}

    def recall(self, query: str, max_tokens: Optional[int] = None) -> str:
        budget = _int(max_tokens, "max_tokens", 32, self.max_recall_tokens, self.recall_tokens)
        return self.backend.context(_text(query, "query"), budget).text

    def search_memory(self, query: str, k: Optional[int] = None) -> list[dict]:
        return self.backend.search(_text(query, "query"), _int(k, "k", 1, self.max_k, 5))

    def list_facts(self, history: Any = False) -> dict:
        return self.backend.facts(_bool(history))

    def pin_fact(self, key: str, value: str) -> dict:
        return self.backend.pin(_text(key, "key"), _text(value, "value"))

    def list_conflicts(self) -> list[dict]:
        return self.backend.conflicts()

    def forget(self, ids: Any) -> dict:
        if isinstance(ids, (int, str)):
            ids = [ids]
        try:
            clean = [int(i) for i in ids]
        except (TypeError, ValueError):
            raise ValueError("ids must be a list of message ids") from None
        return {"deleted": self.backend.forget(clean)}

    # -- specs -----------------------------------------------------------------

    def _all_specs(self) -> dict[str, ToolSpec]:
        s = lambda props, req: {"type": "object", "properties": props, "required": req,
                                "additionalProperties": False}
        q = {"type": "string", "description": "The question or topic, in plain words."}
        specs = [
            ToolSpec("remember",
                     "Store something in long-term memory verbatim: a fact, preference, plan "
                     "or decision worth keeping. Store it in the user's own words. The reply "
                     "lists any earlier memory this one changes.",
                     s({"content": {"type": "string", "description": "The text to keep, exactly as said."},
                        "role": {"type": "string", "enum": ["user", "assistant"],
                                 "description": "Who said it. Defaults to user."}}, ["content"]),
                     self.remember),
            ToolSpec("recall",
                     "Look up long-term memory before answering anything that may depend on "
                     "the past. Returns dated memories relevant to the query plus known facts, "
                     "within a token budget. " + RECALL_HINT,
                     s({"query": q,
                        "max_tokens": {"type": "integer", "minimum": 32,
                                       "maximum": self.max_recall_tokens,
                                       "description": f"Largest answer, in tokens. Default {self.recall_tokens}."}},
                       ["query"]),
                     self.recall),
            ToolSpec("search_memory",
                     "The k best matching stored messages, with ids, dates, scores and the "
                     "rankers that found each one. Use recall for answering; use this to "
                     "inspect or to find ids.",
                     s({"query": q,
                        "k": {"type": "integer", "minimum": 1, "maximum": self.max_k,
                              "description": "How many results. Default 5."}}, ["query"]),
                     self.search_memory),
            ToolSpec("list_facts",
                     "Facts extracted from what the user said (where they live, their job, "
                     "likes ...) and facts pinned with pin_fact.",
                     s({"history": {"type": "boolean",
                                    "description": "Also include facts that were later replaced."}}, []),
                     self.list_facts),
            ToolSpec("pin_fact",
                     "Pin a durable fact, such as a preferred name or a standing rule. Pinned "
                     "facts appear in every recall. Pinning the same key again replaces it.",
                     s({"key": {"type": "string", "description": "Short name, e.g. preferred_name."},
                        "value": {"type": "string", "description": "The fact."}}, ["key", "value"]),
                     self.pin_fact),
            ToolSpec("list_conflicts",
                     "Pairs of memories where a newer one changes an older one, with both "
                     "texts and dates.",
                     s({}, []), self.list_conflicts),
            ToolSpec("forget",
                     "Permanently delete stored messages by id (ids come from search_memory). "
                     "Only use when the user asks to forget something.",
                     s({"ids": {"type": "array", "items": {"type": "integer"},
                                "description": "Message ids to delete."}}, ["ids"]),
                     self.forget),
        ]
        return {sp.name: sp for sp in specs}

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    # -- calling ---------------------------------------------------------------

    def call(self, name: str, arguments: Union[dict, str, None] = None) -> Any:
        """Run a tool by name. Errors come back as ``{"error": ...}``, never raised."""
        spec = self._by_name.get(name)
        if spec is None:
            return {"error": f"unknown tool {name!r}; available: {', '.join(self.names)}"}
        try:
            args = parse_arguments(arguments)
            allowed = spec.parameters["properties"]
            extra = [k for k in args if k not in allowed]
            if extra:
                raise ValueError(f"unexpected argument(s) {extra}; expected {list(allowed)}")
            missing = [k for k in spec.parameters["required"] if k not in args]
            if missing:
                raise ValueError(f"missing required argument(s) {missing}")
            return spec.fn(**args)
        except Exception as e:  # surfaced to the model, which can correct itself
            return {"error": f"{type(e).__name__}: {e}"}

    async def acall(self, name: str, arguments: Union[dict, str, None] = None) -> Any:
        """:meth:`call` off the event loop (SQLite work runs in a thread)."""
        return await asyncio.to_thread(self.call, name, arguments)

    def call_text(self, name: str, arguments: Union[dict, str, None] = None) -> tuple[str, bool]:
        """:meth:`call`, rendered as text for a tool-result message, plus an is-error flag."""
        out = self.call(name, arguments)
        return to_text(out), is_error(out)


def parse_arguments(arguments: Union[dict, str, None]) -> dict:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"arguments are not valid JSON: {e.msg}") from None
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    return dict(arguments)


def is_error(out: Any) -> bool:
    return isinstance(out, dict) and set(out) == {"error"}


def to_text(out: Any) -> str:
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)


# -- message helpers shared by the history adapters ---------------------------


def content_text(content: Any) -> str:
    """Plain text from a message's content: a string or a list of content blocks.

    Understands Anthropic (``{"type": "text", "text": ...}``), OpenAI
    (``input_text`` / ``output_text`` / ``text`` parts) and LangChain content
    lists. Images, tool calls and other non-text blocks are skipped.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type", "text") in ("text", "input_text", "output_text") and \
                    isinstance(block.get("text"), str):
                parts.append(block["text"])
        else:
            text = getattr(block, "text", None)
            if getattr(block, "type", "text") in ("text", "input_text", "output_text") and \
                    isinstance(text, str):
                parts.append(text)
    return "\n".join(p for p in parts if p)


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate NoMem's system blocks (core, facts, memories) from the turns."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    return system, turns


def alternate(turns: list[dict], first: str = "user") -> list[dict]:
    """Make turns strictly alternate user/assistant, starting with ``first``.

    Consecutive turns of one role are merged (joined by a blank line), roles
    other than user/assistant are dropped, and a leading assistant turn is
    dropped when ``first`` is ``"user"``. This is what the Anthropic Messages
    API requires.
    """
    out: list[dict] = []
    for m in turns:
        role = m.get("role")
        text = content_text(m.get("content"))
        if role not in ("user", "assistant") or not text.strip():
            continue
        if not out and first and role != first:
            continue
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": out[-1]["content"] + "\n\n" + text}
        else:
            out.append({"role": role, "content": text})
    return out


def latest_user_text(backend: Backend) -> Optional[str]:
    """The newest stored user message, used as the recall query when none is given."""
    for m in reversed(backend.recent(20)):
        if m.role == "user" and m.content.strip():
            return m.content
    return None


def memory_only(messages: list[dict]) -> str:
    """Only NoMem's system blocks, for frameworks that keep their own recent turns."""
    return split_system(messages)[0]


_PY_TYPES = {"string": str, "integer": int, "boolean": bool, "number": float}


def args_model(spec: ToolSpec):
    """A pydantic model for a tool's arguments, for frameworks that want one.

    Types are kept loose (``int | str`` and so on are coerced by the toolkit)
    only where the model often sends strings; required fields stay required.
    """
    from typing import List

    from pydantic import Field, create_model

    fields = {}
    required = set(spec.parameters.get("required", []))
    for name, prop in spec.parameters.get("properties", {}).items():
        if prop.get("type") == "array":
            typ: Any = List[_PY_TYPES.get(prop.get("items", {}).get("type"), str)]
        else:
            typ = _PY_TYPES.get(prop.get("type"), str)
        desc = prop.get("description")
        if name in required:
            fields[name] = (typ, Field(..., description=desc))
        else:
            fields[name] = (Optional[typ], Field(None, description=desc))
    model_name = "".join(p.title() for p in spec.name.split("_")) + "Args"
    return create_model(model_name, **fields)


def drop_none(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def memory_text(messages: list[dict], exclude: Iterable[str] = ()) -> str:
    """NoMem's context as one block of text for a system prompt.

    For frameworks that keep their own recent turns: system blocks (facts,
    retrieved memories) come first, then stored turns the framework does not
    already show (``exclude`` holds the texts it shows), as "Earlier
    conversation".
    """
    seen = {t.strip() for t in exclude if t}
    blocks, earlier = [], []
    for m in messages:
        if m["role"] == "system":
            blocks.append(m["content"])
        elif m["content"].strip() not in seen:
            earlier.append(f"- {m['role']}: {m['content']}")
    if earlier:
        blocks.append("Earlier conversation:\n" + "\n".join(earlier))
    return "\n\n".join(blocks)
