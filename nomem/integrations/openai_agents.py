"""NoMem for the OpenAI Agents SDK. ``pip install "nomem-ai[openai-agents]"``.

* :func:`memory_tools` — NoMem tools as ``FunctionTool`` objects for
  ``Agent(tools=...)``.
* :class:`NoMemSession` — a ``Session`` for ``Runner.run(..., session=...)``:
  turns are stored in NoMem, and each run gets a token-budgeted history
  (recent turns + retrieved memories + known facts) instead of the whole log.

For the best recall, also pass the session's input callback, so the memories
are retrieved for the message being answered rather than the previous one::

    result = await Runner.run(agent, "Where do I live?", session=session,
                              run_config=RunConfig(session_input_callback=session.input_callback))
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from ..core import NoMem
from ._toolkit import Backend, Memory, MemoryToolkit, content_text, latest_user_text, to_text

try:
    from agents import FunctionTool
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('nomem.integrations.openai_agents needs the OpenAI Agents SDK: '
                      'pip install "nomem-ai[openai-agents]"') from e


def memory_tools(memory: Memory, *, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, tools: Optional[Iterable[str]] = None,
                 **toolkit_kwargs) -> list:
    """NoMem tools as Agents SDK ``FunctionTool`` objects."""
    toolkit = memory if isinstance(memory, MemoryToolkit) else MemoryToolkit(
        memory, user_id=user_id, session_id=session_id, tools=tools, **toolkit_kwargs)

    def make(spec):
        async def invoke(ctx, arguments: str) -> str:
            return to_text(await toolkit.acall(spec.name, arguments))

        # Optional arguments mean the schema cannot be "strict" (strict requires
        # every property); NoMem validates the arguments itself instead.
        return FunctionTool(name=spec.name, description=spec.description,
                            params_json_schema=spec.parameters, on_invoke_tool=invoke,
                            strict_json_schema=False)

    return [make(s) for s in toolkit]


def _role(item: dict) -> Optional[str]:
    if item.get("type") not in (None, "message"):
        return None
    role = item.get("role")
    return role if role in ("user", "assistant") else None


class NoMemSession:
    """An Agents SDK session backed by NoMem.

    Every user and assistant message is stored verbatim. ``get_items`` does
    not replay the whole log: it returns NoMem's context for the session, so
    long conversations stay within ``max_tokens`` and old turns that matter
    come back through retrieval.

    Tool calls, tool outputs and reasoning items are not stored (they are
    rarely worth remembering and often large); only their text messages are.

    Args:
        memory: A :class:`~nomem.NoMem`, or a :class:`~nomem.MemoryService`
            with ``user_id`` (``session_id`` names the conversation).
        session_id: The conversation. Defaults to the NoMem's own.
        max_tokens: Budget for the history handed to each run.
    """

    session_settings = None

    def __init__(self, memory: Memory, session_id: Optional[str] = None, *,
                 user_id: Optional[str] = None, max_tokens: Optional[int] = None):
        if isinstance(memory, Backend):
            backend = memory
        elif isinstance(memory, NoMem):
            backend = Backend(memory)
        else:
            backend = Backend(memory, user_id, session_id)
        self.backend = backend.with_session(session_id) if session_id else backend
        self.session_id = self.backend.session_id
        self.max_tokens = max_tokens

    # -- reading ----------------------------------------------------------

    def _items(self, query: Optional[str], limit: Optional[int]) -> list[dict]:
        if limit is not None and limit <= 0:
            return []
        res = self.backend.context(query or latest_user_text(self.backend), self.max_tokens)
        items = [{"role": m["role"], "content": m["content"]} for m in res.messages]
        return items[-limit:] if limit else items

    async def get_items(self, limit: Optional[int] = None) -> list[dict]:
        return await asyncio.to_thread(self._items, None, limit)

    def input_callback(self, history: list, new_items: list) -> list:
        """A ``session_input_callback`` that recalls memories for the new input.

        ``get_items`` only knows the previous user message; this rebuilds the
        history with the message being answered as the query.
        """
        query = " ".join(content_text(i.get("content")) for i in new_items
                         if isinstance(i, dict) and _role(i) == "user").strip()
        if query:
            history = self._items(query, None)
        return list(history) + list(new_items)

    # -- writing ----------------------------------------------------------

    def _add(self, items: list) -> None:
        for item in items:
            if not isinstance(item, dict):
                item = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            role = _role(item)
            text = content_text(item.get("content")) if role else ""
            if role and text.strip():
                self.backend.add(text, role)

    async def add_items(self, items: list) -> None:
        await asyncio.to_thread(self._add, items)

    def _pop(self) -> Optional[dict]:
        last = self.backend.recent(1)
        if not last:
            return None
        msg = last[0]
        self.backend.forget([msg.id])
        return {"role": msg.role, "content": msg.content}

    async def pop_item(self) -> Optional[dict]:
        return await asyncio.to_thread(self._pop)

    async def clear_session(self) -> None:
        await asyncio.to_thread(self.backend.clear)
