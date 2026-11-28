"""NoMem for LlamaIndex. ``pip install "nomem-ai[llamaindex]"``.

* :func:`memory_tools` — NoMem tools as LlamaIndex ``FunctionTool`` objects
  for any agent (``FunctionAgent``, ``ReActAgent`` ...).
* :class:`NoMemMemoryBlock` — a long-term memory block for
  ``llama_index.core.memory.Memory``: messages that leave the short-term
  buffer are stored in NoMem, and each turn the block contributes the facts
  and dated memories relevant to the conversation.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from ..core import NoMem
from ._toolkit import (Backend, Memory, MemoryToolkit, args_model, content_text, drop_none,
                       latest_user_text, memory_text, to_text)

try:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.memory import BaseMemoryBlock
    from llama_index.core.tools import FunctionTool
    from pydantic import Field, PrivateAttr
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('nomem.integrations.llamaindex needs llama-index-core: '
                      'pip install "nomem-ai[llamaindex]"') from e


def _backend(memory: Any, user_id: Optional[str], session_id: Optional[str]) -> Backend:
    if isinstance(memory, Backend):
        return memory
    if isinstance(memory, NoMem):
        return Backend(memory)
    return Backend(memory, user_id, session_id)


def memory_tools(memory: Memory, *, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, tools: Optional[Iterable[str]] = None,
                 **toolkit_kwargs) -> list:
    """NoMem tools as LlamaIndex ``FunctionTool`` objects (sync and async)."""
    toolkit = memory if isinstance(memory, MemoryToolkit) else MemoryToolkit(
        _backend(memory, user_id, session_id), tools=tools, **toolkit_kwargs)

    def make(spec):
        def run(**kwargs) -> str:
            return to_text(toolkit.call(spec.name, drop_none(kwargs)))

        async def arun(**kwargs) -> str:
            return to_text(await toolkit.acall(spec.name, drop_none(kwargs)))

        return FunctionTool.from_defaults(fn=run, async_fn=arun, name=spec.name,
                                          description=spec.description, fn_schema=args_model(spec))

    return [make(s) for s in toolkit]


def _role(message: ChatMessage) -> Optional[str]:
    role = getattr(message.role, "value", message.role)
    return role if role in ("user", "assistant") else None


class NoMemMemoryBlock(BaseMemoryBlock[str]):
    """A LlamaIndex memory block backed by NoMem.

    Args:
        memory: A :class:`~nomem.NoMem` or :class:`~nomem.MemoryService`.
        user_id / session_id: Which memory a ``MemoryService`` serves.
        max_tokens: Budget for what the block returns.
    """

    name: str = "nomem"
    description: Optional[str] = "Long-term memory: known facts and dated earlier memories."
    memory: Any = Field(exclude=True)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    max_tokens: Optional[int] = None
    _backend: Backend = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._backend = _backend(self.memory, self.user_id, self.session_id)

    def _text(self, messages: Optional[List[ChatMessage]]) -> str:
        query = None
        for m in reversed(messages or []):
            if _role(m) == "user" and content_text(m.content).strip():
                query = content_text(m.content)
                break
        res = self._backend.context(query or latest_user_text(self._backend), self.max_tokens)
        return memory_text(res.messages, [content_text(m.content) for m in messages or []])

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **kwargs: Any) -> str:
        import asyncio

        return await asyncio.to_thread(self._text, messages)

    def _put(self, messages: List[ChatMessage]) -> None:
        for m in messages:
            role = _role(m)
            text = content_text(m.content) if role else ""
            if role and text.strip():
                self._backend.add(text, role)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        import asyncio

        await asyncio.to_thread(self._put, messages)
