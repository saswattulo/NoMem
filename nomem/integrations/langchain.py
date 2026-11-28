"""NoMem for LangChain. ``pip install "nomem-ai[langchain]"``.

* :class:`NoMemChatMessageHistory` — a chat message history for
  ``RunnableWithMessageHistory``: turns are stored in NoMem, and the history
  the chain sees is a token-budgeted context, not the whole log.
* :class:`NoMemRetriever` — a retriever over stored memories (hybrid keyword,
  entity, time and optional vector search), returning ``Document`` objects.
* :func:`memory_tools` — NoMem tools as ``StructuredTool`` objects, for any
  tool-calling model or agent (including LangGraph's ``ToolNode``).
* :func:`to_messages` — NoMem context messages as LangChain messages.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional, Sequence

from ..core import NoMem
from ._toolkit import Backend, Memory, MemoryToolkit, content_text, latest_user_text, to_text

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import StructuredTool
    from pydantic import ConfigDict, PrivateAttr
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('nomem.integrations.langchain needs langchain-core: '
                      'pip install "nomem-ai[langchain]"') from e


_ROLE = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}


def _backend(memory: Any, user_id: Optional[str], session_id: Optional[str]) -> Backend:
    if isinstance(memory, Backend):
        return memory.with_session(session_id) if session_id else memory
    if isinstance(memory, NoMem):
        b = Backend(memory)
        return b.with_session(session_id) if session_id else b
    return Backend(memory, user_id, session_id)


def to_messages(messages: Sequence[dict]) -> list[BaseMessage]:
    """NoMem context dicts (``{"role", "content"}``) as LangChain messages."""
    out: list[BaseMessage] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            out.append(HumanMessage(content))
        elif role == "assistant":
            out.append(AIMessage(content))
        else:
            out.append(SystemMessage(content))
    return out


def message_role(message: BaseMessage) -> Optional[str]:
    """``user`` / ``assistant`` for chat turns worth storing, else ``None``."""
    return _ROLE.get(getattr(message, "type", ""))


class NoMemChatMessageHistory(BaseChatMessageHistory):
    """Chat history stored in NoMem and read back within a token budget.

    Args:
        memory: A :class:`~nomem.NoMem`, or a :class:`~nomem.MemoryService`
            (then ``session_id`` picks the conversation and ``user_id`` the user).
        session_id: The conversation.
        max_tokens: Budget for ``messages``. Defaults to the memory's own.

    ``messages`` recalls memories for the newest stored user message. To
    recall for something else, call :meth:`messages_for`.

    Only text is stored: tool calls, tool results and system messages are
    skipped, as are AI messages that carry nothing but tool calls.
    """

    def __init__(self, memory: Memory, session_id: Optional[str] = None, *,
                 user_id: Optional[str] = None, max_tokens: Optional[int] = None):
        self.backend = _backend(memory, user_id, session_id)
        self.session_id = self.backend.session_id
        self.max_tokens = max_tokens

    def messages_for(self, query: Optional[str]) -> list[BaseMessage]:
        res = self.backend.context(query or latest_user_text(self.backend), self.max_tokens)
        return to_messages(res.messages)

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        return self.messages_for(None)

    async def aget_messages(self) -> list[BaseMessage]:
        return await asyncio.to_thread(self.messages_for, None)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for m in messages:
            role = message_role(m)
            text = content_text(m.content) if role else ""
            if role and text.strip():
                self.backend.add(text, role)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        await asyncio.to_thread(self.add_messages, messages)

    def clear(self) -> None:
        self.backend.clear()

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)


class NoMemRetriever(BaseRetriever):
    """Retrieve stored memories as ``Document`` objects, best first.

    Each document's ``page_content`` is the verbatim message; ``metadata``
    holds ``id``, ``role``, ``created_at`` (ISO 8601), ``score`` and
    ``reasons`` (which rankers found it).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: Any
    k: int = 5
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    _backend: Backend = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._backend = _backend(self.memory, self.user_id, self.session_id)

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if not query or not query.strip():
            return []
        docs = []
        for h in self._backend.search(query, self.k):
            meta = {k: h[k] for k in ("id", "role", "created_at", "score", "reasons")}
            docs.append(Document(page_content=h["content"], metadata=(h.get("metadata") or {}) | meta))
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)


def memory_tools(memory: Memory, *, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, tools: Optional[Iterable[str]] = None,
                 **toolkit_kwargs) -> list[StructuredTool]:
    """NoMem tools as LangChain ``StructuredTool`` objects (sync and async)."""
    toolkit = memory if isinstance(memory, MemoryToolkit) else MemoryToolkit(
        _backend(memory, user_id, session_id), tools=tools, **toolkit_kwargs)

    def make(spec):
        def run(**kwargs) -> str:
            return to_text(toolkit.call(spec.name, kwargs))

        async def arun(**kwargs) -> str:
            return to_text(await toolkit.acall(spec.name, kwargs))

        return StructuredTool.from_function(func=run, coroutine=arun, name=spec.name,
                                            description=spec.description,
                                            args_schema=spec.parameters)

    return [make(s) for s in toolkit]
