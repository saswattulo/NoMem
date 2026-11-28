"""NoMem tools for CrewAI. ``pip install "nomem-ai[crewai]"``.

:func:`memory_tools` returns CrewAI ``BaseTool`` objects to give an agent
(``Agent(tools=memory_tools(mem))``), so it can remember and recall across
crews and runs without CrewAI's own LLM-based memory.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core import NoMem
from ._toolkit import Backend, Memory, MemoryToolkit, args_model, drop_none, to_text

try:
    from crewai.tools import BaseTool
    from pydantic import PrivateAttr
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('nomem.integrations.crewai needs crewai: '
                      'pip install "nomem-ai[crewai]"') from e


class NoMemCrewTool(BaseTool):
    """One NoMem tool as a CrewAI tool. Built by :func:`memory_tools`."""

    _toolkit: Any = PrivateAttr()

    def __init__(self, *, toolkit: MemoryToolkit, **data: Any):
        super().__init__(**data)
        self._toolkit = toolkit

    def _run(self, **kwargs: Any) -> str:
        return to_text(self._toolkit.call(self.name, drop_none(kwargs)))

    async def _arun(self, **kwargs: Any) -> str:
        return to_text(await self._toolkit.acall(self.name, drop_none(kwargs)))


def memory_tools(memory: Memory, *, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, tools: Optional[Iterable[str]] = None,
                 **toolkit_kwargs) -> list:
    """NoMem tools as CrewAI ``BaseTool`` objects."""
    if isinstance(memory, MemoryToolkit):
        toolkit = memory
    else:
        backend = memory if isinstance(memory, Backend) else (
            Backend(memory) if isinstance(memory, NoMem) else Backend(memory, user_id, session_id))
        toolkit = MemoryToolkit(backend, tools=tools, **toolkit_kwargs)

    return [NoMemCrewTool(name=s.name, description=s.description, args_schema=args_model(s),
                          toolkit=toolkit) for s in toolkit]
