"""NoMem for the Claude API (Anthropic SDK). ``pip install "nomem-ai[anthropic]"``.

Two ways to use it, which combine:

* **Context** — :meth:`AnthropicMemory.context` returns ``system`` and
  ``messages`` ready for ``client.messages.create(**ctx, ...)``: NoMem's
  memory blocks go in ``system`` and the turns are made to alternate the way
  the Messages API requires.
* **Tools** — :meth:`AnthropicMemory.tools` gives tool definitions for
  ``tools=``, and :meth:`AnthropicMemory.handle` runs the ``tool_use`` blocks
  of a response and returns the ``tool_result`` blocks to send back. For the
  SDK's tool runner use :meth:`AnthropicMemory.runner_tools`.

Nothing here calls a model: the adapter only shapes data for your own calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from ._toolkit import Memory, MemoryToolkit, alternate, content_text, latest_user_text, split_system


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class AnthropicMemory:
    """NoMem shaped for ``client.messages.create``.

    Args:
        memory: A :class:`~nomem.NoMem` or :class:`~nomem.MemoryService`.
        user_id / session_id: Which memory a ``MemoryService`` serves.
        tools: Tool names to expose (see :data:`nomem.integrations.TOOL_NAMES`).
        **toolkit_kwargs: Passed to :class:`~nomem.integrations.MemoryToolkit`
            (``recall_tokens``, ``max_k`` ...).
    """

    def __init__(self, memory: Memory, *, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, tools: Optional[Iterable[str]] = None,
                 **toolkit_kwargs):
        self.toolkit = MemoryToolkit(memory, user_id=user_id, session_id=session_id,
                                     tools=tools, **toolkit_kwargs)
        self.backend = self.toolkit.backend

    # -- context -----------------------------------------------------------

    def context(self, query: Optional[str] = None, *, system: str = "",
                max_tokens: Optional[int] = None) -> dict:
        """``{"system": ..., "messages": [...]}`` for ``messages.create``.

        ``query`` defaults to the newest stored user message. ``system`` is
        your own instructions, placed before NoMem's memory blocks. The
        turns start with ``user`` and alternate; if nothing is stored yet,
        ``messages`` is empty and you append the new user turn yourself.
        """
        query = query or latest_user_text(self.backend)
        res = self.backend.context(query, max_tokens)
        mem_system, turns = split_system(res.messages)
        joined = "\n\n".join(p for p in (system.strip(), mem_system) if p)
        return {"system": joined, "messages": alternate(turns)}

    # -- writing -----------------------------------------------------------

    def add_user(self, content: Any) -> Optional[dict]:
        """Store a user turn (a string or a list of content blocks). Empty text is skipped."""
        text = content_text(content)
        return self.backend.add(text, "user") if text.strip() else None

    def add_assistant(self, content: Any) -> Optional[dict]:
        """Store an assistant turn: a string, content blocks, or a whole ``Message``."""
        if not isinstance(content, (str, list)):
            content = _get(content, "content", "")
        text = content_text(content)
        return self.backend.add(text, "assistant") if text.strip() else None

    record = add_assistant

    # -- tools -------------------------------------------------------------

    def tools(self, cache_control: Optional[dict] = None) -> list[dict]:
        """Tool definitions for ``messages.create(tools=...)``.

        ``cache_control`` (e.g. ``{"type": "ephemeral"}``) is set on the last
        tool so the definitions can be prompt-cached.
        """
        out = [{"name": s.name, "description": s.description, "input_schema": s.parameters}
               for s in self.toolkit]
        if cache_control and out:
            out[-1] = out[-1] | {"cache_control": cache_control}
        return out

    def handles(self, name: str) -> bool:
        return name in self.toolkit.names

    def handle(self, response: Any) -> list[dict]:
        """Run the NoMem ``tool_use`` blocks in a response.

        ``response`` is a ``Message``, its ``content`` list, or dicts of the
        same shape. Returns one ``tool_result`` block per NoMem tool call, in
        order, to send back as the next user turn's content. Tool calls for
        other tools are left for you; errors come back with ``is_error``.
        """
        results = []
        for block in self._tool_uses(response):
            text, err = self.toolkit.call_text(_get(block, "name"), _get(block, "input") or {})
            result = {"type": "tool_result", "tool_use_id": _get(block, "id"), "content": text}
            if err:
                result["is_error"] = True
            results.append(result)
        return results

    async def ahandle(self, response: Any) -> list[dict]:
        return await asyncio.to_thread(self.handle, response)

    def _tool_uses(self, response: Any) -> list:
        content = response if isinstance(response, list) else _get(response, "content", []) or []
        return [b for b in content
                if _get(b, "type") == "tool_use" and self.handles(_get(b, "name"))]

    def runner_tools(self) -> list:
        """The tools as SDK function tools, for ``client.beta.messages.tool_runner``."""
        try:
            from anthropic import beta_tool
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError('runner_tools needs a recent anthropic SDK: '
                              'pip install "nomem-ai[anthropic]"') from e

        def make(spec):
            def run(**kwargs) -> str:
                return self.toolkit.call_text(spec.name, kwargs)[0]
            run.__name__ = spec.name
            run.__doc__ = spec.description
            return beta_tool(run, name=spec.name, description=spec.description,
                             input_schema=spec.parameters)

        return [make(s) for s in self.toolkit]


def memory_tools(memory: Memory, **kwargs) -> list[dict]:
    """Shortcut: Anthropic tool definitions for ``memory``."""
    return AnthropicMemory(memory, **kwargs).tools()
