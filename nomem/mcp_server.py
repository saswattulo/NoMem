"""NoMem as an MCP server (``pip install nomem[mcp]``, then ``nomem mcp``).

Gives any MCP client (Claude Desktop, Claude Code, Cursor, ...) a memory it
can write to and read from. The tools wrap :class:`~nomem.service.MemoryService`;
nothing behind them calls a language model.
"""

from __future__ import annotations

from typing import Optional

from .service import MemoryService

INSTRUCTIONS = (
    "Long-term memory for this user. Call `remember` with anything worth keeping "
    "(facts, preferences, plans, decisions), verbatim. Before answering a question "
    "that may depend on the past, call `recall` with the question. Memories are "
    "shown with their dates; a [CONFLICT: ...] marker means a later memory changed "
    "an earlier one, so prefer the later one unless the user says otherwise."
)


def _server_class():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        pass
    try:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError as e:
        raise SystemExit("The MCP server needs the mcp package: pip install 'nomem[mcp]'") from e


def build_server(service: MemoryService, user_id: str = "default", session_id: str = "default"):
    """Create an MCP server exposing NoMem tools for one user and session."""
    from . import __version__

    mcp = _server_class()("nomem", instructions=INSTRUCTIONS)
    try:
        mcp._mcp_server.version = __version__  # best effort; not all versions expose it
    except AttributeError:
        pass

    @mcp.tool()
    def remember(content: str, role: str = "user") -> dict:
        """Store a message or fact verbatim in long-term memory.

        Args:
            content: The text to keep, exactly as said.
            role: Who said it: "user" (default) or "assistant".
        """
        out = service.add(content, role, user_id=user_id, session_id=session_id)
        return {"id": out["id"], "stored": True, "conflicts": out["conflicts"]}

    @mcp.tool()
    def recall(query: str, max_tokens: int = 1500) -> str:
        """Memory relevant to a question, as dated text that fits max_tokens.

        Args:
            query: The question or topic to recall memories for.
            max_tokens: Largest size of the returned text, in tokens.
        """
        return service.context(query, user_id=user_id, session_id=session_id,
                               max_tokens=max_tokens)["text"]

    @mcp.tool()
    def search_memory(query: str, k: int = 5) -> list[dict]:
        """The k best matching memories, with scores and why each matched."""
        return service.search(query, user_id=user_id, session_id=session_id, k=k)

    @mcp.tool()
    def list_facts(history: bool = False) -> dict:
        """Facts NoMem extracted (location, job, likes ...) and facts pinned with pin_fact.

        Args:
            history: Also include facts that were later replaced.
        """
        return service.facts(user_id=user_id, history=history)

    @mcp.tool()
    def pin_fact(key: str, value: str) -> dict:
        """Pin a durable fact, such as a preferred name, shown in every recall."""
        return service.remember(key, value, user_id=user_id)

    @mcp.tool()
    def list_conflicts() -> list[dict]:
        """Pairs of memories where a newer one changes an older one, with dates."""
        return service.conflicts(user_id=user_id)

    return mcp


def run(service: MemoryService, user_id: str = "default", session_id: str = "default",
        transport: str = "stdio", host: Optional[str] = None, port: Optional[int] = None) -> None:
    server = build_server(service, user_id, session_id)
    kwargs = {}
    if transport != "stdio":
        if host:
            kwargs["host"] = host
        if port:
            kwargs["port"] = port
    try:
        server.run(transport=transport, **kwargs)
    except TypeError:
        if transport != "stdio":  # older SDKs take host/port in the constructor
            server.settings.host = host or server.settings.host
            server.settings.port = port or server.settings.port
        server.run(transport=transport)
