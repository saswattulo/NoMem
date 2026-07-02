"""Core data structures for NoMem.

These are deliberately plain dataclasses so they serialise cleanly, print
nicely in a notebook, and don't couple the core to any framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A single turn stored in a memory tier."""

    role: str
    content: str
    id: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    tokens: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LogEntry:
    """One transparent decision made while assembling context.

    The whole point of NoMem is that nothing about the context window is a
    black box: every message that is kept, dropped, retrieved, or summarised
    produces one of these.
    """

    action: str  # "core" | "added" | "dropped" | "budget_exhausted" | "retrieved" | "summarized"
    tier: str  # "core" | "working" | "archival"
    detail: str
    tokens: int = 0
    budget_remaining: Optional[int] = None
    ref: Optional[Any] = None  # message id or similar

    def __str__(self) -> str:
        left = "" if self.budget_remaining is None else f" [{self.budget_remaining} left]"
        return f"{self.action:<16} {self.tier:<9} {self.tokens:>5}t{left}  {self.detail}"


@dataclass
class ContextResult:
    """The output of :meth:`NoMem.build_context`.

    ``text`` is the rendered prompt string, ``messages`` the structured form
    for chat APIs, and ``log`` the full trail of what happened and why.
    """

    text: str
    messages: list[dict]
    tokens_used: int
    max_tokens: int
    log: list[LogEntry] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text

    def explain(self) -> str:
        """Human-readable dump of every context-assembly decision."""
        header = f"context: {self.tokens_used}/{self.max_tokens} tokens, {len(self.messages)} messages"
        lines = [header, "-" * len(header)]
        lines += [str(e) for e in self.log]
        return "\n".join(lines)
