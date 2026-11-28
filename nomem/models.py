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

    action: str  # "core" | "facts" | "added" | "dropped" | "budget_exhausted" | "retrieved" | "retrieval" | "skipped" | "summarized"
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

    def usage(self) -> dict:
        """Tokens spent per part of the context, from the log.

        Keys: ``core`` (persona), ``facts`` (Known-facts card), ``retrieved``
        (archival memories), ``summary`` (extractive summary), ``recent``
        (the working window) and ``free`` (budget left unused).
        """
        kinds = {"core": "core", "facts": "facts", "retrieved": "retrieved",
                 "summarized": "summary", "added": "recent"}
        out = {k: 0 for k in ("core", "facts", "retrieved", "summary", "recent")}
        for e in self.log:
            key = kinds.get(e.action)
            if key:
                out[key] += e.tokens
        out["free"] = max(self.max_tokens - self.tokens_used, 0)
        return out

    def gauge(self, width: int = 30, color: bool = False) -> str:
        """The budget as a bar: ``▕█████████████████████████░░░░░▏ 286/300 tokens``."""
        from .banner import gauge
        return gauge(self.tokens_used, self.max_tokens, width, color)

    def budget_map(self, width: int = 30, color: bool = False) -> str:
        """The budget as cells, one per ``max_tokens/width`` tokens, by part:
        ``◆●●●■■■■■■■■■■■■▪▪▪▪▪▪▪▪▪▪▪▪▪·`` plus a legend with token counts."""
        from .banner import budget_map
        return budget_map(self.usage(), self.max_tokens, width, color)

    def explain(self, color: bool = False) -> str:
        """Human-readable dump of every context-assembly decision.

        Opens with the budget gauge and the budget map, then one line per
        decision. ``color=True`` adds ANSI colours for a terminal.
        """
        header = f"context {self.gauge(color=color)}, {len(self.messages)} messages"
        lines = [header, "budget  " + self.budget_map(color=color), "-" * 64]
        lines += [str(e) for e in self.log]
        return "\n".join(lines)
