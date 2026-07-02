"""Compaction interface.

When working memory overflows, dropped messages are handed to a ``Compactor``
which turns them into a compact summary that goes back into context. The
default (shipping in M3) is a deterministic, LLM-free *extractive* compactor;
an ``LLMCompactor`` can be dropped in behind the same interface for abstractive
summaries.

This module defines the extension point so the rest of the code and the public
API are stable ahead of M3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Message


class Compactor(ABC):
    @abstractmethod
    def compact(self, messages: list[Message], previous_summary: str = "") -> str:
        """Fold ``messages`` (plus any prior summary) into a compact summary."""
        ...


class NullCompactor(Compactor):
    """No-op compactor: drops overflow without summarising it.

    This is the M1 behaviour — a pure sliding window. Real compaction lands in
    M3.
    """

    def compact(self, messages: list[Message], previous_summary: str = "") -> str:
        return previous_summary
