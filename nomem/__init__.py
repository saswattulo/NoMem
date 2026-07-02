"""NoMem — Not Only Memory.

A transparent, token-budgeted memory layer for agentic and RAG applications.
Framework-neutral core; local-first (SQLite); no LLM or API key required.
"""

from .compactor import Compactor, NullCompactor
from .core import NoMem
from .models import ContextResult, LogEntry, Message
from .store import SQLiteStore
from .tokens import (
    AnthropicCounter,
    ApproxCounter,
    OpenAICounter,
    TokenCounter,
    default_counter,
)

__version__ = "0.1.0"

__all__ = [
    "NoMem",
    "Message",
    "LogEntry",
    "ContextResult",
    "SQLiteStore",
    "TokenCounter",
    "ApproxCounter",
    "OpenAICounter",
    "AnthropicCounter",
    "default_counter",
    "Compactor",
    "NullCompactor",
]
