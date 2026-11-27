"""NoMem — Not Only Memory.

A transparent, token-budgeted memory layer for agentic and RAG applications.
Framework-neutral core; local-first (SQLite); no LLM anywhere, by design.
"""

from .compactor import Compactor, ExtractiveCompactor, NullCompactor
from .core import NoMem
from .embed import Embedder, HashingEmbedder, Model2VecEmbedder, SentenceTransformerEmbedder
from .retrieval import Hit, RetrievalConfig
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
    "ExtractiveCompactor",
    "NullCompactor",
    "RetrievalConfig",
    "Hit",
    "Embedder",
    "HashingEmbedder",
    "Model2VecEmbedder",
    "SentenceTransformerEmbedder",
]
