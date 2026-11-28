"""NoMem — Not Only Memory.

A transparent, token-budgeted memory layer for agentic and RAG applications.
Framework-neutral core; local-first (SQLite); no LLM anywhere, by design.
"""

from .compactor import Compactor, ExtractiveCompactor, NullCompactor
from .conflicts import ConflictConfig
from .core import NoMem
from .embed import Embedder, HashingEmbedder, Model2VecEmbedder, SentenceTransformerEmbedder
from .retrieval import Hit, RetrievalConfig
from .models import ContextResult, LogEntry, Message
from .service import MemoryService
from .store import SQLiteStore
from .tokens import (
    AnthropicCounter,
    ApproxCounter,
    OpenAICounter,
    TokenCounter,
    default_counter,
)

__version__ = "0.3.0"

__all__ = [
    "NoMem",
    "Message",
    "LogEntry",
    "ContextResult",
    "SQLiteStore",
    "MemoryService",
    "TokenCounter",
    "ApproxCounter",
    "OpenAICounter",
    "AnthropicCounter",
    "default_counter",
    "Compactor",
    "ExtractiveCompactor",
    "NullCompactor",
    "RetrievalConfig",
    "ConflictConfig",
    "Hit",
    "Embedder",
    "HashingEmbedder",
    "Model2VecEmbedder",
    "SentenceTransformerEmbedder",
]
