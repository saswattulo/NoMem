"""Optional vector retrieval. Embedding models only: never a generative model.

NoMem works without any of this. Pass an embedder to add a ``vector`` ranker
to hybrid retrieval:

    from nomem import NoMem
    from nomem.embed import Model2VecEmbedder
    mem = NoMem(embedder=Model2VecEmbedder())      # pip install nomem[embed]

Embedders:
- :class:`Model2VecEmbedder` — **recommended default.** Static embeddings
  distilled from a sentence transformer (``minishlab/potion-base-8M``,
  ~8M parameters, ~30 MB, CPU only, thousands of sentences per second).
- :class:`SentenceTransformerEmbedder` — any sentence-transformers model
  (e.g. ``BAAI/bge-small-en-v1.5``); stronger, heavier, slower.
- :class:`HashingEmbedder` — zero-dependency character n-gram hashing. Not a
  neural model; catches spelling variants and word-form overlap, useful
  offline and in tests.
- Subclass :class:`Embedder` and implement :meth:`encode` for anything else.

Vectors are stored in the same SQLite file and held in an in-memory matrix
per session for brute-force cosine search (fast with numpy installed).
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from array import array
from typing import Optional

try:  # optional accelerator
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None


class Embedder(ABC):
    """Turns text into L2-normalised vectors and searches them.

    ``min_similarity`` drops weak matches: a hit must reach this cosine and
    at least ``relative_floor`` x the best hit's cosine, so an unrelated
    query returns nothing rather than the least-unrelated messages.
    """

    min_similarity: float = 0.1
    relative_floor: float = 0.5

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...

    # -- storage + search (shared by every embedder) ----------------------

    def _cache(self) -> dict:
        if not hasattr(self, "_mats"):
            self._mats: dict = {}
        return self._mats

    def index(self, store, msg, user_id: str, session_id: str) -> None:
        vec = _normalise(self.encode([msg.content])[0])
        store.add_vector(msg.id, user_id, session_id, array("f", vec).tobytes(), commit=False)
        entry = self._cache().get((id(store), user_id, session_id))
        if entry is not None:
            entry["ids"].append(msg.id)
            entry["rows"].append(vec)
            entry["mat"] = None  # rebuilt lazily on next search

    def _matrix(self, store, user_id: str, session_id: str):
        key = (id(store), user_id, session_id)
        entry = self._cache().get(key)
        if entry is None:
            ids, rows = [], []
            for mid, blob in store.vectors(user_id, session_id):
                ids.append(mid)
                rows.append(list(array("f", blob)))
            entry = {"ids": ids, "rows": rows, "mat": None}
            self._cache()[key] = entry
        if _np is not None and entry["mat"] is None and entry["rows"]:
            entry["mat"] = _np.asarray(entry["rows"], dtype=_np.float32)
        return entry

    def search(self, store, user_id: str, session_id: str, query: str, k: int) -> list[int]:
        entry = self._matrix(store, user_id, session_id)
        if not entry["ids"]:
            return []
        q = _normalise(self.encode([query])[0])
        if _np is not None:
            sims = entry["mat"] @ _np.asarray(q, dtype=_np.float32)
            k = min(k, len(sims))
            top = _np.argpartition(-sims, k - 1)[:k]
            top = top[_np.argsort(-sims[top])].tolist()
            sims = sims.tolist()
        else:
            sims = [sum(a * b for a, b in zip(row, q)) for row in entry["rows"]]
            top = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        if not top:
            return []
        floor = max(self.min_similarity, self.relative_floor * sims[top[0]])
        return [entry["ids"][i] for i in top if sims[i] >= floor]


def _normalise(v) -> list[float]:
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashingEmbedder(Embedder):
    """Character n-gram feature hashing (no model, no dependency).

    Each word contributes its character 3- to 5-grams (with boundary marks)
    hashed into ``dim`` buckets with a sign bit; the vector is L2-normalised.
    """

    def __init__(self, dim: int = 512, ngrams: tuple = (3, 4, 5)):
        self.dim = dim
        self.ngrams = ngrams

    def encode(self, texts: list[str]) -> list[list[float]]:
        from .annotate import content_terms

        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in content_terms(t):
                w = f"<{w}>"
                for n in self.ngrams:
                    for i in range(max(1, len(w) - n + 1)):
                        h = int.from_bytes(hashlib.blake2b(w[i:i + n].encode(), digest_size=4).digest(), "big")
                        v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            out.append(v)
        return out


class Model2VecEmbedder(Embedder):
    """Static Model2Vec embeddings (``pip install nomem[embed]``).

    Default model ``minishlab/potion-base-8M``: small, CPU-only, and fast
    enough to embed every message on write. The model downloads once from the
    Hugging Face Hub; pass a local path to run fully offline.
    """

    def __init__(self, model: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel  # lazy: keeps the core dependency-free

        self.model_name = model
        self._model = StaticModel.from_pretrained(model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts)]


class SentenceTransformerEmbedder(Embedder):
    """Any sentence-transformers encoder (``pip install sentence-transformers``)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", **kwargs):
        from sentence_transformers import SentenceTransformer  # lazy

        self.model_name = model
        self._model = SentenceTransformer(model, **kwargs)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


def default_embedder() -> Optional[Embedder]:
    """Model2Vec if installed, else ``None`` (NoMem stays model-free)."""
    try:
        return Model2VecEmbedder()
    except Exception:
        return None
