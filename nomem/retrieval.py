"""Hybrid, LLM-free archival retrieval.

Several cheap rankers each propose candidates; Reciprocal Rank Fusion merges
them; a light prior (importance, recency, repetition) re-orders the fused
list; near-duplicates are collapsed. Every step is deterministic and every
candidate carries the reasons it was picked, so ``explain()`` can show why.

Rankers:
- ``keyword``  SQLite FTS5 BM25 (always on when FTS5 is available)
- ``entity``   messages sharing the query's named entities, rarest first
- ``time``     messages sent in, or talking about, a time range in the query
- ``vector``   cosine similarity, when an :class:`~nomem.embed.Embedder` is set
- ``context``  the turns either side of a strong hit (an answer often sits
               next to the turn that shares the question's words)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import annotate as A
from .models import Message


@dataclass
class RetrievalConfig:
    """Knobs for :class:`Retriever`. Defaults are what the benchmarks back."""

    pool: int = 50                 # candidates taken from each ranker
    rrf_k: int = 60                # Cormack et al. 2009
    weights: dict = field(default_factory=lambda: {
        "keyword": 1.0, "entity": 0.5, "time": 0.5, "vector": 1.0, "context": 0.2, "slot": 0.5,
    })
    importance_weight: float = 0.3  # final *= (1 - w/2 + w * importance)
    recency_half_life_days: Optional[float] = None  # None = no recency prior
    near_dup_bits: int = 3         # SimHash Hamming distance treated as a near-duplicate
    context_span: int = 1          # neighbours either side of a top hit
    context_seeds: int = 5         # how many top hits get neighbours
    entity_max_df: float = 0.02    # ignore entities in more than this share of messages


@dataclass
class Hit:
    message: Message
    score: float
    reasons: list[str]


class Retriever:
    def __init__(self, store, user_id: str, session_id: str,
                 config: Optional[RetrievalConfig] = None, embedder=None):
        self.store = store
        self.user_id = user_id
        self.session_id = session_id
        self.config = config or RetrievalConfig()
        self.embedder = embedder

    # ------------------------------------------------------------------
    def rank_lists(self, query: str, now: float) -> dict[str, list[int]]:
        c, s, u, sid = self.config, self.store, self.user_id, self.session_id
        lists: dict[str, list[int]] = {}
        w = c.weights
        if w.get("keyword"):
            lists["keyword"] = s.fts_ranked(u, sid, query, c.pool)
        if w.get("entity"):
            ents = A.entities(query)
            if ents:
                max_df = max(3, int(c.entity_max_df * s.count_messages(u, sid)))
                lists["entity"] = s.entity_ranked(u, sid, ents, c.pool, max_df)
        if w.get("time"):
            ranges = A.parse_times(query, now)
            if ranges:
                lists["time"] = s.time_ranked(u, sid, ranges, c.pool)
        if w.get("slot"):
            sl = slot_matches(s, u, query)
            if sl:
                lists["slot"] = [r["source_message_id"] for r in sl if r["source_message_id"]][: c.pool]
        if w.get("vector") and self.embedder is not None:
            lists["vector"] = self.embedder.search(s, u, sid, query, c.pool)
        return lists

    def search(self, query: str, k: int, now: float) -> list[Hit]:
        if k <= 0 or not query or not query.strip():
            return []
        c = self.config
        lists = self.rank_lists(query, now)
        fused = _fuse(lists, c.weights, c.rrf_k, scores=True)
        if not fused:
            return []
        # Neighbouring turns: an answer often sits right after the turn that
        # shares the question's words. Each neighbour of a top hit gains a
        # share of that hit's score (never a rank of its own, so a neighbour
        # with no evidence of its own can't outrank the hit that found it).
        cw = c.weights.get("context", 0)
        if cw and c.context_span > 0:
            seeds = fused[: c.context_seeds]
            nb = self.store.neighbors(self.user_id, self.session_id, [i for i, _ in seeds], c.context_span)
            acc = dict(fused)
            ctx: list[int] = []
            for i, sc in seeds:
                for j in nb.get(i, []):
                    acc[j] = acc.get(j, 0.0) + cw * sc
                    if j not in ctx:
                        ctx.append(j)
            lists["context"] = ctx
            fused = sorted(acc.items(), key=lambda t: (-t[1], -t[0]))
        ids = [i for i, _ in fused]
        msgs = self.store.get_messages(ids)
        ann = self.store.annotations(ids)
        reasons: dict[int, list[str]] = {}
        for name, lst in lists.items():
            for rank, i in enumerate(lst):
                reasons.setdefault(i, []).append(f"{name}#{rank + 1}")

        scored: list[tuple[float, int]] = []
        for i, base in fused:
            if i not in msgs:
                continue
            a = ann.get(i, {})
            imp = a.get("importance", 0.5)
            score = base * (1 - c.importance_weight / 2 + c.importance_weight * imp)
            if c.recency_half_life_days:
                age_days = max(0.0, (now - msgs[i].created_at) / 86400)
                score *= 0.5 + 0.5 * math.exp(-math.log(2) * age_days / c.recency_half_life_days)
            scored.append((score, i))
        scored.sort(key=lambda t: (-t[0], -t[1]))

        # Collapse exact and near duplicates: keep the best-scored copy and
        # note how many times it was said (repetition is a signal, not noise).
        hits: list[Hit] = []
        kept_hashes: list[tuple[str, int, Hit]] = []
        for score, i in scored:
            a = ann.get(i, {})
            h, sh = a.get("norm_hash"), a.get("simhash")
            dup = None
            for kh, ksh, hit in kept_hashes:
                if (h and h == kh) or (sh is not None and ksh is not None
                                       and A.hamming(sh, ksh) <= c.near_dup_bits):
                    dup = hit
                    break
            if dup is not None:
                dup.reasons.append(f"repeats #{i}")
                continue
            hit = Hit(msgs[i], score, reasons.get(i, []))
            hits.append(hit)
            kept_hashes.append((h, sh, hit))
            if len(hits) >= k:
                break
        return hits


# Words in a question that point at a slot attribute.
SLOT_SYNONYMS = {
    "name": {"name", "called"},
    "age": {"age", "old", "birthday"},
    "location": {"live", "living", "lives", "location", "city", "based", "moved"},
    "job": {"job", "work", "occupation", "profession", "career", "do for a living"},
    "employer": {"work", "employer", "company", "job"},
    "diet": {"diet", "eat", "vegetarian", "vegan", "food"},
    "allergy": {"allergy", "allergic", "allergies"},
    "likes": {"like", "likes", "love", "enjoy", "favorite", "favourite", "prefer"},
    "dislikes": {"dislike", "hate", "avoid"},
}


def slot_matches(store, user_id: str, query: str, include_history: bool = True) -> list[dict]:
    """Slot rows a query asks about: by attribute name/synonym or by value.

    Current values come first, then superseded ones (newest first), so a
    question about "where do I live" surfaces the latest answer and, behind
    it, the history that explains it.
    """
    terms = set(A.content_terms(query)) | {w.lower() for w in A.words(query)}
    rows = store.slot_history(user_id) if include_history else store.current_slots(user_id)
    # Whose facts? A named subject in the question wins; otherwise "my/I"
    # means the user; otherwise any subject.
    subjects = {r["subject"] for r in rows}
    named = {sub for sub in subjects if sub.lower() in terms}
    if named:
        rows = [r for r in rows if r["subject"] in named]
    elif terms & {"i", "my", "me", "i'm", "mine"} and "user" in subjects:
        rows = [r for r in rows if r["subject"] == "user"]
    out = []
    for r in rows:
        attr = r["attribute"]
        names = SLOT_SYNONYMS.get(attr, set()) | set(attr.split("_"))
        value_terms = set(A.content_terms(r["value"]))
        if (names & terms) or (value_terms and value_terms <= terms):
            out.append(r)
    out.sort(key=lambda r: (r["valid_to"] is not None, -(r["valid_from"] or 0)))
    return out


def _fuse(lists: dict[str, list[int]], weights: dict, rrf_k: int, scores: bool = False):
    acc: dict[int, float] = {}
    for name, lst in lists.items():
        w = weights.get(name, 1.0)
        for rank, i in enumerate(lst):
            acc[i] = acc.get(i, 0.0) + w / (rrf_k + rank + 1)
    order = sorted(acc.items(), key=lambda t: (-t[1], -t[0]))
    return order if scores else [i for i, _ in order]
