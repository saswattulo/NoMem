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
import warnings
from dataclasses import dataclass, field
from typing import Optional

from . import annotate as A
from .models import Message


FUSIONS = ("score", "rrf")


@dataclass
class RetrievalConfig:
    """Knobs for :class:`Retriever`. Defaults are what the benchmarks back.

    ``RetrievalConfig.legacy()`` gives NoMem 0.2's rank-only fusion.
    """

    pool: int = 100                # candidates taken from each ranker
    # How ranker results are merged:
    # "score" - keyword and vector add their scores (BM25 relative to the
    #           query's best hit, squared; raw cosine), so a turn that is
    #           strong on one ranker isn't outvoted by one that is merely
    #           listed by several. Rank-only rankers (entity, time, slot) add
    #           ``weight * (rrf_k+1)/(rrf_k+rank+1)``: worth ``weight`` at #1.
    # "rrf"   - Reciprocal Rank Fusion over every ranker (rank only).
    fusion: str = "score"
    rrf_k: int = 60                # Cormack et al. 2009
    keyword_power: float = 2.0     # BM25 share is raised to this power ("score")
    weights: dict = field(default_factory=lambda: {
        "keyword": 1.0, "entity": 0.5, "time": 0.5, "vector": 1.0, "context": 0.35, "slot": 0.5,
    })
    importance_weight: float = 0.3  # final *= (1 - w/2 + w * importance)
    recency_half_life_days: Optional[float] = None  # None = no recency prior
    # Long turns (pasted documents, an assistant's 3,000-character answer)
    # match many query words by sheer size. Above ``length_pivot`` characters
    # a hit's score is multiplied by (pivot / length) ** length_penalty.
    length_penalty: float = 0.3
    length_pivot: int = 400
    near_dup_bits: int = 3         # SimHash Hamming distance treated as a near-duplicate
    context_span: int = 1          # neighbours either side of a top hit
    context_seeds: int = 5         # how many top hits get neighbours
    # Only lift a neighbour said in the same role as the hit (two people
    # talking, both stored as "user"): an assistant's reply to a user's turn
    # is usually advice about it, not more of the user's own facts.
    context_same_role: bool = True
    entity_max_df: float = 0.02    # ignore entities in more than this share of messages
    max_query_terms: int = 64      # longest distinct query words sent to the keyword index

    def __post_init__(self):
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {self.fusion!r}")
        if self.pool < 1:
            raise ValueError(f"pool must be >= 1, got {self.pool}")
        if self.rrf_k < 0:
            raise ValueError(f"rrf_k must be >= 0, got {self.rrf_k}")
        bad = {k: v for k, v in self.weights.items() if not (isinstance(v, (int, float)) and v >= 0)}
        if bad:
            raise ValueError(f"ranker weights must be numbers >= 0, got {bad}")
        if self.length_penalty < 0 or self.length_pivot < 1:
            raise ValueError("length_penalty must be >= 0 and length_pivot >= 1")

    @classmethod
    def legacy(cls, **overrides) -> "RetrievalConfig":
        """NoMem 0.2 retrieval: RRF over ranks, no length penalty, any-role neighbours."""
        base = dict(
            fusion="rrf", length_penalty=0.0, context_same_role=False,
            weights={"keyword": 1.0, "entity": 0.5, "time": 0.5, "vector": 0.15, "context": 0.2, "slot": 0.5},
        )
        return cls(**(base | overrides))


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
        return {name: [i for i, _ in lst] for name, lst in self.scored_lists(query, now).items()}

    def scored_lists(self, query: str, now: float) -> dict[str, list[tuple[int, Optional[float]]]]:
        """Each ranker's candidates, best first, with its raw score (None = rank only)."""
        c, s, u, sid = self.config, self.store, self.user_id, self.session_id
        lists: dict[str, list] = {}
        w = c.weights
        if w.get("keyword"):
            lists["keyword"] = s.fts_scored(u, sid, query, c.pool, c.max_query_terms)
        if w.get("entity"):
            ents = A.entities(query)
            if ents:
                max_df = max(3, int(c.entity_max_df * s.count_messages(u, sid)))
                lists["entity"] = [(i, None) for i in s.entity_ranked(u, sid, ents, c.pool, max_df)]
        if w.get("time"):
            ranges = A.parse_times(query, now)
            if ranges:
                lists["time"] = [(i, None) for i in s.time_ranked(u, sid, ranges, c.pool)]
        if w.get("slot"):
            sl = slot_matches(s, u, query)
            src = [r["source_message_id"] for r in sl if r["source_message_id"]]
            if src:
                # Slots are per user, but search is per session: only point
                # at source messages that belong to this session.
                ok = s.in_session(u, sid, src)
                lists["slot"] = [(i, None) for i in dict.fromkeys(src) if i in ok][: c.pool]
        if w.get("vector") and self.embedder is not None:
            try:
                lists["vector"] = self.embedder.search_scored(s, u, sid, query, c.pool)
            except Exception as e:  # degrade to the other rankers rather than fail the read
                warnings.warn(f"vector ranker skipped, embedder failed: {e!r}", RuntimeWarning, stacklevel=3)
        return {name: lst for name, lst in lists.items() if lst}

    def search(self, query: str, k: int, now: float) -> list[Hit]:
        if k <= 0 or not query or not query.strip():
            return []
        c = self.config
        scored = self.scored_lists(query, now)
        lists = {name: [i for i, _ in lst] for name, lst in scored.items()}
        if c.fusion == "rrf":
            fused = _fuse(lists, c.weights, c.rrf_k, scores=True)
        else:
            fused = _fuse_scores(scored, c.weights, c.rrf_k, c.keyword_power)
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
            roles = {}
            if c.context_same_role:
                allj = [i for i, _ in seeds] + [j for v in nb.values() for j in v]
                roles = {m.id: m.role for m in self.store.get_messages(allj).values()}
            for i, sc in seeds:
                for j in nb.get(i, []):
                    if c.context_same_role and roles.get(j) != roles.get(i):
                        continue
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
            if c.length_penalty:
                n = len(msgs[i].content)
                if n > c.length_pivot:
                    score *= (c.length_pivot / n) ** c.length_penalty
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
    "location": {"live", "lives", "location", "city", "based", "moved"},
    "job": {"job", "work", "occupation", "profession", "career", "living"},
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


def _fuse_scores(scored: dict[str, list], weights: dict, rrf_k: int,
                 keyword_power: float = 2.0) -> list[tuple[int, float]]:
    """Score fusion (see ``RetrievalConfig.fusion``), best first.

    - keyword: ``(bm25 / best bm25) ** keyword_power``, so the query's top
      keyword hit is worth 1 and weaker ones fall off quickly. BM25 has no
      fixed scale, hence the per-query normalisation.
    - vector: the cosine itself, clipped to [0, 1]. It already has a fixed
      scale, so a weak semantic match stays weak even when it is the best.
    - anything else (rank only): ``(rrf_k + 1) / (rrf_k + rank + 1)``.

    Each is multiplied by its weight; the sum is divided by ``rrf_k + 1`` so
    scores sit on RRF's scale and the importance/recency priors weigh the
    same under either fusion.
    """
    acc: dict[int, float] = {}
    for name, lst in scored.items():
        w = weights.get(name, 1.0)
        if not lst or not w:
            continue
        if name == "keyword" and lst[0][1] is not None:
            top = max(lst[0][1], 1e-9)
            vals = [(i, w * min(1.0, max(0.0, s) / top) ** keyword_power) for i, s in lst]
        elif name == "vector" and lst[0][1] is not None:
            vals = [(i, w * min(1.0, max(0.0, s))) for i, s in lst]
        else:
            vals = [(i, w * (rrf_k + 1) / (rrf_k + r + 1)) for r, (i, _) in enumerate(lst)]
        for i, v in vals:
            if v == v:  # NaN never enters a score
                acc[i] = acc.get(i, 0.0) + v
    scale = 1.0 / (rrf_k + 1)
    return sorted(((i, v * scale) for i, v in acc.items()), key=lambda t: (-t[1], -t[0]))
