"""Compaction: fold turns that fell out of the window into a short summary.

NoMem's compactors are *extractive*: every line of a summary is a sentence
the user or assistant actually wrote, quoted with its date. Nothing is
paraphrased, so nothing can be invented, and the transparency log can point
at the source of every line. There is no LLM compactor by design.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Optional

from . import annotate as A
from .models import Message


class Compactor(ABC):
    @abstractmethod
    def compact(
        self,
        messages: list[Message],
        previous_summary: str = "",
        *,
        max_tokens: Optional[int] = None,
        counter=None,
    ) -> str:
        """Fold ``messages`` (oldest first) into a summary of at most ``max_tokens``."""
        ...


class NullCompactor(Compactor):
    """No-op: overflow is dropped without a summary (a pure sliding window)."""

    def compact(self, messages, previous_summary="", *, max_tokens=None, counter=None) -> str:
        return previous_summary


class ExtractiveCompactor(Compactor):
    """Pick the most informative sentences that fit the budget.

    Each sentence is scored by its rule-based importance (facts, names,
    numbers, cue words) and its centrality (how much of the dropped stretch's
    vocabulary it shares). Selection is greedy with a Maximal Marginal
    Relevance penalty so near-repeats don't crowd out other topics. Chosen
    sentences are emitted in their original order.

    Args:
        importance_weight: Share of the score from importance vs centrality.
        redundancy: MMR penalty for overlap with already chosen sentences.
        min_importance: Sentences scoring below this (pure filler) are skipped.
    """

    def __init__(self, importance_weight: float = 0.6, redundancy: float = 0.5,
                 min_importance: float = 0.2):
        self.importance_weight = importance_weight
        self.redundancy = redundancy
        self.min_importance = min_importance

    def compact(self, messages, previous_summary="", *, max_tokens=None, counter=None) -> str:
        cands = []  # (order, text_line, terms, importance)
        for m in messages:
            for j, sent in enumerate(A.sentences(m.content)):
                imp = A.importance(sent, A.extract_slots(sent), A.entities(sent))
                if imp < self.min_importance:
                    continue
                terms = set(A.content_terms(sent))
                if not terms:
                    continue
                line = f"- [{_date(m.created_at)}] {m.role}: {sent.strip()}"
                cands.append(((m.id or 0, j), line, terms, imp))
        if not cands:
            return previous_summary

        df = Counter(t for c in cands for t in c[2])
        n = len(cands)
        def centrality(terms):
            return sum(math.log(1 + df[t]) for t in terms) / (1 + math.log(1 + n) * len(terms))
        cent = [centrality(c[2]) for c in cands]
        top = max(cent) or 1.0
        base = [self.importance_weight * c[3] + (1 - self.importance_weight) * (ce / top)
                for c, ce in zip(cands, cent)]

        head = "Earlier conversation (key sentences):"
        fits = (lambda text: True) if max_tokens is None or counter is None else \
            (lambda text: counter.count_message("system", text) <= max_tokens)
        order = lambda idx: sorted(idx, key=lambda i: cands[i][0])
        chosen: list[int] = []
        maxov = [0.0] * n
        remaining = set(range(n))
        while remaining:
            best = max(remaining, key=lambda i: (base[i] - self.redundancy * maxov[i], -cands[i][0][0]))
            remaining.discard(best)
            if base[best] - self.redundancy * maxov[best] <= 0:
                break
            trial = chosen + [best]
            if not fits(_render(head, previous_summary, [cands[i] for i in order(trial)])):
                continue
            chosen = trial
            for i in remaining:
                maxov[i] = max(maxov[i], _jaccard(cands[i][2], cands[best][2]))
        if not chosen:
            return previous_summary
        return _render(head, previous_summary, [cands[i] for i in order(chosen)])


def _render(head: str, previous: str, items) -> str:
    lines = [head]
    if previous:
        lines.append(previous)
    lines += [c[1] for c in items]
    return "\n".join(lines)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d %b %Y").lstrip("0")
