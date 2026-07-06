"""The NoMem engine.

A framework-neutral, token-budgeted memory with three tiers:

- **core**    fixed persona/rules (+ durable user facts, arriving in M3),
- **working** recent conversation, budget-enforced sliding window,
- **archival**every message persisted, relevant ones retrieved on demand.

Every context assembly produces a full, inspectable decision log. Nothing
about what the agent "remembers" is hidden.

Example
-------
>>> mem = NoMem(max_tokens=200, core="You are a travel assistant.")
>>> mem.add_user("I'm going to Tokyo in March, budget $2000.")
>>> mem.add_assistant("Great, cherry blossoms are late March.")
>>> result = mem.build_context(query="what's my budget?")
>>> print(result.text)          # fits max_tokens, guaranteed
>>> print(result.explain())     # why each piece is (or isn't) included
"""

from __future__ import annotations

from typing import Optional, Union

from .models import ContextResult, LogEntry, Message
from .store import SQLiteStore
from .tokens import TokenCounter, default_counter


class NoMem:
    """Token-budgeted, transparent memory for one ``user_id`` + ``session_id``.

    Args:
        max_tokens: Hard cap on the assembled context. Never exceeded.
        core: Fixed persona/rules, always included first.
        db: SQLite path, ``":memory:"``, or a shared ``SQLiteStore``.
        counter: How tokens are counted (pluggable). Defaults to the best
            available local counter.
        retrieval_top_k: How many archival matches to consider per query.
        archival_ratio: Fraction of the post-core budget reserved for
            retrieved memories when a ``query`` is given (the rest goes to
            recent conversation; unused reservation rolls over to it).
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        core: str = "",
        *,
        user_id: str = "default",
        session_id: str = "default",
        db: Union[str, SQLiteStore] = ":memory:",
        counter: Optional[TokenCounter] = None,
        retrieval_top_k: int = 5,
        archival_ratio: float = 0.3,
    ):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if not 0.0 <= archival_ratio <= 1.0:
            raise ValueError(f"archival_ratio must be within [0.0, 1.0], got {archival_ratio}")
        if retrieval_top_k < 0:
            raise ValueError(f"retrieval_top_k must be >= 0, got {retrieval_top_k}")
        self.max_tokens = max_tokens
        self.core = core
        self.user_id = user_id
        self.session_id = session_id
        self.counter = counter or default_counter()
        self.retrieval_top_k = retrieval_top_k
        self.archival_ratio = archival_ratio
        self._owns_store = not isinstance(db, SQLiteStore)
        self.store = db if isinstance(db, SQLiteStore) else SQLiteStore(db)
        self._log: list[LogEntry] = []

    # -- writing ---------------------------------------------------------

    def add(self, role: str, content: str, *, metadata: Optional[dict] = None) -> Message:
        """Persist a message to the working/archival tiers."""
        tokens = self.counter.count_message(role, content)
        return self.store.add_message(
            self.user_id, self.session_id, role, content, tokens, metadata
        )

    def add_user(self, content: str, **kwargs) -> Message:
        return self.add("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs) -> Message:
        return self.add("assistant", content, **kwargs)

    # -- reading ---------------------------------------------------------

    def build_context(self, query: Optional[str] = None) -> ContextResult:
        """Assemble a prompt guaranteed to fit ``max_tokens``.

        Priority order (all share one budget, highest first):
        ``core`` -> retrieved ``archival`` memories (only when ``query`` is
        given) -> recent ``working`` conversation. Lower-priority items are
        included only if they fit, and every decision is logged.
        """
        log: list[LogEntry] = []
        messages: list[dict] = []
        budget = self.max_tokens
        used = 0

        # 1. Core memory — always included, even if it alone blows the budget
        #    (we surface that rather than silently dropping the persona).
        if self.core:
            core_tokens = self.counter.count_message("system", self.core)
            used += core_tokens
            budget -= core_tokens
            messages.append({"role": "system", "content": self.core})
            if budget < 0:
                log.append(LogEntry("budget_exhausted", "core",
                    f"core memory ({core_tokens} tok) alone exceeds max_tokens ({self.max_tokens})",
                    core_tokens, budget))
            else:
                log.append(LogEntry("core", "core", "included fixed core memory", core_tokens, budget))

        # 2. Archival retrieval — pull older, relevant messages back into
        #    context, up to a reserved slice of the remaining budget.
        retrieved: list[Message] = []
        retrieved_ids: set[int] = set()
        if query and self.retrieval_top_k > 0 and self.archival_ratio > 0 and budget > 0:
            cap = int(budget * self.archival_ratio)
            if cap > 0:
                # Retrieval is for what the window has lost: anything that will
                # already appear as a recent turn must stay a recent turn, not
                # get relocated into the retrieved block.
                window_ids = {m.id for m in self._fill_working(budget - cap, set(), [])}
                retrieved, retrieved_ids, r_used = self._retrieve(query, cap, window_ids, log)
                used += r_used
                budget -= r_used
        if retrieved:
            messages.append(self._retrieved_block(retrieved))

        # 3. Working memory — recent messages fill the remaining budget,
        #    newest first, skipping anything retrieval already surfaced.
        kept = self._fill_working(budget, retrieved_ids, log)
        for m in kept:
            used += m.tokens if m.tokens is not None else self.counter.count_message(m.role, m.content)
        messages.extend(m.as_dict() for m in kept)

        text = self._render(messages)
        self._log = log
        return ContextResult(
            text=text,
            messages=messages,
            tokens_used=used,
            max_tokens=self.max_tokens,
            log=log,
        )

    # -- assembly helpers ------------------------------------------------

    def _retrieve(self, query: str, cap: int, exclude: set[int], log: list[LogEntry]):
        """Greedily fill up to ``cap`` tokens with the best archival matches."""
        if not getattr(self.store, "fts_enabled", True):
            log.append(LogEntry("skipped", "archival",
                "retrieval unavailable (SQLite built without FTS5)", 0, cap))
            return [], set(), 0
        candidates = self.store.search(self.user_id, self.session_id, query, self.retrieval_top_k)
        included: list[Message] = []
        used = 0
        for m in candidates:
            if m.id in exclude:
                log.append(LogEntry("skipped", "archival",
                    f"{m.role} message #{m.id} (already in the recent window)", 0, cap - used, m.id))
                continue
            cost = self.counter.count_message("system", self._retrieved_block(included + [m])["content"])
            if cost <= cap:
                log.append(LogEntry("retrieved", "archival",
                    f"{m.role} message #{m.id} (relevance match)", cost - used, cap - cost, m.id))
                included.append(m)
                used = cost
            else:
                log.append(LogEntry("dropped", "archival",
                    f"{m.role} message #{m.id} (would exceed {cap}-tok archival budget)", cost - used, cap - used, m.id))
        log.append(LogEntry("retrieval", "archival",
            f"retrieved {len(included)} of {len(candidates)} match(es) for query", used, cap - used))
        return included, {m.id for m in included}, used

    def _fill_working(self, budget: int, skip_ids: set[int], log: list[LogEntry]) -> list[Message]:
        """Keep the most recent messages that fit; log the rest."""
        kept: list[Message] = []
        dropped = 0
        for m in self.store.recent_messages(self.user_id, self.session_id):
            if m.id in skip_ids:
                continue  # already surfaced by retrieval
            mt = m.tokens if m.tokens is not None else self.counter.count_message(m.role, m.content)
            if dropped == 0 and mt <= budget:
                kept.append(m)
                budget -= mt
                log.append(LogEntry("added", "working", f"{m.role} message #{m.id}", mt, budget, m.id))
            else:
                dropped += 1
                log.append(LogEntry("dropped", "working",
                    f"{m.role} message #{m.id} (needs {mt}, {budget} left)", mt, budget, m.id))
        if dropped:
            log.append(LogEntry("budget_exhausted", "working",
                f"{dropped} older message(s) dropped to fit the budget", 0, budget))
        kept.reverse()  # chronological order for the final prompt
        return kept

    @staticmethod
    def _retrieved_block(items: list[Message]) -> dict:
        """Render retrieved memories as a single labelled system message."""
        lines = ["Relevant earlier context:"] + [f"- {m.role}: {m.content}" for m in items]
        return {"role": "system", "content": "\n".join(lines)}

    @staticmethod
    def _render(messages: list[dict]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    # -- introspection ---------------------------------------------------

    @property
    def log(self) -> list[LogEntry]:
        """Decisions from the most recent :meth:`build_context` call."""
        return self._log

    def message_count(self) -> int:
        return self.store.count_messages(self.user_id, self.session_id)

    def close(self) -> None:
        """Close the store if this instance created it.

        A ``SQLiteStore`` passed in via ``db`` may be shared with other
        instances, so its lifetime stays with whoever constructed it.
        """
        if self._owns_store:
            self.store.close()
