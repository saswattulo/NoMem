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

import time
from datetime import datetime
from typing import Optional, Union

from . import annotate as A
from .compactor import Compactor, ExtractiveCompactor
from .models import ContextResult, LogEntry, Message
from .retrieval import Hit, RetrievalConfig, Retriever, slot_matches
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
        retrieval: Tuning for the hybrid retriever (see
            :class:`~nomem.retrieval.RetrievalConfig`).
        embedder: Optional :class:`~nomem.embed.Embedder` adding semantic
            (vector) retrieval. ``None`` keeps NoMem model-free.
        annotate: Extract entities, dates, keyphrases and slot facts on
            write (deterministic rules, no model). Default ``True``.
        facts_ratio: Most of the post-core budget the "known facts" state
            card may use (current slot facts + :meth:`remember` facts,
            query-relevant first). Unused budget rolls over. ``0`` disables.
        summary_ratio: Share of the post-core budget reserved for an
            extractive summary of turns that just fell out of the window.
            Default ``0`` (off): a pure sliding window plus retrieval.
        compactor: How dropped turns are summarised. Defaults to
            :class:`~nomem.compactor.ExtractiveCompactor` (quotes, no model).
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
        retrieval: Optional[RetrievalConfig] = None,
        embedder=None,
        annotate: bool = True,
        facts_ratio: float = 0.1,
        summary_ratio: float = 0.0,
        compactor: Optional[Compactor] = None,
    ):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if not 0.0 <= summary_ratio <= 1.0:
            raise ValueError(f"summary_ratio must be within [0.0, 1.0], got {summary_ratio}")
        if not 0.0 <= facts_ratio <= 1.0:
            raise ValueError(f"facts_ratio must be within [0.0, 1.0], got {facts_ratio}")
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
        self.annotate = annotate
        self.facts_ratio = facts_ratio
        self.summary_ratio = summary_ratio
        self.summary_lookback = 200  # turns before the window the compactor may quote
        self.compactor = compactor if compactor is not None else ExtractiveCompactor()
        self.embedder = embedder
        self.retriever = Retriever(self.store, user_id, session_id, retrieval, embedder)

    # -- writing ---------------------------------------------------------

    def add(
        self,
        role: str,
        content: str,
        *,
        metadata: Optional[dict] = None,
        timestamp: Optional[float] = None,
    ) -> Message:
        """Persist a message verbatim, then index it.

        ``timestamp`` (epoch seconds) backdates the message, e.g. when
        importing an existing transcript; it anchors relative dates such as
        "yesterday" and the dates shown on retrieved memories.
        """
        tokens = self.counter.count_message(role, content)
        msg = self.store.add_message(
            self.user_id, self.session_id, role, content, tokens, metadata,
            created_at=timestamp, commit=False,
        )
        if self.annotate:
            speaker = (metadata or {}).get("speaker") or A.speaker_of(content)
            ann = A.annotate(content, msg.created_at, subject=speaker or "user")
            self.store.add_annotation(msg, self.user_id, self.session_id, ann, commit=False)
            # Facts about a person come from what *they* say: assistant turns
            # never write slots unless a speaker is named.
            if role == "user" or speaker:
                self._apply_slots(ann.slots, msg)
        if self.embedder is not None:
            self.embedder.index(self.store, msg, self.user_id, self.session_id)
        self.store.commit()
        return msg

    def _apply_slots(self, slots: list, msg: Message) -> None:
        """Record slot facts; a new single value supersedes the old one.

        Nothing is deleted: a superseded slot gets ``valid_to`` and a pointer
        to the message that closed it, so history stays queryable.
        """
        st, uid, ts = self.store, self.user_id, msg.created_at
        opposite = {"likes": "dislikes", "dislikes": "likes"}
        for sl in slots:
            if sl.negated:
                for row in st.current_slots(uid, sl.subject, sl.attribute):
                    if row["value"].lower() == sl.value.lower():
                        st.close_slot(row["id"], ts, msg.id, commit=False)
                continue
            current = st.current_slots(uid, sl.subject, sl.attribute)
            if any(r["value"].lower() == sl.value.lower() for r in current):
                continue
            if not sl.multi:
                for row in current:
                    st.close_slot(row["id"], ts, msg.id, commit=False)
            if sl.attribute in opposite:
                for row in st.current_slots(uid, sl.subject, opposite[sl.attribute]):
                    if row["value"].lower() == sl.value.lower():
                        st.close_slot(row["id"], ts, msg.id, commit=False)
            st.add_slot(uid, sl.subject, sl.attribute, sl.value, sl.multi, ts, msg.id, commit=False)

    def add_user(self, content: str, **kwargs) -> Message:
        return self.add("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs) -> Message:
        return self.add("assistant", content, **kwargs)

    # -- reading ---------------------------------------------------------

    def search(self, query: str, k: Optional[int] = None) -> list[Message]:
        """Archival search: the best ``k`` matching messages, best first."""
        return [h.message for h in self.search_hits(query, k)]

    def search_hits(self, query: str, k: Optional[int] = None) -> list[Hit]:
        """Like :meth:`search`, with scores and the rankers that found each hit."""
        k = self.retrieval_top_k if k is None else k
        if not getattr(self.store, "fts_enabled", True) and self.embedder is None:
            return []
        return self.retriever.search(query, k, time.time())

    def remember(self, key: str, value: str) -> None:
        """Pin a durable fact for this user; it joins the state card."""
        self.store.upsert_fact(self.user_id, key, value)

    def facts(self, subject: Optional[str] = None) -> list[dict]:
        """Current slot facts (``valid_to`` unset), oldest first."""
        return self.store.current_slots(self.user_id, subject)

    def fact_history(self) -> list[dict]:
        """Every slot fact ever recorded, including superseded ones."""
        return self.store.slot_history(self.user_id)

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

        # 1b. State card — current facts, query-relevant first, from slots
        #     the rules extracted and facts pinned with remember().
        if self.facts_ratio > 0 and budget > 0:
            cap = int(budget * self.facts_ratio)
            card, cost = self._state_card(query, cap, log)
            if card:
                messages.append(card)
                used += cost
                budget -= cost

        # Reserve the summary slice now so retrieval and the window plan
        # around it; it is spent after the window settles which turns drop.
        s_cap = int(budget * self.summary_ratio) if self.summary_ratio > 0 and budget > 0 else 0
        budget -= s_cap

        # One read of the session's messages serves both the retrieval
        # exclusion probe and the final window fill.
        # Only the newest messages can ever fit, so never read more than the
        # most the budget could hold (+1 to see the first one that doesn't).
        # With a summary slice, also read a stretch before the window for
        # the compactor to quote from.
        read_limit = self._window_limit() + (self.summary_lookback if s_cap else 0)
        recent = self.store.recent_messages(
            self.user_id, self.session_id, limit=read_limit
        )

        # 2. Archival retrieval — pull older, relevant messages back into
        #    context, up to a reserved slice of the remaining budget.
        retrieved: list[Message] = []
        retrieved_ids: set[int] = set()
        if query and self.retrieval_top_k > 0 and self.archival_ratio > 0 and budget > 0:
            cap = int(budget * self.archival_ratio)
            if cap > 0:
                retrieved, retrieved_ids, r_used, r_reserved = self._resolve_archival(
                    query, recent, budget, cap, log
                )
                used += r_used
                budget -= r_reserved
        if retrieved:
            messages.append(self._retrieved_block(retrieved))

        # 3. Working memory — recent messages fill the remaining budget,
        #    newest first, skipping anything retrieval already surfaced.
        kept = self._fill_working(recent, budget, retrieved_ids, log)
        if s_cap:
            summary, s_used = self._summarise(recent, kept, retrieved_ids, s_cap, log)
            if summary:
                messages.append(summary)
                used += s_used
        unread = self.message_count() - len(recent) if len(recent) == read_limit else 0
        if unread > 0:
            log.append(LogEntry("budget_exhausted", "working",
                f"{unread} older message(s) not read: they cannot fit the budget", 0, None))
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

    def _resolve_archival(
        self,
        query: str,
        recent: list[Message],
        budget: int,
        cap: int,
        log: list[LogEntry],
    ):
        """Settle retrieval against the recent window it shares a budget with.

        The two are mutually dependent: what retrieval spends shrinks the
        window, and whatever the window will already show must not be pulled
        into the retrieved block as well. So assume retrieval spends its whole
        reservation (giving the smallest window), then recompute with what it
        actually spent, until the two agree.

        Settling matters. If the window used to exclude candidates is smaller
        than the one finally rendered, a message can be retrieved *and* still
        fit as a recent turn — which leaves a hole in the middle of the
        conversation and quotes the missing turn out of order above it.

        Returns the retrieved messages, their ids, the tokens they actually
        cost, and the reservation to charge the window (equal, once settled).
        """
        if not getattr(self.store, "fts_enabled", True) and self.embedder is None:
            log.append(LogEntry("skipped", "archival",
                "retrieval unavailable (SQLite built without FTS5)", 0, cap))
            return [], set(), 0, 0

        hits = self.search_hits(query, self.retrieval_top_k)
        self._why = {h.message.id: ", ".join(h.reasons) for h in hits}
        candidates = [h.message for h in hits]

        spend = cap
        for _ in range(self.retrieval_top_k + 2):
            window = self._window_ids(recent, budget - spend)
            settled = self._retrieve(candidates, cap, window, [])[2]
            if settled == spend:
                break
            spend = settled
        else:
            # Didn't settle: passing over a bulky match can let two smaller
            # ones in, so the spend can oscillate rather than converge. Fall
            # back to reserving the whole slice — the window then cannot
            # outgrow the exclusion set, at the cost of any unused rollover.
            spend = cap

        # Replay the settled pass, this time onto the real log.
        window = self._window_ids(recent, budget - spend)
        retrieved, ids, used = self._retrieve(candidates, cap, window, log)
        return retrieved, ids, used, spend

    def _retrieve(
        self,
        candidates: list[Message],
        cap: int,
        exclude: set[int],
        log: list[LogEntry],
    ):
        """Greedily fill up to ``cap`` tokens with the best archival matches."""
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
                    f"{m.role} message #{m.id} (matched by {getattr(self, '_why', {}).get(m.id) or 'relevance'})",
                    cost - used, cap - cost, m.id))
                included.append(m)
                used = cost
            else:
                log.append(LogEntry("dropped", "archival",
                    f"{m.role} message #{m.id} (would exceed {cap}-tok archival budget)", cost - used, cap - used, m.id))
        log.append(LogEntry("retrieval", "archival",
            f"retrieved {len(included)} of {len(candidates)} match(es) for query", used, cap - used))
        return included, {m.id for m in included}, used

    def _window_limit(self) -> int:
        smallest = max(1, self.counter.count_message("user", ""))
        return self.max_tokens // smallest + 1

    def _window_ids(self, recent: list[Message], budget: int) -> set[int]:
        """Ids the recent window would hold at ``budget``.

        The same greedy walk as :meth:`_fill_working`, stopping at the first
        message that doesn't fit rather than logging every one behind it —
        this runs once per settling pass, so it stays off the long tail.
        """
        ids: set[int] = set()
        for m in recent:
            mt = m.tokens if m.tokens is not None else self.counter.count_message(m.role, m.content)
            if mt > budget:
                break
            ids.add(m.id)
            budget -= mt
        return ids

    def _fill_working(
        self,
        recent: list[Message],
        budget: int,
        skip_ids: set[int],
        log: list[LogEntry],
    ) -> list[Message]:
        """Keep the most recent messages that fit; log the rest.

        The window is a contiguous run of newest turns: once one message
        doesn't fit, every older one is dropped too, rather than cherry-picking
        small old messages out of the middle of the conversation. ``skip_ids``
        cannot punch a hole in it — :meth:`_resolve_archival` only ever
        retrieves messages that fall outside this window, so by the time the
        walk reaches one it is already past the boundary and keeping nothing.
        """
        kept: list[Message] = []
        dropped = 0
        for m in recent:
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

    def _summarise(self, recent, kept, retrieved_ids, cap: int, log: list[LogEntry]):
        """Extractive summary of the turns just before the window."""
        kept_ids = {m.id for m in kept}
        dropped = [m for m in reversed(recent) if m.id not in kept_ids and m.id not in retrieved_ids]
        if not dropped:
            log.append(LogEntry("skipped", "working", "no dropped turns to summarise", 0, cap))
            return None, 0
        text = self.compactor.compact(dropped, max_tokens=cap, counter=self.counter)
        cost = self.counter.count_message("system", text) if text else 0
        if not text or cost > cap:
            log.append(LogEntry("skipped", "working",
                f"nothing from {len(dropped)} dropped turn(s) fit the {cap}-tok summary slice", 0, cap))
            return None, 0
        n_lines = text.count("\n")
        log.append(LogEntry("summarized", "working",
            f"{n_lines} key sentence(s) quoted from {len(dropped)} dropped turn(s)", cost, cap - cost))
        return {"role": "system", "content": text}, cost

    def _state_card(self, query: Optional[str], cap: int, log: list[LogEntry]):
        """Greedy-fill a "Known facts" block within ``cap`` tokens."""
        pinned = self.store.get_facts(self.user_id)
        current = self.store.current_slots(self.user_id)
        if not pinned and not current:
            return None, 0
        relevant = [r for r in slot_matches(self.store, self.user_id, query, False)] if query else []
        # With a query, only facts it asks about earn budget; without one the
        # card is a general profile, newest first.
        rest = [] if query else sorted(current, key=lambda r: -r["valid_from"])
        lines: list[str] = [f"- {k}: {v}" for k, v in pinned.items()]
        for r in relevant + rest:
            who = "" if r["subject"] == "user" else f"{r['subject']} "
            lines.append(f"- {who}{r['attribute'].replace('_', ' ')}: {r['value']} "
                         f"(since {_date(r['valid_from'])})")
        head = "Known facts:"
        kept: list[str] = []
        cost = 0
        for line in lines:
            trial = self.counter.count_message("system", "\n".join([head] + kept + [line]))
            if trial <= cap:
                kept.append(line)
                cost = trial
        if not kept:
            log.append(LogEntry("dropped", "core", f"state card ({len(lines)} facts) does not fit {cap} tokens", 0, cap))
            return None, 0
        log.append(LogEntry("facts", "core",
            f"state card: {len(kept)} of {len(lines)} fact(s), {len(relevant)} matched the query", cost, cap - cost))
        return {"role": "system", "content": "\n".join([head] + kept)}, cost

    def _retrieved_block(self, items: list[Message]) -> dict:
        """Render retrieved memories as a single labelled system message.

        A memory whose fact was later superseded says so, with the newer
        value, so the agent sees the change instead of a silent conflict.
        """
        notes = self._superseded_notes(items) if items else {}
        lines = ["Relevant earlier context:"] + [
            f"- [{_date(m.created_at)}] {m.role}: {m.content}{notes.get(m.id, '')}" for m in items
        ]
        return {"role": "system", "content": "\n".join(lines)}

    def _superseded_notes(self, items: list[Message]) -> dict[int, str]:
        src = self.store.slots_from_messages(self.user_id, [m.id for m in items])
        notes: dict[int, str] = {}
        for mid, rows in src.items():
            parts = []
            for r in rows:
                if r["valid_to"] is None or not r["closed_by"]:
                    continue
                newer = [n for n in self.store.slots_from_messages(self.user_id, [r["closed_by"]]).get(r["closed_by"], [])
                         if n["attribute"] == r["attribute"]]
                now = f"; now {newer[0]['value']}" if newer else ""
                parts.append(f"{r['attribute'].replace('_', ' ')} changed on {_date(r['valid_to'])}{now}")
            if parts:
                notes[mid] = " (superseded: " + "; ".join(parts) + ")"
        return notes

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


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d %b %Y").lstrip("0")
