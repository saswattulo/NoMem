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
import warnings
from datetime import datetime
from typing import Optional, Union

from . import annotate as A
from .compactor import Compactor, ExtractiveCompactor
from .conflicts import (ConflictConfig, detect as detect_conflicts, slot_conflict_confidence,
                        snippet, speaker_key)
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
        archival_max_ratio: How far retrieval may stretch past
            ``archival_ratio`` when a matching memory doesn't fit the reserved
            slice: up to this fraction of the post-core budget, taken from the
            recent window. ``max_tokens`` is never exceeded either way. Set it
            equal to ``archival_ratio`` (or lower) for a fixed slice.
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
        conflicts: Contradiction detection on write (see
            :class:`~nomem.conflicts.ConflictConfig`). When a newer memory
            changes an older one, both are kept and linked, and retrieved
            memories carry a ``[CONFLICT: ...]`` marker quoting the other
            side with its date. Requires ``annotate``.
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
        archival_max_ratio: float = 0.8,
        retrieval: Optional[RetrievalConfig] = None,
        embedder=None,
        annotate: bool = True,
        facts_ratio: float = 0.1,
        summary_ratio: float = 0.0,
        compactor: Optional[Compactor] = None,
        conflicts: Optional[ConflictConfig] = None,
    ):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if not 0.0 <= summary_ratio <= 1.0:
            raise ValueError(f"summary_ratio must be within [0.0, 1.0], got {summary_ratio}")
        if not 0.0 <= facts_ratio <= 1.0:
            raise ValueError(f"facts_ratio must be within [0.0, 1.0], got {facts_ratio}")
        if not 0.0 <= archival_ratio <= 1.0:
            raise ValueError(f"archival_ratio must be within [0.0, 1.0], got {archival_ratio}")
        if not 0.0 <= archival_max_ratio <= 1.0:
            raise ValueError(f"archival_max_ratio must be within [0.0, 1.0], got {archival_max_ratio}")
        if retrieval_top_k < 0:
            raise ValueError(f"retrieval_top_k must be >= 0, got {retrieval_top_k}")
        self.max_tokens = max_tokens
        self.core = core
        self.user_id = user_id
        self.session_id = session_id
        self.counter = counter or default_counter()
        self.retrieval_top_k = retrieval_top_k
        self.archival_ratio = archival_ratio
        self.archival_max_ratio = archival_max_ratio
        self._owns_store = not isinstance(db, SQLiteStore)
        self.store = db if isinstance(db, SQLiteStore) else SQLiteStore(db)
        self._log: list[LogEntry] = []
        self.annotate = annotate
        self.facts_ratio = facts_ratio
        self.summary_ratio = summary_ratio
        self.summary_lookback = 200  # turns before the window the compactor may quote
        self.compactor = compactor if compactor is not None else ExtractiveCompactor()
        self.embedder = embedder
        self.conflict_config = conflicts or ConflictConfig()
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
        if self.embedder is not None:
            try:
                self.embedder.index(self.store, msg, self.user_id, self.session_id)
            except Exception as e:  # the message must not be lost to a failing model
                warnings.warn(f"embedder failed, message {msg.id} is stored without a vector "
                              f"(keyword search still finds it): {e!r}", RuntimeWarning, stacklevel=2)
        if self.annotate:
            speaker = (metadata or {}).get("speaker") or A.speaker_of(content)
            ann = A.annotate(content, msg.created_at, subject=speaker or "user")
            self.store.add_annotation(msg, self.user_id, self.session_id, ann, commit=False)
            # Facts about a person come from what *they* say: assistant turns
            # never write slots unless a speaker is named.
            linked: set[int] = set()
            if role == "user" or speaker:
                linked = self._apply_slots(ann.slots, msg)
            cc = self.conflict_config
            if cc.enabled:
                for c in detect_conflicts(self.store, self.user_id, self.session_id, msg,
                                          speaker_key(role, content, metadata), cc, self.embedder):
                    if c.older_id not in linked:
                        self.store.add_conflict(self.user_id, c.older_id, c.newer_id, c.kind,
                                                c.detail, c.confidence, c.attribute)
        self.store.commit()
        return msg

    def _apply_slots(self, slots: list, msg: Message) -> set[int]:
        """Record slot facts; a new single value supersedes the old one.

        Nothing is deleted: a superseded slot gets ``valid_to`` and a pointer
        to the message that closed it, so history stays queryable, and the
        two messages are linked as a conflict so both can be shown.
        Returns the ids of the earlier messages linked this way.
        """
        st, uid, ts = self.store, self.user_id, msg.created_at
        opposite = {"likes": "dislikes", "dislikes": "likes"}
        linked: set[int] = set()
        record = self.annotate and self.conflict_config.enabled

        def close(row: dict, detail: str, new_value: Optional[str] = None) -> None:
            st.close_slot(row["id"], ts, msg.id, commit=False)
            src = row["source_message_id"]
            conf = slot_conflict_confidence(row["attribute"], row["value"], new_value)
            if record and src and src != msg.id and conf >= self.conflict_config.min_confidence:
                st.add_conflict(uid, src, msg.id, "slot", detail, conf, row["attribute"])
                linked.add(src)

        for sl in slots:
            attr = sl.attribute.replace("_", " ")
            if sl.negated:
                for row in st.current_slots(uid, sl.subject, sl.attribute):
                    if row["value"].lower() == sl.value.lower():
                        close(row, f"{attr}: {row['value']} -> no longer")
                continue
            current = st.current_slots(uid, sl.subject, sl.attribute)
            if any(r["value"].lower() == sl.value.lower() for r in current):
                continue
            if not sl.multi:
                for row in current:
                    close(row, f"{attr}: {row['value']} -> {sl.value}", sl.value)
            if sl.attribute in opposite:
                for row in st.current_slots(uid, sl.subject, opposite[sl.attribute]):
                    if row["value"].lower() == sl.value.lower():
                        close(row, f"{row['attribute']} -> {sl.attribute}: {sl.value}")
            st.add_slot(uid, sl.subject, sl.attribute, sl.value, sl.multi, ts, msg.id, commit=False)
        return linked

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

    def conflicts(self, message_id: Optional[int] = None) -> list[dict]:
        """Pairs of memories where a newer one may contradict an older one.

        Each dict has ``older_id``/``newer_id``, ``older``/``newer`` (the
        verbatim messages), ``older_at``/``newer_at`` (epoch seconds),
        ``kind`` (``slot``, ``frame`` or ``update``), ``detail`` and
        ``confidence``. Nothing is resolved: both memories stay as written.
        """
        rows = self.store.conflicts(self.user_id, None if message_id is None else [message_id])
        msgs = self.store.get_messages([r["older_id"] for r in rows] + [r["newer_id"] for r in rows])
        out = []
        for r in rows:
            o, n = msgs.get(r["older_id"]), msgs.get(r["newer_id"])
            if o is None or n is None:
                continue
            out.append(r | {"older": o.content, "newer": n.content,
                            "older_at": o.created_at, "newer_at": n.created_at})
        return out

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
            flex = max(cap, int(budget * self.archival_max_ratio))
            if cap > 0:
                retrieved, retrieved_ids, r_used, r_reserved = self._resolve_archival(
                    query, recent, budget, cap, flex, log
                )
                used += r_used
                budget -= r_reserved
        if retrieved:
            messages.append(self._retrieved_block(retrieved))
            for mid, note in self._conflict_notes(retrieved).items():
                log.append(LogEntry("conflict", "archival",
                    f"message #{mid} flagged:{note}", 0, None, mid))

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
        flex: int,
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

        ``cap`` is the reserved slice and ``flex`` the most retrieval may
        stretch to for a match that doesn't fit it (see ``archival_max_ratio``).

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

        # Turns the window shows when only the fixed slice is reserved.
        # Stretching for one of those would just move it out of the window.
        shown = self._window_ids(recent, budget - cap) if flex > cap else set()
        spend = flex
        for _ in range(self.retrieval_top_k + 2):
            window = self._window_ids(recent, budget - spend)
            settled = self._retrieve(candidates, cap, flex, window, [], shown)[2]
            if settled == spend:
                break
            spend = settled
        else:
            # Didn't settle: passing over a bulky match can let two smaller
            # ones in, so the spend can oscillate rather than converge. Fall
            # back to reserving the whole slice — the window then cannot
            # outgrow the exclusion set, at the cost of any unused rollover.
            spend = flex

        # Replay the settled pass, this time onto the real log.
        window = self._window_ids(recent, budget - spend)
        retrieved, ids, used = self._retrieve(candidates, cap, flex, window, log, shown)
        return retrieved, ids, used, spend

    def _retrieve(
        self,
        candidates: list[Message],
        cap: int,
        flex: int,
        exclude: set[int],
        log: list[LogEntry],
        shown: Union[set[int], frozenset[int]] = frozenset(),
    ):
        """Greedily fill up to ``cap`` tokens with the best archival matches.

        Two passes. The first fills the reserved ``cap`` in rank order,
        passing over matches that don't fit, exactly as a fixed slice would.
        The second goes back to the matches it passed over, best first, and
        takes each that fits ``flex``: the slice stretches by what that match
        needs and the recent window gives the tokens up. So stretching only
        ever adds to what the fixed slice holds, never displaces it.

        Stretching costs recent turns, so it is kept for direct matches:
        never for one found only as the neighbour of another match
        (``context`` ranker), and never for one in ``shown`` (a turn the
        window holds with only the fixed slice reserved), which would just
        move it out of the window.
        """
        why = getattr(self, "_why", {})
        rank = {m.id: i for i, m in enumerate(candidates)}
        included: list[Message] = []
        passed: list[Message] = []
        used = 0

        def cost_with(m: Message) -> int:
            items = sorted(included + [m], key=lambda x: rank[x.id])
            return self.counter.count_message("system", self._retrieved_block(items)["content"])

        for m in candidates:
            if m.id in exclude:
                log.append(LogEntry("skipped", "archival",
                    f"{m.role} message #{m.id} (already in the recent window)", 0, cap - used, m.id))
                continue
            cost = cost_with(m)
            if cost <= cap:
                log.append(LogEntry("retrieved", "archival",
                    f"{m.role} message #{m.id} (matched by {why.get(m.id) or 'relevance'})",
                    cost - used, cap - cost, m.id))
                included.append(m)
                used = cost
            else:
                passed.append(m)
        for m in passed:
            if m.id in shown:
                log.append(LogEntry("skipped", "archival",
                    f"{m.role} message #{m.id} (no stretch: a recent turn the fixed slice leaves in the window)",
                    0, max(cap, used) - used, m.id))
                continue
            found_by = [x for x in why.get(m.id, "").split(", ") if x]
            strong = any(not x.startswith(("context#", "repeats ")) for x in found_by) or not found_by
            cost = cost_with(m) if flex > cap and strong else flex + 1
            if cost <= flex:
                log.append(LogEntry("retrieved", "archival",
                    f"{m.role} message #{m.id} (matched by {why.get(m.id) or 'relevance'}; "
                    f"slice stretched {max(cap, used)} -> {cost} tok, taken from the recent window)",
                    cost - used, flex - cost, m.id))
                included.append(m)
                used = cost
            else:
                limit = f"the {flex}-tok archival limit" if flex > cap and strong else f"{cap}-tok archival budget"
                log.append(LogEntry("dropped", "archival",
                    f"{m.role} message #{m.id} (would exceed {limit})",
                    max(0, cost_with(m) - used), max(cap, used) - used, m.id))
        included.sort(key=lambda x: rank[x.id])
        log.append(LogEntry("retrieval", "archival",
            f"retrieved {len(included)} of {len(candidates)} match(es) for query", used, max(cap, used) - used))
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
        previous = self._previous_values(relevant + rest)
        for r in relevant + rest:
            who = "" if r["subject"] == "user" else f"{r['subject']} "
            was = previous.get(r["id"])
            was = f"; before that {was[0]}, said {_date(was[1])}" if was else ""
            lines.append(f"- {who}{r['attribute'].replace('_', ' ')}: {r['value']} "
                         f"(since {_date(r['valid_from'])}{was})")
        if not lines:
            log.append(LogEntry("skipped", "core", "state card: no stored fact matches the query", 0, cap))
            return None, 0
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

    def _previous_values(self, rows: list[dict]) -> dict[int, tuple[str, float]]:
        """For each current slot row, the value it replaced, if it is a real update."""
        cc = self.conflict_config
        if not rows or not cc.enabled or cc.show_per_memory <= 0:
            return {}
        by_closer: dict[tuple, dict] = {}
        for h in self.store.slot_history(self.user_id):
            if h["closed_by"]:
                by_closer[(h["closed_by"], h["subject"], h["attribute"])] = h
        out: dict[int, tuple[str, float]] = {}
        for r in rows:
            h = by_closer.get((r["source_message_id"], r["subject"], r["attribute"]))
            if h and not r["multi"] and slot_conflict_confidence(
                    r["attribute"], h["value"], r["value"]) >= cc.show_min_confidence:
                out[r["id"]] = (h["value"], h["valid_from"])
        return out

    def _retrieved_block(self, items: list[Message]) -> dict:
        """Render retrieved memories as a single labelled system message.

        A memory linked to a contradicting one carries a ``[CONFLICT: ...]``
        marker that quotes the other side with its date, so the agent sees
        both versions and decides, instead of meeting a silent contradiction.
        """
        notes = self._conflict_notes(items) if items else {}
        lines = ["Relevant earlier context:"] + [
            f"- [{_date(m.created_at)}] {m.role}: {m.content}{notes.get(m.id, '')}" for m in items
        ]
        return {"role": "system", "content": "\n".join(lines)}

    def _conflict_notes(self, items: list[Message]) -> dict[int, str]:
        cc = self.conflict_config
        if not cc.enabled or cc.show_per_memory <= 0:
            return {}
        ids = {m.id for m in items}
        rows = [r for r in self.store.conflicts(self.user_id, list(ids)) if r["confidence"] >= cc.show_min_confidence]
        if not rows:
            return {}
        need = [i for r in rows for i in (r["older_id"], r["newer_id"]) if i not in ids]
        by_id = {m.id: m for m in items} | self.store.get_messages(need)
        notes: dict[int, list[str]] = {}
        for r in sorted(rows, key=lambda r: -r["confidence"]):
            o, n = by_id.get(r["older_id"]), by_id.get(r["newer_id"])
            if o is None or n is None:
                continue
            for me, other, newer in ((o, n, False), (n, o, True)):
                if me.id not in ids or len(notes.get(me.id, [])) >= cc.show_per_memory:
                    continue
                verb = f"updates {_date(other.created_at)}" if newer else f"changed on {_date(other.created_at)}"
                quote = (f"see the {_date(other.created_at)} memory in this list"
                         if other.id in ids else f'{other.role} said "{snippet(other.content, cc.quote_chars)}"')
                notes.setdefault(me.id, []).append(f"{verb} ({r['detail']}); {quote}")
        return {mid: " [CONFLICT: " + " | ".join(parts) + "]" for mid, parts in notes.items()}

    @staticmethod
    def _render(messages: list[dict]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    # -- forgetting ------------------------------------------------------

    def forget(self, message_ids: Union[int, list[int]]) -> int:
        """Delete stored messages of this user, and every index built from them.

        This is the one way to undo :meth:`add`: the text, its search index,
        vectors, extracted facts and conflict links all go. Facts the message
        had replaced become current again. Returns how many messages went.
        """
        ids = [message_ids] if isinstance(message_ids, int) else list(message_ids)
        if not ids:
            return 0
        n = self.store.delete_messages(self.user_id, ids)
        self._drop_vector_cache()
        return n

    def forget_session(self) -> int:
        """Delete every message of this ``user_id`` + ``session_id``. Pinned facts stay."""
        return self.forget(self.store.message_ids(self.user_id, self.session_id))

    def _drop_vector_cache(self) -> None:
        cache = getattr(self.embedder, "_mats", None)
        if isinstance(cache, dict):
            tok = getattr(self.store, "_nomem_vec_token", None)  # see Embedder._key
            for key in [k for k in cache if k[0] == tok and k[1] == self.user_id]:
                cache.pop(key, None)

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
