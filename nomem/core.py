"""The NoMem engine.

A framework-neutral, token-budgeted memory with three tiers:

- **core**    fixed persona/rules (+ durable user facts, arriving in M3),
- **working** recent conversation, budget-enforced sliding window,
- **archival**every message persisted, retrieved on demand (wired in M2).

Every context assembly produces a full, inspectable decision log. Nothing
about what the agent "remembers" is hidden.
"""

from __future__ import annotations

from typing import Optional, Union

from .models import ContextResult, LogEntry, Message
from .store import SQLiteStore
from .tokens import TokenCounter, default_counter


class NoMem:
    def __init__(
        self,
        max_tokens: int = 3000,
        core: str = "",
        *,
        user_id: str = "default",
        session_id: str = "default",
        db: Union[str, SQLiteStore] = ":memory:",
        counter: Optional[TokenCounter] = None,
    ):
        self.max_tokens = max_tokens
        self.core = core
        self.user_id = user_id
        self.session_id = session_id
        self.counter = counter or default_counter()
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
        """Assemble a prompt that is guaranteed to fit ``max_tokens``.

        Assembly order (highest priority first): core -> recent working
        messages, newest kept first. ``query`` is accepted now and drives
        archival retrieval once M2 lands.
        """
        log: list[LogEntry] = []
        budget = self.max_tokens
        used = 0

        # 1. Core memory — always included, even if it alone blows the budget
        #    (we surface that rather than silently dropping the persona).
        if self.core:
            core_tokens = self.counter.count_message("system", self.core)
            used += core_tokens
            budget -= core_tokens
            if budget < 0:
                log.append(
                    LogEntry(
                        "budget_exhausted",
                        "core",
                        f"core memory ({core_tokens} tok) alone exceeds max_tokens ({self.max_tokens})",
                        core_tokens,
                        budget,
                    )
                )
            else:
                log.append(
                    LogEntry("core", "core", "included fixed core memory", core_tokens, budget)
                )

        # 2. Working memory — walk newest -> oldest, keep what fits.
        kept: list[Message] = []
        dropped = 0
        for m in self.store.recent_messages(self.user_id, self.session_id):
            mt = m.tokens if m.tokens is not None else self.counter.count_message(m.role, m.content)
            if dropped == 0 and mt <= budget:
                kept.append(m)
                used += mt
                budget -= mt
                log.append(
                    LogEntry("added", "working", f"{m.role} message #{m.id}", mt, budget, m.id)
                )
            else:
                dropped += 1
                log.append(
                    LogEntry(
                        "dropped",
                        "working",
                        f"{m.role} message #{m.id} (needs {mt}, {budget} left)",
                        mt,
                        budget,
                        m.id,
                    )
                )
        if dropped:
            log.append(
                LogEntry(
                    "budget_exhausted",
                    "working",
                    f"{dropped} older message(s) dropped to fit the budget",
                    0,
                    budget,
                )
            )

        kept.reverse()  # chronological order for the final prompt

        # 3. Assemble output.
        messages: list[dict] = []
        if self.core:
            messages.append({"role": "system", "content": self.core})
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
        self.store.close()
