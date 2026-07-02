"""Token counting.

The hard token budget is NoMem's headline feature, so counting is a first
class, pluggable concern. Adapters can inject an exact per-model counter; the
core ships a fast local default.

- :class:`OpenAICounter` — exact and local (via ``tiktoken``).
- :class:`AnthropicCounter` — exact via Anthropic's ``count_tokens`` API.
- :class:`ApproxCounter` — zero-dependency heuristic, deterministic, ~15% off.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod


class TokenCounter(ABC):
    """Counts tokens for a piece of text or a chat message.

    ``per_message_overhead`` approximates the per-message formatting cost
    (role markers, separators) that a provider adds on top of the raw content.
    """

    per_message_overhead: int = 4

    @abstractmethod
    def count(self, text: str) -> int:
        ...

    def count_message(self, role: str, content: str) -> int:
        return self.count(f"{role}: {content}") + self.per_message_overhead


class ApproxCounter(TokenCounter):
    """Character-based heuristic. No dependencies, fully deterministic.

    Roughly 4 characters per token for English. Good enough to start; swap in
    an exact counter for anything you actually ship.
    """

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / self.chars_per_token))


class OpenAICounter(TokenCounter):
    """Exact, local token counts for OpenAI-family models via ``tiktoken``."""

    def __init__(self, model: str = "gpt-4o", encoding: str = "cl100k_base"):
        import tiktoken  # imported lazily so tiktoken stays optional

        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class AnthropicCounter(TokenCounter):
    """Exact token counts for Claude models via the Anthropic API.

    Requires the ``anthropic`` SDK and an API key. This makes a network call,
    so prefer it in adapters/offline batch paths rather than a hot loop.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", client=None):
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily so anthropic stays optional

            self._client = anthropic.Anthropic()
        return self._client

    def count(self, text: str) -> int:
        resp = self._get_client().messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return resp.input_tokens


def default_counter() -> TokenCounter:
    """Best available zero-config counter.

    Uses ``tiktoken`` when installed (exact, local); otherwise falls back to
    the heuristic :class:`ApproxCounter` and warns once.
    """
    try:
        return OpenAICounter()
    except ImportError:
        warnings.warn(
            "tiktoken not installed; using ApproxCounter (~15% error). "
            "Install nomem[openai] or pass an exact TokenCounter for a strict budget.",
            stacklevel=2,
        )
        return ApproxCounter()
