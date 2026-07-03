# NoMem — Not Only Memory

A transparent, token-budgeted **memory layer for agentic and RAG applications**.
Framework-neutral core, local-first (SQLite), and — by design — **no LLM, no API
key, and no model download required** to get started.

> Context engineering is the real bottleneck for agents: overflowing windows,
> incoherent long conversations, and black-box "memory" you can't inspect.
> NoMem gives you a hard token budget and a full audit trail of every decision.

**Status:** `v0.1` — Milestones 1–2 (core engine + archival retrieval). See the [roadmap](#roadmap).

## Why NoMem

- **Hard token budget.** Context is *guaranteed* to fit `max_tokens`. No more
  manual counting or overflow errors.
- **Fully transparent.** Every message kept, dropped, retrieved, or summarised
  is logged with the reason and the token cost. No black boxes.
- **Three memory tiers.** Core (fixed persona/facts) → Working (recent
  conversation) → Archival (everything, retrieved on demand).
- **Lightweight & local-first.** One SQLite file. Zero required dependencies.
  Runs offline in seconds.
- **Harness-neutral.** A clean core with thin adapters (LangChain, LlamaIndex,
  raw provider loops) — not tied to any one framework.

## Install

```bash
pip install nomem            # core: zero dependencies
pip install nomem[openai]    # exact local token counts via tiktoken
pip install nomem[anthropic] # exact token counts via the Anthropic API
```

*(Pre-release: install from source with `pip install -e .`)*

## Quickstart

```python
from nomem import NoMem

mem = NoMem(
    max_tokens=120,
    core="You are a concise travel assistant.",
    user_id="saswat",
    session_id="trip-planning",
)

mem.add_user("I'm planning a trip to Tokyo, early March, budget ~$2000.")
mem.add_assistant("Great — cherry blossoms start late March, so you may be early.")
mem.add_user("Let's stay in Shinjuku. Any food I must try?")

result = mem.build_context(query="where am I staying and what's my budget?")

print(result.text)        # prompt string, guaranteed <= max_tokens
print(result.messages)    # structured form for chat APIs
print(result.tokens_used) # exact
print(result.explain())   # the full transparency log
```

`result.explain()` prints exactly what happened and why:

```
context: 118/120 tokens, 3 messages
------------------------------------
core             core         38t [82 left]  included fixed core memory
added            working      41t [41 left]  user message #3
added            working      39t  [2 left]  assistant message #2
dropped          working      52t  [2 left]  user message #1 (needs 52, 2 left)
budget_exhausted working       0t  [2 left]  1 older message(s) dropped to fit the budget
```

Run the full demo:

```bash
python examples/basic_usage.py
```

## The three tiers

| Tier         | What it holds                                   | Storage                       | Scope                |
| ------------ | ----------------------------------------------- | ----------------------------- | -------------------- |
| **Core**     | Fixed persona/rules + durable user facts        | `facts` table                 | `user_id`            |
| **Working**  | Recent conversation (budget-enforced window)    | `messages` (+ rolling summary)| `user_id`+`session_id`|
| **Archival** | Every message, retrieved on demand              | `messages` + FTS5             | `user_id`+`session_id`|

Everything lives in one local SQLite file (or `:memory:`). Archival retrieval
uses SQLite's built-in **FTS5** keyword search — no embedding model needed.

### Retrieval

Pass a `query` to `build_context(query=...)` and NoMem pulls older, relevant
messages back into context — even ones the sliding window already dropped:

```python
mem.add_user("Important: I'm severely allergic to peanuts.")
# ... many turns later, the allergy has slid out of the window ...
result = mem.build_context(query="what snacks are safe given my peanut allergy?")
# -> the allergy fact is retrieved and re-inserted, and the log shows why
```

Two knobs (sensible defaults): `retrieval_top_k` (how many matches to consider,
default 5) and `archival_ratio` (fraction of the budget reserved for retrieved
memories, default `0.3`; unused reservation rolls over to recent conversation).
Retrieval is keyword-based in v0.1 — semantic/vector retrieval is a later
upgrade behind the same interface. See `examples/retrieval_demo.py`.

## Design choices

- **Pluggable token counting.** `TokenCounter` interface with `OpenAICounter`
  (exact, local via tiktoken), `AnthropicCounter` (exact via API), and a
  zero-dependency `ApproxCounter` fallback.
- **Pluggable, LLM-free compaction.** The default compactor is deterministic and
  *extractive* (keep high-signal turns/facts, drop filler) — reproducible and
  free. An `LLMCompactor` drops in behind the same interface for abstractive
  summaries. *(Lands in M3.)*
- **Local-first, expandable.** SQLite today; in-memory and vector-DB backends
  behind the same store interface later.

## Roadmap

- **M1 — Core engine** ✅ Token-budgeted working memory, SQLite persistence,
  user/session scoping, full transparency log.
- **M2 — Archival retrieval** ✅ FTS5 keyword search wired into `build_context`;
  relevant past messages pulled back in when the budget allows, every retrieval
  trade-off logged.
- **M3 — Rule-based compaction** Extractive summariser + facts extraction into
  the core tier; rolling summaries in working memory.
- **M4 — Adapters & polish** LangChain `BaseMemory` adapter, tool-output
  isolation helper, long-conversation tests, demo notebook.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[Apache-2.0](LICENSE)
