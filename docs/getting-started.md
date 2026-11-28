# Getting started

## Install

NoMem needs Python 3.10 or newer. The core has no dependencies at all.

```bash
pip install nomem-ai
```

The package is called `nomem-ai` on PyPI (the name `nomem` belongs to an
unrelated project). You still `import nomem`, and the command is `nomem`.

Optional extras add features without changing the core:

| Extra | Adds | When to use it |
| --- | --- | --- |
| `nomem-ai[openai]` | `tiktoken` | Exact, local token counts. Recommended for anything you ship. |
| `nomem-ai[anthropic]` | `anthropic` | Exact counts from Anthropic's `count_tokens` API. |
| `nomem-ai[embed]` | `model2vec`, `numpy` | Optional vector ranker (an 8M-parameter static embedding model, CPU only). |
| `nomem-ai[mcp]` | `mcp` | The MCP server (`nomem mcp`). |
| `nomem-ai[all]` | all of the above except `anthropic` | Everything local. |

From source:

```bash
git clone https://github.com/saswattulo/NoMem
cd NoMem
pip install -e ".[dev]"
pytest
```

!!! tip "Token counting"
    Without `tiktoken`, NoMem falls back to `ApproxCounter` (about 4 characters
    per token, roughly 15% off) and warns once. The budget is still enforced,
    but against the approximate count. Install `nomem-ai[openai]` or pass your
    own [`TokenCounter`](api.md#token-counters) for an exact one.

## Your first memory

```python
from nomem import NoMem

mem = NoMem(
    max_tokens=120,                       # hard cap on the assembled context
    core="You are a concise travel assistant.",
    user_id="saswat",
    session_id="trip-planning",
    db="memory.db",                       # or ":memory:" (the default)
)

mem.add_user("I'm planning a trip to Tokyo, early March, budget ~$2000.")
mem.add_assistant("Great, cherry blossoms start late March, so you may be early.")
mem.add_user("Let's stay in Shinjuku. Any food I must try?")

result = mem.build_context(query="where am I staying and what's my budget?")

print(result.text)         # prompt string, guaranteed <= max_tokens
print(result.messages)     # [{"role": ..., "content": ...}] for chat APIs
print(result.tokens_used)  # what the context costs
print(result.explain())    # every decision, with the reason and token cost
```

`add()` stores each message exactly as written. `build_context()` decides,
for this query, what goes into the budget: the core persona, the facts the
query is about, the most relevant older memories, and as much recent
conversation as still fits.

## Reading the trace

`result.explain()` is the point of NoMem. From `python examples/basic_usage.py`,
where a small budget forces drops and retrieval pulls an already-dropped
fact back in:

```text
context ▕██████████████████████████░░░░▏ 102/120 tokens, 5 messages
budget  ◆◆◆◆■■■■■■■■▪▪▪▪▪▪▪▪▪▪▪▪▪▪····  ◆ core 15  ■ retrieved 31  ▪ recent 56  · free 18
----------------------------------------------------------------
core             core         15t [105 left]  included fixed core memory
retrieved        archival     31t [0 left]  user message #3 (matched by keyword#1)
skipped          archival      0t [0 left]  user message #7 (already in the recent window)
dropped          archival     24t [0 left]  assistant message #4 (would exceed 31-tok archival budget)
retrieval        archival     31t [0 left]  retrieved 1 of 5 match(es) for query
added            working      16t [58 left]  user message #7
added            working      22t [36 left]  assistant message #6
added            working      18t [18 left]  user message #5
dropped          working      24t [18 left]  assistant message #4 (needs 24, 18 left)
budget_exhausted working       0t [18 left]  3 older message(s) dropped to fit the budget
```

The first two lines read the budget at a glance: a gauge of how much of
`max_tokens` is used, and a map with one cell per 1/30th of the budget,
marked by what spent it (`◆` core persona, `●` known facts, `■` retrieved
memories, `▲` summary, `▪` recent turns, `·` free). `result.usage()` gives the
same split as a dict, and `explain(color=True)` colours both for a terminal.

Each line after that is a [`LogEntry`](api.md#logentry): what happened (`action`), which
tier, the tokens it cost, the budget left, and why. `result.log` gives you the
same entries as objects.

## Facts come for free

```python
mem.add_user("I live in Pune.")
mem.add_user("I moved to Bangalore last week!")

mem.facts()          # [{'attribute': 'location', 'value': 'Bangalore', ...}]
mem.fact_history()   # Pune too, with valid_to set to when it stopped being true
mem.conflicts()      # the two messages, linked, with dates
```

No model extracted those: first-person patterns did. See
[Facts and contradictions](facts-and-conflicts.md).

## Next steps

- [How it works](concepts.md): the three tiers and the write and read paths.
- [Using it with an agent](integrations.md): a loop with OpenAI or Anthropic.
- [HTTP API and MCP](server.md): run NoMem as a service, or give Claude
  Desktop or Claude Code a memory.
- [Configuration](configuration.md): every knob and its default.
