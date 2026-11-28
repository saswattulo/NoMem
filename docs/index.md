---
title: LLM-free memory for agents
description: LLM-free memory for AI agents. Verbatim storage, a hard token budget, and a log of every decision.
hide:
  - navigation
  - toc
---

<div class="nm-hero" markdown>

<pre class="nm-logo" aria-label="NoMem"><span class="nm-no">███╗   ██╗ ██████╗ </span>███╗   ███╗███████╗███╗   ███╗
<span class="nm-no">████╗  ██║██╔═══██╗</span>████╗ ████║██╔════╝████╗ ████║
<span class="nm-no">██╔██╗ ██║██║   ██║</span>██╔████╔██║█████╗  ██╔████╔██║
<span class="nm-no">██║╚██╗██║██║   ██║</span>██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║
<span class="nm-no">██║ ╚████║╚██████╔╝</span>██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║
<span class="nm-no">╚═╝  ╚═══╝ ╚═════╝ </span>╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝</pre>

<p class="nm-eyebrow">Not Only Memory</p>

# Memory for agents, with no LLM in the loop

<p class="nm-lead">NoMem stores every message verbatim, indexes it with
deterministic rules, and builds a context that is guaranteed to fit your
token budget. Writes cost nothing, and you can see why every line is in the
prompt.</p>

<div class="nm-cta" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/saswattulo/NoMem){ .md-button }
</div>

```bash
pip install nomem-ai
```

</div>

<div class="nm-stats" markdown>
<div markdown><strong>0</strong><span>model calls per write</span></div>
<div markdown><strong>&lt;1 ms</strong><span>to store a message</span></div>
<div markdown><strong>92.1%</strong><span>LongMemEval session recall@5</span></div>
<div markdown><strong>0</strong><span>budget overruns in 500 random trials</span></div>
</div>

## Thirty seconds of NoMem

```python
from nomem import NoMem

mem = NoMem(max_tokens=300, user_id="saswat", db="memory.db")

mem.add_user("I live in Pune and work as a backend engineer.")   # 1 Mar
mem.add_user("My flight to Tokyo leaves at 9am on Friday.")      # 2 Mar
# ... sixty turns about recipes and weekend plans ...
mem.add_user("I moved to Bangalore last week!")                  # 20 Mar
mem.add_user("Actually my flight got moved to 11am.")            # 25 Mar

ctx = mem.build_context(query="when is my flight, and where do I live now?")
print(ctx.text)       # 290 of 300 tokens: it always fits
print(ctx.explain())  # why every line is there, and what was dropped
```

```text
system: Known facts:
- location: Bangalore (since 20 Mar 2026; before that Pune, said 1 Mar 2026)
system: Relevant earlier context:
- [1 Mar 2026] user: I live in Pune and work as a backend engineer. [CONFLICT: changed on 20 Mar 2026 (location: Pune -> Bangalore); user said "I moved to Bangalore last week!"]
- [2 Mar 2026] user: My flight to Tokyo leaves at 9am on Friday. [CONFLICT: changed on 25 Mar 2026 (flight: 09:00 -> 11:00); user said "Actually my flight got moved to 11am."]
...
user: I moved to Bangalore last week!
user: Actually my flight got moved to 11am.
```

Nothing above came from a language model. The facts card, the dates and the
conflict marker are rules over what the user actually said.

## Why NoMem

<div class="grid cards" markdown>

-   :material-cash-remove: **No LLM, no bill**

    ---

    Mem0, Zep and Letta call a model on every write to extract and merge
    memories. NoMem replaces those jobs with rules, hashes and ranking, so a
    write takes well under a millisecond and costs nothing.

-   :material-format-quote-close: **Verbatim, never paraphrased**

    ---

    Every memory is something that was actually said, with its date. Nothing
    is lost to a bad extraction, and nothing is invented.

-   :material-gauge: **A hard token budget**

    ---

    `build_context()` is guaranteed to fit `max_tokens`. Core persona, facts,
    retrieved memories and recent turns share one budget, in priority order.

-   :material-magnify-scan: **Every decision logged**

    ---

    `explain()` shows each message kept, dropped, retrieved or summarised,
    with the reason and its token cost. No black box.

-   :material-swap-horizontal: **Contradictions surfaced, not hidden**

    ---

    When a fact changes, both versions are kept and shown with their dates,
    so your agent's own model decides what is true now.

-   :material-database: **Local-first**

    ---

    One SQLite file, zero required dependencies, runs offline. Optional
    8M-parameter embeddings on CPU if you want a vector ranker.

</div>

## Use it your way

=== "Python"

    ```python
    from nomem import NoMem
    mem = NoMem(max_tokens=2000, db="memory.db")
    mem.add_user("I'm allergic to peanuts.")
    mem.build_context(query="what can't I eat?").text
    ```

=== "HTTP API"

    ```bash
    nomem serve --db memory.db
    curl -s localhost:8420/v1/messages -d '{"content": "I am allergic to peanuts."}'
    curl -s localhost:8420/v1/context  -d '{"query": "what can I not eat?"}'
    ```

=== "MCP (Claude Desktop, Claude Code)"

    ```bash
    pip install "nomem-ai[mcp]"
    claude mcp add nomem -- nomem mcp --db ~/.nomem/memory.db
    ```

## How it compares

NoMem against Mem0's own search engine, on the same questions with the same
scoring and no LLM on either side. Full method and numbers:
[NoMem vs Mem0](comparison.md).

--8<-- "docs/_comparison_table.md"

<div class="nm-cta" markdown>
[Read how it works](concepts.md){ .md-button }
[See all benchmarks](benchmarks.md){ .md-button }
</div>
