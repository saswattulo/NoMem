# NoMem — Not Only Memory

A transparent, token-budgeted **memory layer for agentic and RAG applications**.
Framework-neutral core, local-first (SQLite), and — by design — **no LLM
anywhere in the pipeline**: no API key, no model download, no per-message
model calls. Memory is stored verbatim, indexed with deterministic rules, and
assembled at read time.

> Context engineering is the real bottleneck for agents: overflowing windows,
> incoherent long conversations, and black-box "memory" you can't inspect.
> NoMem gives you a hard token budget and a full audit trail of every decision.

**Status:** `v0.2` — Milestones 1–3 (core engine, hybrid retrieval, LLM-free
facts and compaction). See the [roadmap](#roadmap) and [benchmarks](#benchmarks).

## Why NoMem

- **Hard token budget.** Context is *guaranteed* to fit `max_tokens`. No more
  manual counting or overflow errors.
- **Fully transparent.** Every message kept, dropped, retrieved, or summarised
  is logged with the reason and the token cost. No black boxes.
- **No LLM in the memory layer.** Tools like mem0, Zep and Letta call an LLM on
  every write to extract, merge and summarise memories. NoMem replaces each of
  those jobs with rules, hashes and ranking, so writes take well under a
  millisecond and cost nothing, and every memory is something that was
  actually said.
- **Three memory tiers.** Core (persona + known facts) → Working (recent
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
pip install nomem[embed]     # optional vector retrieval (Model2Vec, CPU-only)
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

`result.explain()` prints exactly what happened and why. Here's the trace from
the fuller demo (`python examples/basic_usage.py`), where the budget forces
drops and retrieval pulls an already-dropped fact back in:

```
context: 102/120 tokens, 5 messages
-----------------------------------
core             core         15t [105 left]  included fixed core memory
retrieved        archival     31t [0 left]  user message #3 (matched by keyword#1)
skipped          archival      0t [0 left]  user message #7 (already in the recent window)
skipped          archival      0t [0 left]  user message #5 (already in the recent window)
skipped          archival      0t [0 left]  assistant message #6 (already in the recent window)
dropped          archival     24t [0 left]  assistant message #4 (would exceed 31-tok archival budget)
retrieval        archival     31t [0 left]  retrieved 1 of 5 match(es) for query
added            working      16t [58 left]  user message #7
added            working      22t [36 left]  assistant message #6
added            working      18t [18 left]  user message #5
dropped          working      24t [18 left]  assistant message #4 (needs 24, 18 left)
dropped          working      17t [18 left]  assistant message #2 (needs 17, 18 left)
dropped          working      14t [18 left]  user message #1 (needs 14, 18 left)
budget_exhausted working       0t [18 left]  3 older message(s) dropped to fit the budget
```

(Exact token numbers depend on the `TokenCounter` in use.)

## The three tiers

| Tier         | What it holds                                              | Storage                          | Scope                 |
| ------------ | ---------------------------------------------------------- | -------------------------------- | --------------------- |
| **Core**     | Fixed persona/rules + a "Known facts" card                 | `slots` + `facts` tables         | `user_id`             |
| **Working**  | Recent conversation (budget-enforced window)               | `messages` (+ optional summary)  | `user_id`+`session_id`|
| **Archival** | Every message, verbatim, retrieved on demand               | `messages` + FTS5 + annotations  | `user_id`+`session_id`|

Everything lives in one local SQLite file (or `:memory:`).

## How it works without an LLM

**Write path: store verbatim, annotate with rules.** `add()` stores the message
exactly as written, then attaches deterministic annotations beside it:
entities, keyphrases, resolved dates ("yesterday", "two weeks ago",
"7 May 2023", anchored to the message's own timestamp), an importance score,
duplicate hashes, and first-person **slot facts** such as
`location = Berlin` or `allergy = peanuts`. A new value for a single-valued
slot closes the old one (`valid_to`, `closed_by`) instead of deleting it, so
history is kept. Pass `timestamp=` to import an existing transcript.

**Read path: decide at query time.** `build_context(query=...)` assembles, in
one hard budget:

1. **Core** persona, then a **Known facts** card with the slot facts the query
   asks about (`facts_ratio`, default 10% of the budget). Ask "what do I do
   for a living?" and it finds `job: backend engineer` with no shared keyword.
2. **Retrieved memories**, each shown with its date. Several rankers propose
   candidates and Reciprocal Rank Fusion merges them: FTS5 BM25 keywords,
   entities (ignoring ones that appear everywhere), dates in the question,
   slot facts, optional vectors, and the turns either side of a strong hit.
   An importance prior re-orders the result and duplicates are collapsed.
   A memory that a later one contradicts carries a conflict marker quoting
   the other side with its date (see [Contradictions](#contradictions)).
3. **Recent conversation**, newest first, as a contiguous window.
4. Optionally, an **extractive summary** of turns that just fell out of the
   window (`summary_ratio`, off by default): the most informative sentences,
   quoted with dates, never paraphrased.

`result.explain()` shows every decision, including which rankers found each
memory (`matched by keyword#1, entity#1, slot#1`).

| Knob | Default | What it does |
| --- | --- | --- |
| `retrieval_top_k` | 5 | Memories considered per query |
| `archival_ratio` | 0.3 | Budget share reserved for retrieved memories (unused rolls over) |
| `facts_ratio` | 0.1 | Most the Known-facts card may use (unused rolls over) |
| `summary_ratio` | 0.0 | Budget share for the extractive summary (off by default) |
| `retrieval` | `RetrievalConfig()` | Ranker weights, RRF k, importance and recency priors |
| `annotate` | `True` | Rule-based annotation on write |
| `embedder` | `None` | Optional vector ranker |

```python
mem.add_user("I live in Pune.")
mem.add_user("I moved to Bangalore last week!")
mem.facts()          # current facts: location = Bangalore
mem.fact_history()   # Pune too, with when it stopped being true
mem.search_hits("where did I live before?")  # hits with scores and reasons
mem.remember("preferred name", "Sas")         # pin a fact yourself
```

See `examples/facts_demo.py` and `examples/retrieval_demo.py`.

### Contradictions

People change their minds, addresses and flight times. NoMem doesn't pick a
winner or rewrite anything: when a newer memory changes an older one, both
are kept, linked as a *conflict*, and shown together with their dates so the
agent's own model decides. Three deterministic detectors run on write:

| Detector | Catches | Example |
| --- | --- | --- |
| `slot` | A single-valued fact gets a new value, or a like is negated | "I live in Pune" ... "I moved to Bangalore" |
| `frame` | The same sentence with a different number, time, date or name | "I have 2 kids" ... "I have 3 kids" |
| `update` | An update cue ("moved", "postponed", "now", "actually", "instead" ...) plus a shared topic word and a changed value of the same type, from the same speaker | "My flight leaves at 9am" ... "Actually my flight got moved to 11am" |

Recency only decides direction: the later memory is marked as the update.
Retrieved memories then read:

```
- [5 Jan 2026] user: My flight leaves at 9am on Friday. [CONFLICT: changed on 25 Jan 2026 (flight: 09:00 -> 11:00); user said "Actually my flight got moved to 11am."]
```

and the Known-facts card says `location: Bangalore (since 20 Mar 2026; before
that Pune, said 1 Mar 2026)`. `mem.conflicts()` lists every pair with both
texts, dates, the detector and a confidence. With an embedder, vector
neighbours add candidates and raise confidence. Tune or turn it off with
`NoMem(conflicts=ConflictConfig(...))`.

The rules aim for precision: on real LoCoMo chat they flag 2 pairs in 5,882
turns. On a fresh held-out set of 25 updates and 25 look-alike controls,
written before the latest rule changes and never tuned on, they catch 19/25
with 0 false flags. Topic words match across singular and plural, weekday
schedules ("on Tuesdays") are compared by name, and a short plain statement
whose wording changes after a strong cue ("my commute is by car" ... "I
commute by train now instead") counts too. Still missed: numbers written as
words ("two cars"), ordinals ("the 7th floor"), and brand names in lower
camel case ("iPhone").

### Optional vectors

The core is model-free. For semantic matching, pass an embedder (an
embedding model, never a generative one):

```python
from nomem import NoMem, Model2VecEmbedder
mem = NoMem(embedder=Model2VecEmbedder())  # minishlab/potion-base-8M by default
```

`Model2VecEmbedder` is the recommended default: ~8M parameters, CPU-only,
~30 MB, fast enough to embed every message on write (~0.4 ms each). On
LoCoMo it lifts search recall@5 from 56.8% to 58.6% and context recall from
57.6% to 59.7%. The vector ranker is a light tie-breaker by default
(`weights["vector"] = 0.15`): static embeddings are weaker than keyword
matching on their own (41.7% recall@5 vector-only) and pull recall *down*
at equal weight. `SentenceTransformerEmbedder` takes any sentence-transformers
model (raise the vector weight for a stronger encoder); subclass `Embedder`
for anything else. `HashingEmbedder` (character n-grams, no model) exists for
offline use and tests.

## Design choices

- **Pluggable token counting.** `TokenCounter` interface with `OpenAICounter`
  (exact, local via tiktoken), `AnthropicCounter` (exact via API), and a
  zero-dependency `ApproxCounter` fallback.
- **No LLM, by design.** Nothing in NoMem calls a language model: not on
  write, not on read, not for compaction. Facts, dates, entities and
  importance come from deterministic rules; compaction is *extractive* (it
  quotes the most informative sentences, with dates, and never paraphrases),
  so every line in the context can be traced back to something actually said.
  There is no `LLMCompactor`, and there won't be one.
- **Local-first, expandable.** SQLite today; in-memory and vector-DB backends
  behind the same store interface later.

## Benchmarks

Measured without any LLM (`python benchmarks/run_all.py`; details and every
step in [`benchmarks/TRAJECTORY.md`](benchmarks/TRAJECTORY.md)). LoCoMo: did
the evidence turn for each of 1,531 answerable questions land in the top-5
search results, and in a 1,000-token context? LongMemEval (`longmemeval_s`,
470 answerable questions, each over its own ~500-turn history): did the
evidence session land in the top 5 sessions, and the evidence turn in the
top 5 turns and the 1,000-token context?

| | v0.1 (FTS5 only) | v0.2 (hybrid, LLM-free) | v0.2 + Model2Vec |
| --- | --- | --- | --- |
| LoCoMo search recall@5 | 50.9% | 56.8% | 58.6% |
| LoCoMo context recall | 52.5% | 57.6% | 59.7% |
| LongMemEval session recall@5 | | 89.9% | 92.1% |
| LongMemEval turn recall@5 | | 52.7% | 55.9% |
| LongMemEval context recall | | 41.1% | 44.2% |
| Query latency at 50k messages (p50) | 190 ms | 3-5 ms | 7-10 ms |
| Planted facts recalled at 50k messages | 7/8 | 8/8 | 8/8 |
| Updated facts caught, fresh held-out (false flags) | | 19/25 (0/25) | 19/25 (0/25) |
| Write throughput | | ~4,000 msg/s | ~1,500 msg/s |
| Write cost | 0 model calls | 0 model calls | 1 embedding (no generative model) |

## Roadmap

- **M1 — Core engine** ✅ Token-budgeted working memory, SQLite persistence,
  user/session scoping, full transparency log.
- **M2 — Archival retrieval** ✅ FTS5 keyword search wired into `build_context`.
- **M3 — LLM-free memory** ✅ Rule-based annotations (entities, dates,
  keyphrases, importance), bi-temporal slot facts with supersession, hybrid
  retrieval with RRF, a Known-facts card, an extractive compactor, optional
  embedders, and a benchmark harness.
- **M3.5 — Contradictions** ✅ Slot, frame, update-cue and wording-change conflict detection;
  both memories surfaced with dates and a conflict marker; updated-facts
  benchmark.
- **M4 — Adapters & polish** LangChain `BaseMemory` adapter, tool-output
  isolation helper, demo notebook.
- **M5 — Optional encoders** GLiNER entities and a co-occurrence graph with
  Personalized PageRank, cross-encoder reranking (`nomem[ner]`,
  `nomem[rerank]`), pluggable pattern packs for other languages.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[Apache-2.0](LICENSE)
