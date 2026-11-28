# How it works

NoMem's design fits in one line: **store verbatim, index richly, decide at
read time.** Nothing is rewritten when it is stored, and nothing is decided
until a question is asked.

## Why no LLM

Memory layers such as Mem0, Zep and Letta call a language model on every
write: to extract "memories" from a conversation, merge them with old ones,
and summarise. That has three costs:

1. **Money and latency on every message.** Each write is one or more model
   calls.
2. **Lossy, unverifiable memory.** What is stored is the model's paraphrase.
   If the extraction misses a detail or invents one, the original is gone.
3. **Decisions made too early.** At write time nobody knows what will be
   asked later, so the model has to guess what matters.

NoMem replaces each of those jobs with rules, hashes and ranking. Writes take
well under a millisecond and cost nothing, and every line that reaches the
prompt is something that was actually said, with its date. The agent's own
model, which you are already paying for, does the reasoning at answer time,
with the evidence in front of it.

"No LLM" here means no generative model anywhere in the pipeline. An
optional small *embedding* model (Model2Vec, 8M parameters) can add a vector
ranker; the core works without it.

## The three tiers

| Tier | What it holds | Storage | Scope |
| --- | --- | --- | --- |
| **Core** | Fixed persona or rules, plus a *Known facts* card | `slots` and `facts` tables | `user_id` |
| **Working** | Recent conversation, as a contiguous window | `messages` | `user_id` + `session_id` |
| **Archival** | Every message ever stored, retrieved on demand | `messages`, FTS5, annotations, optional vectors | `user_id` + `session_id` |

Everything lives in one SQLite file (or `:memory:`). Facts are scoped to the
user, so a fact learned in one session shows up in the next; messages are
scoped to the session.

## Write path

`mem.add(role, content)` does, in one transaction:

1. **Store the message verbatim**, with its token count and timestamp. Pass
   `timestamp=` (epoch seconds) to import an old transcript.
2. **Index it** in SQLite FTS5 (BM25 keyword search).
3. **Annotate it with rules**: named entities, keyphrases, resolved dates
   ("yesterday", "two weeks ago", "7 May 2023", anchored to the message's own
   time), an importance score, an exact and a near-duplicate hash.
4. **Extract slot facts** from first-person statements: `location = Berlin`,
   `job = backend engineer`, `allergy = peanuts`, `likes = hiking`. A new
   value for a single-valued slot *closes* the old one (`valid_to`,
   `closed_by`) instead of deleting it.
5. **Check for contradictions** against earlier memories and link any it
   finds (see [Facts and contradictions](facts-and-conflicts.md)).
6. Optionally, **embed it** if you configured an embedder.

No step calls a model, and none of them changes what was said.

## Read path

`mem.build_context(query=...)` assembles one prompt under a hard budget of
`max_tokens`, in priority order:

1. **Core**: your persona or rules, always first.
2. **Known facts**: current slot facts and pinned facts, the ones the query
   is about first. Uses at most `facts_ratio` (10%) of the budget.
3. **Retrieved memories**: the best matches for the query, each shown with
   its date and any conflict marker. They get `archival_ratio` (30%) of the
   remaining budget, and may stretch to `archival_max_ratio` (80%) for a
   direct match that does not fit, taking tokens from the oldest recent turns.
4. **Recent conversation**: newest first, as a contiguous window, until the
   budget runs out.
5. Optionally, an **extractive summary** of the turns that just fell out of
   the window (`summary_ratio`, off by default): the most informative
   sentences, quoted with dates, never paraphrased.

Unused budget rolls over: if no facts match, their share goes to the recent
window. The result is guaranteed to fit `max_tokens`. That invariant is
checked in 500 randomised trials in the stress benchmark, with 0 violations.

Without a `query`, `build_context()` is a budgeted sliding window plus the
facts card.

## Retrieval

Several rankers each propose candidates, and Reciprocal Rank Fusion merges
them:

| Ranker | Finds | Weight |
| --- | --- | --- |
| `keyword` | BM25 matches from FTS5 | 1.0 |
| `entity` | Messages sharing a named entity with the query (entities that appear everywhere are ignored) | 0.5 |
| `time` | Messages about a date the query mentions ("what did I do last May?") | 0.5 |
| `slot` | Messages that set a fact the query asks about ("what do I do for work?" finds `job`) | 0.5 |
| `vector` | Nearest neighbours by embedding, if an embedder is set | 0.15 |
| `context` | The turns either side of a strong hit | 0.2 |

An importance prior re-orders the fused list, near-duplicates are collapsed,
and `search_hits()` tells you which rankers found each result
(`matched by keyword#1, entity#1, slot#1`). See
[Retrieval and vectors](retrieval.md).

## Transparency

Every decision `build_context()` makes produces a `LogEntry`: included,
dropped (and why), retrieved (and by which rankers), skipped as a duplicate of
the recent window, stretched, summarised. `result.explain()` prints them. If
the agent forgets something, you can see exactly where it fell out.
