# FAQ

**Does NoMem call an LLM anywhere?**
No. Not on write, not on read, not for compaction. Facts, dates, entities,
importance and contradictions come from deterministic rules. The only
optional model is a small embedding model for the vector ranker, and the core
works without it.

**Then who reasons about the memories?**
Your agent's own model, at answer time. NoMem's job is to put the right
evidence in front of it, verbatim and dated, within a budget. Deciding what a
memory means is left to the model that is already reading the prompt.

**Is it slower or worse than an LLM-based memory?**
Writes are much faster (well under a millisecond, versus one or more model
calls per message) and free. For retrieval quality, see
[Benchmarks](benchmarks.md) and [NoMem vs Mem0](comparison.md). What NoMem
cannot do is answer a question that needs a fact the user never stated in a
retrievable way; an LLM extractor sometimes infers those, and sometimes
invents them.

**What happens when facts change?**
Both versions are kept. The newer one is linked to the older one as a
conflict, and retrieved memories show a `[CONFLICT: ...]` marker with both
dates. The Known-facts card shows the current value and what it replaced.
See [Facts and contradictions](facts-and-conflicts.md).

**Is the token budget exact?**
It is exact for whatever counter you use. With `tiktoken` installed
(`nomem-ai[openai]`), counts are exact for OpenAI tokenizers and close for
others. Without it, `ApproxCounter` is about 15% off, so leave headroom.

**Does it work for languages other than English?**
Storage, keyword search, the budget and the log work in any language. The
rule-based annotators (facts, dates, update cues) are English-only today;
pluggable pattern packs are on the roadmap.

**How big can it get?**
The stress benchmark runs at 50,000 messages: queries take 3-5 ms at p50
with rules only, and all planted facts are still recalled. It is one SQLite
file, so the usual SQLite limits apply.

**Can several processes share the database?**
SQLite allows one writer at a time. For many clients, run one
`nomem serve` and talk to it over HTTP.

**Why is it called `nomem-ai` on PyPI?**
The name `nomem` was already taken by an unrelated project. The import name
(`import nomem`) and the command (`nomem`) are unchanged.

**What license?**
Apache-2.0.
