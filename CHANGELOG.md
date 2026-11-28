# Changelog

## Unreleased

- **Score fusion** replaces rank-only RRF: keyword and vector add their
  scores, long turns get a mild length penalty, and the neighbour lift only
  goes to turns in the same role. LongMemEval turn R@5 57.1% -> 77.7% (Mem0
  OSS on the same harness: 69.0%), LoCoMo R@5 58.6% -> 62.3%, with
  Model2Vec. Keyword only: 55.2% -> 72.0% and 56.8% -> 59.6%.
  `RetrievalConfig.legacy()` keeps the old ranking.
- **Fix:** search in one session could return another session's message
  through the slot (facts) ranker.
- **Robustness:** a failing embedder no longer drops the message or breaks
  search (it warns and falls back to keywords); stored vectors of another
  size are skipped; NaN vectors, very long queries and bad
  `RetrievalConfig` values are handled.
- Text-only logo: `nomem logo`, and a startup banner for `nomem serve` and
  `nomem mcp` (`--quiet` to skip).
- NoMem vs Mem0 comparison page in the docs.

## 0.3.0

- **Packaging.** Published on PyPI as `nomem-ai` (the name `nomem` is taken
  by an unrelated project); `import nomem` is unchanged. New extras: `mcp`,
  `all`, `docs`.
- **HTTP API.** `nomem serve`: a JSON API on the standard library only, with
  optional bearer-token auth.
- **MCP server.** `nomem mcp`: `remember`, `recall`, `search_memory`,
  `list_facts`, `pin_fact` and `list_conflicts` tools for Claude Desktop,
  Claude Code and other MCP clients. Works with `mcp` 1.x and 2.x.
- **`MemoryService`.** Many users and sessions over one SQLite file, thread
  safe, JSON-ready results.
- **Docs site** at https://saswattulo.github.io/NoMem/.
- **Flexible archival slice.** `archival_max_ratio` (default 0.8) lets
  retrieved memories stretch past the 30% slice for a direct match.
  LongMemEval context recall 41.1% to 50.0% (rules only).
- **LongMemEval** retrieval benchmark.
- CI on Python 3.10 to 3.13, and a trusted-publishing release workflow.

## 0.2.0

- **Contradictions.** Slot, frame, update-cue and wording-change detectors;
  both memories kept and shown with a `[CONFLICT: ...]` marker.
- **Hybrid retrieval.** Keyword, entity, time, slot, vector and context
  rankers fused with Reciprocal Rank Fusion; importance prior; near-duplicate
  collapsing.
- **Facts.** Bi-temporal slot facts with supersession; Known-facts card.
- **Optional vectors.** `Model2VecEmbedder` (potion-base-8M) by default.
- **Extractive compactor**, rule-based annotations, benchmark harness.

## 0.1.0

- Token-budgeted working memory, SQLite persistence, user and session
  scoping, FTS5 archival search, full transparency log.
