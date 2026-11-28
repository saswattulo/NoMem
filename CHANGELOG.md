# Changelog

## Unreleased

- **Framework adapters** in `nomem.integrations`, each an optional extra:
  Claude API (`AnthropicMemory`: context for `messages.create`, tools, a
  `tool_use` handler, tool-runner tools), OpenAI Agents SDK (`NoMemSession`,
  `memory_tools`), LangChain (`NoMemChatMessageHistory`, `NoMemRetriever`,
  `memory_tools`), LangGraph (`NoMemStore`, a `BaseStore`; `NoMemGraphMemory`),
  LlamaIndex (`memory_tools`, `NoMemMemoryBlock`) and CrewAI (`memory_tools`).
  All share `MemoryToolkit`: the MCP server's tools, which never raise into
  the agent loop.
- **Forgetting.** `NoMem.forget(ids)` and `forget_session()` delete messages
  with their index entries, vectors, facts and conflict links;
  `MemoryService.forget()` and `clear()`.
- Text-only logo (thin `nomem` over a budget bar): `nomem logo`, the README,
  the docs, and a startup banner for `nomem serve` and `nomem mcp`
  (`--quiet` to skip).
- Budget readouts: `explain()` now opens with a token gauge and a budget map
  (one cell per 1/30th of the budget, by core, facts, retrieved, summary and
  recent turns); `ContextResult.usage()`, `gauge()` and `budget_map()`; the
  HTTP `/v1/context` response includes `usage`.
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
