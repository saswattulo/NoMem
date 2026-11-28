# Python API

Everything below is importable from `nomem`.

## `NoMem`

The engine for one `user_id` and `session_id`. Constructor arguments are in
[Configuration](configuration.md).

### Writing

| Method | Returns | What it does |
| --- | --- | --- |
| `add(role, content, *, metadata=None, timestamp=None)` | `Message` | Store a message verbatim, annotate it, extract facts, detect conflicts. `timestamp` is epoch seconds. `metadata={"speaker": "Name"}` attributes facts to a named speaker. |
| `add_user(content, **kw)` | `Message` | `add("user", ...)` |
| `add_assistant(content, **kw)` | `Message` | `add("assistant", ...)` |
| `remember(key, value)` | `None` | Pin a durable fact for this user. |
| `forget(message_ids)` | `int` | Delete messages (an id or a list) and everything derived from them. Facts they had replaced become current again. |
| `forget_session()` | `int` | Delete every message of this user and session. Pinned facts stay. |

### Reading

| Method | Returns | What it does |
| --- | --- | --- |
| `build_context(query=None)` | `ContextResult` | Assemble a prompt guaranteed to fit `max_tokens`. |
| `search(query, k=None)` | `list[Message]` | Best `k` matches (default `retrieval_top_k`), best first. |
| `search_hits(query, k=None)` | `list[Hit]` | The same, with scores and reasons. |
| `facts(subject=None)` | `list[dict]` | Current slot facts. |
| `fact_history()` | `list[dict]` | Every slot fact, including closed ones. |
| `conflicts(message_id=None)` | `list[dict]` | Linked pairs where a newer memory may change an older one. |
| `message_count()` | `int` | Messages in this session. |
| `log` | `list[LogEntry]` | Decisions from the last `build_context()`. |
| `close()` | `None` | Close the store, if this instance opened it. |

A fact dict has `id`, `user_id`, `subject`, `attribute`, `value`, `multi`,
`valid_from`, `valid_to`, `source_message_id` and `closed_by`. A conflict
dict has `older_id`, `newer_id`, `older`, `newer`, `older_at`, `newer_at`,
`kind` (`slot`, `frame` or `update`), `attribute`, `detail` and
`confidence`. Times are epoch seconds.

## Data classes

### `Message`

| Field | Type | |
| --- | --- | --- |
| `role` | `str` | `user`, `assistant`, `system` or `tool` |
| `content` | `str` | Exactly as stored |
| `id` | `int` | Row id |
| `created_at` | `float` | Epoch seconds |
| `tokens` | `int` | Cost under the instance's counter |
| `metadata` | `dict` | Whatever you passed |

`as_dict()` returns `{"role", "content"}`.

### `ContextResult`

| Field | | 
| --- | --- |
| `text` | The rendered prompt (`role: content` lines) |
| `messages` | `[{"role", "content"}]` for chat APIs |
| `tokens_used` | Tokens the context costs, never more than `max_tokens` |
| `max_tokens` | The budget it was built for |
| `log` | `list[LogEntry]` |

`explain()` returns the log as a readable table; `str(result)` is `text`.

### `LogEntry`

`action` (`core`, `facts`, `added`, `dropped`, `budget_exhausted`,
`retrieved`, `retrieval`, `skipped`, `summarized`), `tier` (`core`,
`working`, `archival`), `detail`, `tokens`, `budget_remaining`, `ref`.

### `Hit`

`message` (`Message`), `score` (`float`), `reasons` (`list[str]`, such as
`["keyword#1", "entity#2"]`).

## Configuration objects

- `RetrievalConfig`: ranker weights and priors. See
  [Configuration](configuration.md#retrievalconfig).
- `ConflictConfig`: contradiction detection. See
  [Configuration](configuration.md#conflictconfig).

## Token counters

| Class | Needs | |
| --- | --- | --- |
| `ApproxCounter(chars_per_token=4.0)` | nothing | Deterministic heuristic, about 15% off |
| `OpenAICounter(model="gpt-4o")` | `tiktoken` | Exact and local |
| `AnthropicCounter(model=..., client=None)` | `anthropic` and an API key | Exact, one API call per count |
| `TokenCounter` | | Base class: implement `count(text) -> int`; `per_message_overhead` defaults to 4 |
| `default_counter()` | | `OpenAICounter` if `tiktoken` is installed, else `ApproxCounter` |

## Embedders

`Embedder` (base: implement `encode(texts)`), `Model2VecEmbedder`,
`SentenceTransformerEmbedder`, `HashingEmbedder`. See
[Retrieval and vectors](retrieval.md#optional-vectors).

## Compactors

| Class | |
| --- | --- |
| `ExtractiveCompactor` | Default. Quotes the most informative sentences of dropped turns, with dates. Never paraphrases. |
| `NullCompactor` | No summary. |
| `Compactor` | Base class: implement `compact(messages, previous_summary="", *, max_tokens=None, counter=None) -> str`. |

The compactor only runs when `summary_ratio > 0`.

## Storage

`SQLiteStore(path=":memory:")`: one SQLite connection holding messages,
FTS5, annotations, vectors, facts and conflicts. Pass it as `db=` to share
one file between several `NoMem` instances. If your SQLite build lacks FTS5,
keyword search is disabled and everything else still works.

## Servers

- `MemoryService(db="nomem.db", *, max_tokens=3000, core="", embedder="none", **nomem_kwargs)`:
  many users and sessions over one store; methods `add`, `add_many`,
  `remember`, `context`, `search`, `facts`, `conflicts`, `stats`, `close`.
  See [HTTP API and MCP](server.md#python-memoryservice).
- `nomem.http_api.serve(service, host, port, api_key=None)` and
  `nomem.http_api.make_handler(...)`.
- `nomem.mcp_server.build_server(service, user_id, session_id)` and
  `nomem.mcp_server.run(...)`.
