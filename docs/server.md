# HTTP API and MCP

NoMem ships two servers, both thin wrappers over the same Python engine and
neither calling a model:

- `nomem serve`: a JSON HTTP API, built on the standard library only.
- `nomem mcp`: an [MCP](https://modelcontextprotocol.io) server, so Claude
  Desktop, Claude Code, Cursor or any MCP client can use NoMem as memory.

Both store everything in one SQLite file (`--db`, default `nomem.db`).

## HTTP API

```bash
pip install nomem-ai
nomem serve --db memory.db --port 8420
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--db` | `nomem.db` (or `$NOMEM_DB`) | SQLite file |
| `--host` | `127.0.0.1` | Interface to listen on |
| `--port` | `8420` | Port |
| `--max-tokens` | `3000` | Default context budget (each request can override it) |
| `--core` | empty | Persona or rules placed first in every context |
| `--embedder` | `none` | `model2vec` (needs `nomem-ai[embed]`) or `hashing` |
| `--api-key` | `$NOMEM_API_KEY` | Require `Authorization: Bearer <key>` |
| `--verbose` | off | Log every request |

!!! warning "Exposing it"
    The server is meant for localhost or a private network. If you bind to
    another interface, set `--api-key` and put it behind TLS (a reverse proxy
    such as Caddy or nginx).

Every request takes an optional `user_id` and `session_id` (in the JSON body,
or the query string for `GET`), both defaulting to `"default"`.

### Endpoints

| Method | Path | Body or query | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | | `{"status": "ok", "version": "0.3.0"}` (no auth) |
| `POST` | `/v1/messages` | `{"role", "content", "timestamp"?, "metadata"?}` or `{"messages": [...]}` | The stored message(s) with `id` and any `conflicts` |
| `POST` | `/v1/context` | `{"query"?, "max_tokens"?, "explain"?}` | `{"text", "messages", "tokens_used", "max_tokens", "usage"}`, plus `explain` and `log` if asked |
| `POST` | `/v1/search` | `{"query", "k"?}` | `{"hits": [{id, role, content, created_at, score, reasons}]}` |
| `GET` | `/v1/facts` | `?history=true` | `{"facts": [...], "pinned": {...}}` |
| `POST` | `/v1/facts` | `{"key", "value"}` | The pinned fact |
| `GET` | `/v1/conflicts` | | `{"conflicts": [{older, newer, older_at, newer_at, kind, detail, confidence}]}` |
| `GET` | `/v1/stats` | | `{"messages": n}` for the user and session |

Errors come back as `{"error": "..."}` with status 400, 401, 404 or 413.

### Example

```bash
curl -s localhost:8420/v1/messages -d '{"user_id": "saswat", "content": "I am allergic to peanuts."}'

curl -s localhost:8420/v1/context \
  -d '{"user_id": "saswat", "query": "what am I allergic to?", "max_tokens": 500}'
```

```json
{
  "text": "system: Known facts:\n- allergy: peanuts (since 28 Nov 2026)\nuser: I am allergic to peanuts.",
  "messages": [
    {"role": "system", "content": "Known facts:\n- allergy: peanuts (since 28 Nov 2026)"},
    {"role": "user", "content": "I am allergic to peanuts."}
  ],
  "tokens_used": 31,
  "max_tokens": 500,
  "usage": {"core": 0, "facts": 19, "retrieved": 0, "summary": 0, "recent": 12, "free": 469}
}
```

From Python, any HTTP client works:

```python
import requests

base = "http://localhost:8420"
requests.post(f"{base}/v1/messages", json={"user_id": "u1", "content": "I live in Pune."})
ctx = requests.post(f"{base}/v1/context", json={"user_id": "u1", "query": "where do I live?"}).json()
print(ctx["text"])
```

## MCP server

```bash
pip install "nomem-ai[mcp]"
nomem mcp --db ~/.nomem/memory.db
```

It speaks stdio by default; `--transport streamable-http` (or `sse`) with
`--host` and `--port` serves it over HTTP instead. `--user-id` and
`--session-id` (or `$NOMEM_USER_ID`, `$NOMEM_SESSION_ID`) pick whose memory
it is.

### Tools

| Tool | Arguments | What it does |
| --- | --- | --- |
| `remember` | `content`, `role="user"` | Store text verbatim; returns its id and any conflicts it created |
| `recall` | `query`, `max_tokens=1500` | Dated, budgeted memory for a question, as text |
| `search_memory` | `query`, `k=5` | Ranked hits with scores and the rankers that found them |
| `list_facts` | `history=False` | Extracted and pinned facts |
| `pin_fact` | `key`, `value` | Pin a durable fact |
| `list_conflicts` | | Memories a newer one has changed, with both dates |

The server also sends instructions telling the model to `remember` things
worth keeping and to `recall` before answering questions about the past.

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "nomem": {
      "command": "nomem",
      "args": ["mcp", "--db", "/Users/you/.nomem/memory.db"]
    }
  }
}
```

If `nomem` is not on Claude Desktop's `PATH`, use the full path from
`which nomem`, or `"command": "uvx", "args": ["--from", "nomem-ai[mcp]", "nomem", "mcp", ...]`.

### Claude Code

```bash
claude mcp add nomem -- nomem mcp --db ~/.nomem/memory.db
```

### Cursor and others

Any client that launches a stdio MCP server works with the same command:
`nomem mcp --db <path>`.

## Python: `MemoryService`

Both servers are built on `MemoryService`, which you can use directly to
serve many users from one process, for example inside your own FastAPI or
Flask app:

```python
from nomem import MemoryService

svc = MemoryService("app.db", max_tokens=2000, embedder="none")
svc.add("I moved to Bangalore.", user_id="u1", session_id="chat-7")
svc.context("where do I live?", user_id="u1", session_id="chat-7")   # dict
svc.search("Bangalore", user_id="u1", session_id="chat-7", k=5)       # list of dicts
svc.facts(user_id="u1")
svc.conflicts(user_id="u1")
```

It keeps one `NoMem` per user and session over a shared SQLite store, takes
a lock around every call, and returns plain JSON-ready data.
