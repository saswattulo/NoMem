# Framework adapters

NoMem ships thin adapters for the common agent frameworks. Each one is an
optional extra, so `pip install nomem-ai` stays dependency-free. None of
them calls a model: they only translate between NoMem and the framework.

| Framework | Install | What you get |
| --- | --- | --- |
| Claude API (Anthropic SDK) | `nomem-ai[anthropic]` | Context for `messages.create`, tool definitions, a `tool_use` handler, tool-runner tools |
| OpenAI Agents SDK | `nomem-ai[openai-agents]` | `NoMemSession` (a `Session`), `FunctionTool`s |
| LangChain | `nomem-ai[langchain]` | Chat message history, retriever, `StructuredTool`s |
| LangGraph | `nomem-ai[langgraph]` | `NoMemStore` (a `BaseStore`), graph memory node and prompt |
| LlamaIndex | `nomem-ai[llamaindex]` | `FunctionTool`s, a memory block for `Memory` |
| CrewAI | `nomem-ai[crewai]` | `BaseTool`s |

Every adapter accepts either a `NoMem` (one user and session) or a
`MemoryService` (many users and sessions in one SQLite file, thread safe).

## The tools

All adapters expose the same tools, with the same names and arguments as
the [MCP server](server.md#mcp-server):

| Tool | Default | What it does |
| --- | --- | --- |
| `remember(content, role="user")` | yes | Store text verbatim. The reply lists earlier memories it changes. |
| `recall(query, max_tokens=1500)` | yes | Dated memories and known facts for a question, within a token budget. |
| `search_memory(query, k=5)` | yes | Best matches with ids, scores and the rankers that found them. |
| `pin_fact(key, value)` | yes | A durable fact shown in every recall. |
| `list_facts(history=False)` | no | Extracted and pinned facts. |
| `list_conflicts()` | no | Pairs where a newer memory changes an older one. |
| `forget(ids)` | no | Delete messages by id. Opt-in, because it deletes data. |

Pick a different set with `tools=[...]`. The tools are built to be safe in
an agent loop:

- **They never raise.** Bad or missing arguments, invalid JSON and store
  errors come back as `{"error": "..."}` (with `is_error` set for Claude),
  so the model can read the problem and try again.
- **Arguments are coerced and clamped.** `"5"` works for `k`, `"true"` for
  `history`; `k` and `max_tokens` are capped (`max_k`, `max_recall_tokens`).
- **Unknown arguments are rejected** with the list of valid ones.
- **Every call is thread safe**, and async variants run SQLite work off the
  event loop.

The framework-free version is `MemoryToolkit`, which works with any
tool-calling API:

```python
from nomem import NoMem
from nomem.integrations import MemoryToolkit

kit = MemoryToolkit(NoMem(db="memory.db"), tools=["remember", "recall"])
for spec in kit:              # name, description, JSON Schema, Python function
    print(spec.name, spec.parameters)
kit.call("recall", '{"query": "where do I live"}')   # dict or JSON string
```

## Claude API

```python
import anthropic
from nomem import NoMem
from nomem.integrations.anthropic import AnthropicMemory

client = anthropic.Anthropic()
mem = AnthropicMemory(NoMem(max_tokens=4000, db="memory.db"))

def chat(user_text: str) -> str:
    mem.add_user(user_text)
    ctx = mem.context(system="You are a helpful assistant.")
    reply = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, **ctx)
    mem.add_assistant(reply)
    return reply.content[0].text
```

`context()` puts NoMem's memory blocks into `system` and makes the turns
alternate starting with `user`, as the Messages API requires (consecutive
turns of one role are merged). The recall query defaults to the newest user
message; pass `query=` to choose another.

To let Claude decide when to remember and recall, give it the tools and
run the `tool_use` blocks it returns:

```python
messages = [{"role": "user", "content": "I just moved to Lisbon."}]
while True:
    reply = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                                   tools=mem.tools(), messages=messages)
    if reply.stop_reason != "tool_use":
        break
    messages += [{"role": "assistant", "content": reply.content},
                 {"role": "user", "content": mem.handle(reply)}]
```

`handle()` only answers NoMem's own tools and leaves other tool calls to
you, so it mixes with your other tools. `tools(cache_control={"type":
"ephemeral"})` marks the definitions for prompt caching. With the SDK's tool
runner, use `mem.runner_tools()`:

```python
runner = client.beta.messages.tool_runner(
    model="claude-sonnet-5-5", max_tokens=1024,
    tools=mem.runner_tools(), messages=[{"role": "user", "content": "What's my dog called?"}])
for message in runner:
    ...
```

## OpenAI Agents SDK

`NoMemSession` is a `Session`: the SDK stores every turn in NoMem, and each
run gets NoMem's token-budgeted context (recent turns, retrieved memories and
known facts) instead of the whole log.

```python
from agents import Agent, RunConfig, Runner
from nomem import MemoryService
from nomem.integrations.openai_agents import NoMemSession, memory_tools

svc = MemoryService("memory.db", max_tokens=4000)
session = NoMemSession(svc, "chat-42", user_id="saswat")
agent = Agent(name="Assistant", instructions="Be helpful.",
              tools=memory_tools(svc, user_id="saswat", session_id="chat-42"))

result = await Runner.run(
    agent, "Where do I live?", session=session,
    run_config=RunConfig(session_input_callback=session.input_callback),
)
```

The `session_input_callback` is optional but recommended: the SDK asks a
session for its history before it shows it the new message, so without it
memories are retrieved for the previous user message. With it, they are
retrieved for the message being answered.

Only text messages are stored. Tool calls, tool outputs and reasoning items
are skipped. `pop_item()` deletes the newest stored message and
`clear_session()` the whole session; pinned facts stay.

## LangChain

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from nomem import MemoryService
from nomem.integrations.langchain import NoMemChatMessageHistory, NoMemRetriever, memory_tools

svc = MemoryService("memory.db")

chat = RunnableWithMessageHistory(
    chain,  # e.g. a prompt with MessagesPlaceholder("history") | model
    lambda session_id: NoMemChatMessageHistory(svc, session_id, user_id="saswat"),
    input_messages_key="input", history_messages_key="history",
)

retriever = NoMemRetriever(memory=svc, user_id="saswat", session_id="chat-42", k=5)
docs = retriever.invoke("what did I say about the trip?")

model_with_tools = model.bind_tools(memory_tools(svc, user_id="saswat"))
```

`NoMemChatMessageHistory.messages` is NoMem's context for the session, with
its memory blocks as `SystemMessage`s, so a long conversation never
overflows. Retrieved documents carry `id`, `role`, `created_at`, `score` and
`reasons` in their metadata. `to_messages()` converts any NoMem context to
LangChain messages.

## LangGraph

### Long-term store

`NoMemStore` is a LangGraph `BaseStore`, so it drops in wherever
`InMemoryStore` or `PostgresStore` would:

```python
from langgraph.prebuilt import create_react_agent
from nomem.integrations.langgraph import NoMemStore

store = NoMemStore("memory.db")
store.put(("memories", "saswat"), "home", {"content": "I live in Pune"})
store.search(("memories", "saswat"), query="where do I live")   # NoMem's hybrid search

agent = create_react_agent(model, tools, store=store)
```

Semantic search works without an embedding model or an `index` config: it
uses NoMem's keyword, entity, time and fact rankers (and vectors, if the
`MemoryService` has an embedder). Each item's text is its `content`,
`text`, `memory` or `data` field, the fields named in `index=[...]`, or the
JSON of the value. Each namespace is a separate NoMem user, so facts and
conflicts never leak between namespaces. `filter` supports equality and
`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`. TTLs are not supported.

### Conversation memory

`NoMemGraphMemory` adds NoMem to a graph's model calls. The thread's own
messages stay with LangGraph's checkpointer; NoMem adds facts, dated
memories and turns from earlier threads.

```python
from langgraph.graph import START, END, MessagesState, StateGraph
from nomem import MemoryService
from nomem.integrations.langgraph import NoMemGraphMemory

memory = NoMemGraphMemory(MemoryService("memory.db"), system="You are helpful.")

def call_model(state, config: RunnableConfig):
    return {"messages": [model.invoke(memory.prompt(state, config))]}

graph = StateGraph(MessagesState)
graph.add_node("record_in", memory.record)
graph.add_node("model", call_model)
graph.add_node("record_out", memory.record)
graph.add_edge(START, "record_in")
graph.add_edge("record_in", "model")
graph.add_edge("model", "record_out")
graph.add_edge("record_out", END)
app = graph.compile(checkpointer=checkpointer)

app.invoke({"messages": [("user", "What's my dog called?")]},
           {"configurable": {"thread_id": "t2", "user_id": "saswat"}})
```

`record` stores the human and AI turns that are not stored yet (it checks
message ids), so it is safe to run on every step. With a `MemoryService`,
the user comes from `config["configurable"]["user_id"]` and all of a user's
threads share one memory; pass `per_thread=True` to key it on `thread_id`.
`prompt` also works as `create_react_agent(model, tools, prompt=memory.prompt)`.
`memory.tools(config)` gives LangChain tools for a `ToolNode`.

## LlamaIndex

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from nomem import NoMem
from nomem.integrations.llamaindex import NoMemMemoryBlock, memory_tools

mem = NoMem(db="memory.db")
memory = Memory.from_defaults(session_id="chat-42", token_limit=4000,
                              memory_blocks=[NoMemMemoryBlock(memory=mem)])
agent = FunctionAgent(llm=llm, tools=memory_tools(mem))
await agent.run("Where do I live?", memory=memory)
```

The memory block stores messages that leave LlamaIndex's short-term buffer
and returns facts and dated memories for the current conversation.

## CrewAI

```python
from crewai import Agent
from nomem import NoMem
from nomem.integrations.crewai import memory_tools

researcher = Agent(role="Researcher", goal="...", backstory="...",
                   tools=memory_tools(NoMem(db="memory.db")))
```

!!! note "CrewAI and the OpenAI Agents SDK"
    At the time of writing CrewAI pins `openai<3` and the OpenAI Agents SDK
    needs `openai>=3`, so the two extras cannot share one environment.

## Forgetting

All of this rests on two new core methods: `NoMem.forget(ids)` deletes
messages and everything derived from them (search index, vectors, facts,
conflict links; facts a deleted message had replaced become current
again), and `NoMem.forget_session()` clears a session. `MemoryService` has
`forget(ids, user_id=, session_id=)` and `clear(user_id=, session_id=)`.
