# Using it with an agent

NoMem is framework-neutral: it gives you a list of messages and a string, and
you hand them to whatever model you use. For LangChain, LangGraph, the OpenAI
Agents SDK, LlamaIndex, CrewAI and ready-made Claude helpers, see
[Framework adapters](frameworks.md). The pattern is always the same:

1. Store the user's message.
2. Build a context for it.
3. Call your model with that context.
4. Store the model's reply.

## With the Anthropic SDK

```python
import anthropic
from nomem import NoMem

client = anthropic.Anthropic()
mem = NoMem(max_tokens=4000, user_id="saswat", db="memory.db")

def chat(user_text: str) -> str:
    mem.add_user(user_text)
    ctx = mem.build_context(query=user_text)
    system = "\n".join(m["content"] for m in ctx.messages if m["role"] == "system")
    turns = [m for m in ctx.messages if m["role"] in ("user", "assistant")]
    reply = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        system="You are a helpful assistant.\n\n" + system,
        messages=turns,
    ).content[0].text
    mem.add_assistant(reply)
    return reply
```

The Messages API wants alternating turns starting with `user`; if your
context can start with an assistant turn, drop leading assistant turns or
merge consecutive ones before sending.

## With the OpenAI SDK

```python
from openai import OpenAI
from nomem import NoMem, OpenAICounter

client = OpenAI()
mem = NoMem(max_tokens=4000, counter=OpenAICounter("gpt-4o"), db="memory.db")

def chat(user_text: str) -> str:
    mem.add_user(user_text)
    ctx = mem.build_context(query=user_text)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=ctx.messages,
    ).choices[0].message.content
    mem.add_assistant(reply)
    return reply
```

## Importing an existing transcript

Pass `timestamp=` (epoch seconds) so relative dates resolve correctly and
retrieved memories show the right date:

```python
from datetime import datetime

for turn in transcript:
    ts = datetime.fromisoformat(turn["time"]).timestamp()
    mem.add(turn["role"], turn["text"], timestamp=ts)
```

For multi-party transcripts, put the speaker in `metadata={"speaker": "Caroline"}`
or prefix the text (`"Caroline: ..."`): facts are then recorded for that
speaker, and contradictions are only checked within one speaker.

## Token counters

The budget is only as exact as the counter. Pick the one for your model:

```python
from nomem import NoMem, OpenAICounter, AnthropicCounter, ApproxCounter

NoMem(counter=OpenAICounter("gpt-4o"))   # exact, local (tiktoken)
NoMem(counter=AnthropicCounter())        # exact, one API call per count
NoMem(counter=ApproxCounter())           # no dependency, about 15% off
```

`AnthropicCounter` makes a network call for each count, so for a hot loop
prefer `OpenAICounter` (close enough for most budgets) or leave some headroom
in `max_tokens`. You can also subclass `TokenCounter` and implement
`count(text) -> int`.

## As a tool for the agent

Instead of building the context yourself, you can let the agent decide when
to remember and recall, through the [MCP server](server.md#mcp-server) or the
same tools in your framework (see [Framework adapters](frameworks.md#the-tools)).
