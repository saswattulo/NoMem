"""Framework adapters. Each framework's tests skip when it isn't installed."""

import asyncio
import json

import pytest

from nomem import MemoryService, NoMem
from nomem.integrations import TOOL_NAMES, MemoryToolkit
from nomem.integrations._toolkit import alternate, content_text, memory_text


def run(coro):
    return asyncio.run(coro)


# -- core: forget ---------------------------------------------------------------


def test_forget_removes_message_and_its_indexes():
    mem = NoMem()
    a = mem.add_user("I live in Pune.")
    b = mem.add_user("I moved to Bangalore last week!")
    assert mem.conflicts() and mem.facts()[-1]["value"] == "Bangalore"
    assert mem.forget(b.id) == 1
    assert mem.conflicts() == []
    assert [f["value"] for f in mem.facts() if f["attribute"] == "location"] == ["Pune"]
    assert [m.id for m in mem.search("Bangalore")] == []
    assert mem.forget_session() == 1 and mem.message_count() == 0
    assert a.id not in [m.id for m in mem.search("Pune")]


def test_forget_drops_vector_cache():
    from nomem import HashingEmbedder

    mem = NoMem(embedder=HashingEmbedder())
    m = mem.add_user("quantum chromodynamics lecture notes")
    assert mem.search("chromodynamics")
    mem.forget(m.id)
    assert mem.search("chromodynamics") == []


def test_forget_is_scoped_to_user():
    svc = MemoryService(":memory:")
    a = svc.add("hello from a", user_id="a")
    assert svc.forget(a["id"], user_id="b")["deleted"] == 0
    assert svc.forget(a["id"], user_id="a")["deleted"] == 1
    svc.add("x", user_id="a", session_id="s")
    assert svc.clear(user_id="a", session_id="s")["deleted"] == 1


# -- the neutral toolkit -----------------------------------------------------------


def test_toolkit_tools_and_errors_never_raise():
    kit = MemoryToolkit(NoMem())
    assert kit.names == ["remember", "recall", "search_memory", "pin_fact"]
    out = kit.call("remember", {"content": "My sister Maya lives in Lisbon."})
    assert out["stored"] and out["id"] == 1
    assert "Lisbon" in kit.call("recall", '{"query": "where does Maya live"}')
    assert kit.call("search_memory", {"query": "Lisbon", "k": "2"})[0]["content"].startswith("My sister")
    assert kit.call("pin_fact", {"key": "name", "value": "Saswat"}) == {"key": "name", "value": "Saswat"}
    # Every failure is a readable error for the model, not an exception.
    assert "error" in kit.call("remember", {"content": ""})
    assert "error" in kit.call("recall", {})
    assert "error" in kit.call("recall", "{not json")
    assert "error" in kit.call("recall", {"query": "x", "bogus": 1})
    assert "error" in kit.call("search_memory", {"query": "x", "k": "many"})
    assert "error" in kit.call("forget", {"ids": [1]})  # not exposed by default
    assert "error" in kit.call("remember", {"content": "x", "role": "system"})
    text, err = kit.call_text("recall", {})
    assert err and json.loads(text)["error"]


def test_toolkit_schemas_are_valid_and_clamped():
    kit = MemoryToolkit(NoMem(), tools=TOOL_NAMES, max_k=3)
    for spec in kit:
        assert spec.parameters["type"] == "object"
        assert set(spec.parameters["required"]) <= set(spec.parameters["properties"])
    for i in range(6):
        kit.call("remember", {"content": f"note {i} about gardening"})
    assert len(kit.call("search_memory", {"query": "gardening", "k": 50})) == 3
    assert kit.call("forget", {"ids": [1, 2]}) == {"deleted": 2}
    assert kit.call("list_facts", {"history": "true"})["facts"] == []
    with pytest.raises(ValueError):
        MemoryToolkit(NoMem(), tools=["nope"])


def test_toolkit_on_service_is_scoped():
    svc = MemoryService(":memory:")
    a = MemoryToolkit(svc, user_id="a")
    b = MemoryToolkit(svc, user_id="b")
    a.call("remember", {"content": "I live in Pune."})
    assert "Pune" not in b.call("recall", {"query": "where do I live"})
    assert "Pune" in a.call("recall", {"query": "where do I live"})


def test_recall_budget_does_not_leak_into_memory():
    mem = NoMem(max_tokens=3000)
    kit = MemoryToolkit(mem)
    for i in range(30):
        kit.call("remember", {"content": f"Fact number {i} about the weather in town."})
    kit.call("recall", {"query": "weather", "max_tokens": 64})
    assert mem.max_tokens == 3000


def test_message_helpers():
    assert content_text([{"type": "text", "text": "a"}, {"type": "image"}, {"type": "output_text", "text": "b"}]) == "a\nb"
    turns = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "x"},
             {"role": "user", "content": "y"}, {"role": "tool", "content": "t"},
             {"role": "assistant", "content": "z"}]
    assert alternate(turns) == [{"role": "user", "content": "x\n\ny"}, {"role": "assistant", "content": "z"}]
    text = memory_text([{"role": "system", "content": "Known facts:\n- a"},
                        {"role": "user", "content": "old"}, {"role": "user", "content": "new"}],
                       exclude=["new"])
    assert "Known facts" in text and "- user: old" in text and "new" not in text


# -- Anthropic -----------------------------------------------------------------


def test_anthropic_context_and_tool_round_trip():
    from nomem.integrations.anthropic import AnthropicMemory

    am = AnthropicMemory(NoMem())
    am.add_user([{"type": "text", "text": "I live in Pune."}])
    am.add_assistant({"content": [{"type": "text", "text": "Noted."}]})
    am.add_user("What city do I live in?")
    ctx = am.context(system="Be brief.")
    assert ctx["system"].startswith("Be brief.") and "Pune" in ctx["system"]
    assert ctx["messages"][0]["role"] == "user"
    roles = [m["role"] for m in ctx["messages"]]
    assert all(r1 != r2 for r1, r2 in zip(roles, roles[1:]))

    tools = am.tools(cache_control={"type": "ephemeral"})
    assert {t["name"] for t in tools} == {"remember", "recall", "search_memory", "pin_fact"}
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    response = {"content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "tu_1", "name": "recall", "input": {"query": "where do I live"}},
        {"type": "tool_use", "id": "tu_2", "name": "remember", "input": {}},
        {"type": "tool_use", "id": "tu_3", "name": "web_search", "input": {"q": "x"}},
    ]}
    results = am.handle(response)
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
    assert "Pune" in results[0]["content"] and "is_error" not in results[0]
    assert results[1]["is_error"] is True


def test_anthropic_sdk_objects_and_runner_tools():
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types import Message, ToolUseBlock

    from nomem.integrations.anthropic import AnthropicMemory

    am = AnthropicMemory(NoMem())
    am.add_user("My favourite colour is teal.")
    msg = Message(id="m", type="message", role="assistant", model="claude-sonnet-5-5",
                  stop_reason="tool_use", stop_sequence=None,
                  usage={"input_tokens": 1, "output_tokens": 1},
                  content=[ToolUseBlock(type="tool_use", id="t1", name="recall",
                                        input={"query": "favourite colour"})])
    [res] = run(am.ahandle(msg))
    assert "teal" in res["content"]
    am.record(msg)  # tool-only reply: nothing to store
    assert am.backend.nomem.message_count() == 1
    tools = am.runner_tools()
    assert [t.name for t in tools] == am.toolkit.names
    assert "teal" in tools[1].call({"query": "colour"})
    assert tools[0].to_dict()["input_schema"]["required"] == ["content"]


# -- OpenAI Agents SDK -------------------------------------------------------------


def _scripted_model(steps):
    """A Model that replays ``steps``: a str is a reply, a tuple a tool call."""
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (ResponseFunctionToolCall, ResponseOutputMessage,
                                        ResponseOutputText)

    class Scripted(Model):
        def __init__(self):
            self.inputs = []

        async def get_response(self, system_instructions, input, *args, **kwargs):
            self.inputs.append(input)
            step = steps.pop(0)
            if isinstance(step, tuple):
                name, args_ = step
                out = [ResponseFunctionToolCall(type="function_call", call_id=f"c{len(steps)}",
                                                name=name, arguments=json.dumps(args_))]
            else:
                out = [ResponseOutputMessage(
                    id="msg", type="message", role="assistant", status="completed",
                    content=[ResponseOutputText(type="output_text", text=step, annotations=[])])]
            return ModelResponse(output=out, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    return Scripted()


def test_openai_agents_session_and_tools():
    pytest.importorskip("agents")
    from agents import Agent, RunConfig, Runner, set_tracing_disabled

    set_tracing_disabled(True)

    from nomem.integrations.openai_agents import NoMemSession, memory_tools

    svc = MemoryService(":memory:", max_tokens=2000)
    session = NoMemSession(svc, "chat-1", user_id="saswat")
    model = _scripted_model(["Got it.", ("recall", {"query": "where do I live"}), "You live in Pune."])
    agent = Agent(name="a", instructions="Be helpful.", model=model,
                  tools=memory_tools(svc, user_id="saswat", session_id="chat-1"))

    run(Runner.run(agent, "I live in Pune.", session=session))
    cfg = RunConfig(session_input_callback=session.input_callback)
    result = run(Runner.run(agent, "Where do I live?", session=session, run_config=cfg))
    assert result.final_output == "You live in Pune."

    second = model.inputs[1]
    assert any(i.get("role") == "user" and "I live in Pune" in str(i.get("content")) for i in second)
    tool_out = [i for i in model.inputs[2] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert "Pune" in tool_out[0]["output"]

    stored = [m["content"] for m in svc.search("Pune", user_id="saswat", session_id="chat-1", k=10)]
    assert "I live in Pune." in stored and "You live in Pune." in stored
    items = run(session.get_items())
    assert items[-1] == {"role": "assistant", "content": "You live in Pune."}
    assert run(session.get_items(limit=1)) == items[-1:]
    assert run(session.pop_item()) == {"role": "assistant", "content": "You live in Pune."}
    run(session.clear_session())
    assert run(session.get_items()) == []


def test_openai_agents_tool_errors_are_returned():
    pytest.importorskip("agents")
    from nomem.integrations.openai_agents import memory_tools

    tools = {t.name: t for t in memory_tools(NoMem())}
    out = run(tools["recall"].on_invoke_tool(None, '{"query": ""}'))
    assert "error" in json.loads(out)


# -- LangChain -------------------------------------------------------------------


def test_langchain_history_retriever_tools():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from nomem.integrations.langchain import NoMemChatMessageHistory, NoMemRetriever, memory_tools

    svc = MemoryService(":memory:")
    hist = NoMemChatMessageHistory(svc, "s1", user_id="u")
    hist.add_messages([HumanMessage("I work at Acme as a welder."),
                       AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
                       AIMessage([{"type": "text", "text": "Nice job."}]),
                       SystemMessage("ignored")])
    msgs = hist.messages
    assert [m.type for m in msgs][-2:] == ["human", "ai"] and msgs[-1].content == "Nice job."
    assert run(hist.aget_messages())[-1].content == "Nice job."

    docs = NoMemRetriever(memory=svc, user_id="u", session_id="s1", k=2).invoke("where do I work")
    assert docs[0].page_content == "I work at Acme as a welder." and docs[0].metadata["id"] == 1
    assert NoMemRetriever(memory=svc, user_id="u", session_id="s1").invoke("  ") == []

    tools = {t.name: t for t in memory_tools(svc, user_id="u", session_id="s1")}
    assert "Acme" in tools["recall"].invoke({"query": "job"})
    assert "Acme" in run(tools["recall"].ainvoke({"query": "job"}))
    run(hist.aclear())
    assert svc.stats(user_id="u", session_id="s1")["messages"] == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_langchain_runnable_with_message_history():
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    from nomem.integrations.langchain import NoMemChatMessageHistory

    svc = MemoryService(":memory:")
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("history"), ("human", "{input}")])
    chain = prompt | FakeListChatModel(responses=["Hello!", "Bruno."])
    chat = RunnableWithMessageHistory(
        chain, lambda sid: NoMemChatMessageHistory(svc, sid, user_id="u"),
        input_messages_key="input", history_messages_key="history")
    cfg = {"configurable": {"session_id": "s"}}
    chat.invoke({"input": "My dog is called Bruno."}, cfg)
    chat.invoke({"input": "What is my dog called?"}, cfg)
    assert svc.stats(user_id="u", session_id="s")["messages"] == 4


# -- LangGraph -------------------------------------------------------------------


def test_langgraph_store_crud_search_and_namespaces():
    pytest.importorskip("langgraph")
    from nomem.integrations.langgraph import NoMemStore

    store = NoMemStore()
    store.put(("memories", "u1"), "home", {"content": "I live in Pune", "kind": "place"})
    store.put(("memories", "u1"), "hobby", {"content": "I love hiking on weekends", "kind": "hobby"})
    store.put(("memories", "u2"), "home", {"content": "I live in Oslo"})
    store.put(("docs",), "d", {"title": "Release notes", "body": "Bug fixes"}, index=["title"])

    item = store.get(("memories", "u1"), "home")
    assert item.value == {"content": "I live in Pune", "kind": "place"} and item.key == "home"
    assert store.get(("memories", "u1"), "missing") is None

    hits = store.search(("memories", "u1"), query="where do I live")
    assert hits[0].value["content"] == "I live in Pune" and hits[0].score is not None
    assert all(h.namespace == ("memories", "u1") for h in hits)  # namespaces never mix
    assert {h.namespace for h in store.search(("memories",), query="live")} == {("memories", "u1"), ("memories", "u2")}
    assert [h.key for h in store.search(("memories", "u1"), filter={"kind": "hobby"})] == ["hobby"]
    assert store.search(("docs",), query="release")[0].key == "d"
    assert len(store.search(("memories",), limit=1)) == 1

    created = item.created_at
    store.put(("memories", "u1"), "home", {"content": "I moved to Delhi"})
    again = store.get(("memories", "u1"), "home")
    assert again.value == {"content": "I moved to Delhi"} and again.created_at == created
    assert len(store.search(("memories", "u1"))) == 2

    assert store.list_namespaces() == [("docs",), ("memories", "u1"), ("memories", "u2")]
    assert store.list_namespaces(prefix=("memories",), max_depth=1) == [("memories",)]
    assert store.list_namespaces(suffix=("u2",)) == [("memories", "u2")]
    store.delete(("memories", "u1"), "home")
    assert store.get(("memories", "u1"), "home") is None
    assert run(store.aget(("memories", "u2"), "home")).value["content"] == "I live in Oslo"
    with pytest.raises(ValueError):
        store.put((), "k", {"a": 1})


def test_langgraph_graph_memory_across_threads():
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    from nomem.integrations.langgraph import NoMemGraphMemory

    svc = MemoryService(":memory:")
    gm = NoMemGraphMemory(svc, system="You are helpful.")
    prompts = []

    def model(state, config: RunnableConfig):
        prompts.append(gm.prompt(state, config))
        return {"messages": [AIMessage("Noted.")]}

    g = StateGraph(MessagesState)
    g.add_node("record_in", gm.record)
    g.add_node("model", model)
    g.add_node("record_out", gm.record)
    g.add_edge(START, "record_in")
    g.add_edge("record_in", "model")
    g.add_edge("model", "record_out")
    g.add_edge("record_out", END)
    app = g.compile(checkpointer=InMemorySaver())

    t1 = {"configurable": {"thread_id": "t1", "user_id": "saswat"}}
    app.invoke({"messages": [HumanMessage("My dog is called Bruno.")]}, t1)
    app.invoke({"messages": [HumanMessage("Thanks.")]}, t1)
    # record runs twice per turn over the whole thread, but stores each turn once
    assert svc.stats(user_id="saswat")["messages"] == 4

    t2 = {"configurable": {"thread_id": "t2", "user_id": "saswat"}}
    app.invoke({"messages": [HumanMessage("What is my dog called?")]}, t2)
    system = prompts[-1][0]
    assert system.type == "system" and system.content.startswith("You are helpful.")
    assert "Bruno" in system.content
    assert prompts[-1][1].content == "What is my dog called?"
    # the thread's own turn is not repeated in the system message
    assert "What is my dog called?" not in system.content
    assert svc.stats(user_id="other")["messages"] == 0

    tools = gm.tools(t2)
    assert "Bruno" in {t.name: t for t in tools}["recall"].invoke({"query": "dog"})


# -- LlamaIndex and CrewAI ---------------------------------------------------------


def test_llamaindex_tools_and_memory_block():
    pytest.importorskip("llama_index.core")
    from llama_index.core.base.llms.types import ChatMessage

    from nomem.integrations.llamaindex import NoMemMemoryBlock, memory_tools

    mem = NoMem()
    tools = {t.metadata.name: t for t in memory_tools(mem)}
    tools["remember"].call(content="I live in Pune.")
    assert "Pune" in str(tools["recall"].call(query="where do I live"))
    assert "Pune" in str(run(tools["search_memory"].acall(query="Pune")))

    block = NoMemMemoryBlock(memory=mem)
    run(block.aput([ChatMessage(role="user", content="My cat is Miso."),
                    ChatMessage(role="assistant", content="Cute!"),
                    ChatMessage(role="system", content="skip me")]))
    assert mem.message_count() == 3
    text = run(block.aget([ChatMessage(role="user", content="What is my cat called?")]))
    assert "Miso" in text


def test_crewai_tools():
    pytest.importorskip("crewai")
    from nomem.integrations.crewai import memory_tools

    tools = {t.name: t for t in memory_tools(NoMem())}
    tools["remember"].run(content="The launch is on Friday.")
    assert "Friday" in tools["recall"].run(query="when is the launch")
