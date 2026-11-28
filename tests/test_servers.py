import asyncio
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from nomem import MemoryService
from nomem.http_api import make_handler


def test_service_users_and_sessions_are_separate():
    svc = MemoryService(":memory:")
    svc.add("I live in Pune.", user_id="a", session_id="s1")
    svc.add("I live in Oslo.", user_id="b", session_id="s1")
    assert [h["content"] for h in svc.search("where do I live", user_id="a", session_id="s1")] == ["I live in Pune."]
    assert svc.stats(user_id="a", session_id="s2")["messages"] == 0
    facts = svc.facts(user_id="a")["facts"]
    assert any(f["attribute"] == "location" and f["value"] == "Pune" for f in facts)


def test_service_reports_conflicts_on_write():
    svc = MemoryService(":memory:")
    svc.add("I live in Pune.")
    out = svc.add("I moved to Bangalore last week!")
    assert out["conflicts"] and out["conflicts"][0]["older"] == "I live in Pune."
    assert len(svc.conflicts()) == 1


def test_service_context_budget_and_validation():
    svc = MemoryService(":memory:", max_tokens=500)
    for i in range(40):
        svc.add(f"Message number {i} about the weather in town.")
    ctx = svc.context("weather", max_tokens=60, explain=True)
    assert ctx["tokens_used"] <= 60 and ctx["max_tokens"] == 60
    assert "explain" in ctx and ctx["log"]
    with pytest.raises(ValueError):
        svc.add("")
    with pytest.raises(ValueError):
        svc.add("hi", role="robot")


@pytest.fixture
def api():
    svc = MemoryService(":memory:")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc, api_key="k", version="t"))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None, key="k"):
        req = urllib.request.Request(base + path, method=method,
                                     data=None if body is None else json.dumps(body).encode())
        req.add_header("Content-Type", "application/json")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    yield call
    httpd.shutdown()
    httpd.server_close()


def test_http_api_round_trip(api):
    assert api("GET", "/health", key=None) == (200, {"status": "ok", "version": "t"})
    assert api("GET", "/v1/stats", key=None)[0] == 401
    assert api("GET", "/v1/stats", key="wrong")[0] == 401
    st, out = api("POST", "/v1/messages", {"messages": [
        {"role": "user", "content": "My flight leaves at 9am on Friday."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Actually my flight got moved to 11am."},
    ], "user_id": "u"})
    assert st == 201 and len(out["messages"]) == 3
    st, out = api("POST", "/v1/context", {"query": "when is my flight?", "user_id": "u", "max_tokens": 200})
    assert st == 200 and "11am" in out["text"] and out["tokens_used"] <= 200
    st, out = api("POST", "/v1/search", {"query": "flight", "user_id": "u", "k": 2})
    assert st == 200 and len(out["hits"]) == 2 and out["hits"][0]["reasons"]
    assert api("POST", "/v1/facts", {"key": "name", "value": "Sas", "user_id": "u"})[0] == 201
    assert api("GET", "/v1/facts?user_id=u")[1]["pinned"] == {"name": "Sas"}
    assert len(api("GET", "/v1/conflicts?user_id=u")[1]["conflicts"]) == 1
    assert api("GET", "/v1/stats?user_id=u")[1]["messages"] == 3


def test_http_api_errors(api):
    assert api("POST", "/v1/messages", {"content": ""})[0] == 400
    assert api("POST", "/v1/messages", {"content": "x", "role": "robot"})[0] == 400
    assert api("GET", "/v1/nope")[0] == 404


def test_mcp_server_tools():
    pytest.importorskip("mcp")
    from nomem.mcp_server import build_server

    srv = build_server(MemoryService(":memory:"))

    async def go():
        names = {t.name for t in await srv.list_tools()}
        assert {"remember", "recall", "search_memory", "list_facts", "pin_fact", "list_conflicts"} <= names
        await srv.call_tool("remember", {"content": "I am allergic to peanuts."})
        res = await srv.call_tool("recall", {"query": "what am I allergic to?"})
        assert "peanuts" in str(res)

    asyncio.run(go())


def test_cli_version(capsys):
    from nomem.cli import main
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "nomem" in capsys.readouterr().out
