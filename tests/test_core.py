import pytest

from nomem import NoMem
from nomem.tokens import TokenCounter


class CharCounter(TokenCounter):
    """Deterministic 1-char-per-token counter for exact budget assertions.

    ``per_message_overhead`` of 1 accounts for the newline that joins rendered
    messages, so ``len(text) <= tokens_used`` holds (mirrors why the real
    counters carry an overhead).
    """

    per_message_overhead = 1

    def count(self, text: str) -> int:
        return len(text)


def _fts_available() -> bool:
    from nomem.store import SQLiteStore

    return SQLiteStore(":memory:").fts_enabled


def make(max_tokens=100, core="", **kw) -> NoMem:
    return NoMem(max_tokens=max_tokens, core=core, counter=CharCounter(), **kw)


def test_add_and_build_basic():
    mem = make(max_tokens=1000)
    mem.add_user("hello")
    mem.add_assistant("hi there")
    result = mem.build_context()
    assert result.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert "hello" in result.text and "hi there" in result.text
    assert result.tokens_used <= result.max_tokens


def test_budget_never_exceeded():
    mem = make(max_tokens=50)
    for i in range(200):
        mem.add_user(f"message number {i} with some filler text")
    result = mem.build_context()
    assert result.tokens_used <= 50
    # And the concrete rendered text also respects the budget under CharCounter.
    assert len(result.text) <= 50


def test_sliding_window_keeps_most_recent():
    mem = make(max_tokens=40)
    mem.add_user("OLDEST-oldest-oldest-oldest")
    mem.add_user("MIDDLE-middle-middle-middle")
    mem.add_user("NEWEST-newest-newest-newest")
    result = mem.build_context()
    assert "NEWEST" in result.text
    assert "OLDEST" not in result.text  # dropped to honor the budget


def test_kept_messages_stay_chronological():
    mem = make(max_tokens=1000)
    mem.add_user("first")
    mem.add_assistant("second")
    mem.add_user("third")
    contents = [m["content"] for m in mem.build_context().messages]
    assert contents == ["first", "second", "third"]


def test_core_is_always_included_and_counted():
    mem = make(max_tokens=1000, core="You are a pirate. Always say arr.")
    mem.add_user("hello")
    result = mem.build_context()
    assert result.messages[0] == {"role": "system", "content": "You are a pirate. Always say arr."}
    assert "pirate" in result.text


def test_core_larger_than_budget_is_surfaced_not_hidden():
    big_core = "x" * 500
    mem = make(max_tokens=50, core=big_core)
    mem.add_user("hello")
    result = mem.build_context()
    # Core is preserved (we don't silently drop the persona)...
    assert big_core in result.text
    # ...and the violation is logged transparently.
    assert any(e.action == "budget_exhausted" and e.tier == "core" for e in result.log)


def test_log_records_added_and_dropped():
    mem = make(max_tokens=30)
    for i in range(10):
        mem.add_user(f"msg-{i}-padding-padding")
    mem.build_context()
    actions = {e.action for e in mem.log}
    assert "added" in actions
    assert "dropped" in actions
    assert "budget_exhausted" in actions


def test_persistence_across_instances(tmp_path):
    db = str(tmp_path / "mem.db")
    mem = make(max_tokens=1000, db=db)
    mem.add_user("remember this")
    mem.close()

    reopened = make(max_tokens=1000, db=db)
    result = reopened.build_context()
    assert "remember this" in result.text
    assert reopened.message_count() == 1


def test_user_and_session_scoping_isolate_memory():
    db_store = None  # share one in-memory DB across both handles
    from nomem.store import SQLiteStore

    db_store = SQLiteStore(":memory:")
    alice = make(db=db_store, user_id="alice", session_id="s1")
    bob = make(db=db_store, user_id="bob", session_id="s1")
    alice.add_user("alice secret")
    bob.add_user("bob secret")

    assert "alice secret" in alice.build_context().text
    assert "alice secret" not in bob.build_context().text
    assert "bob secret" in bob.build_context().text


def test_fts_archival_search_groundwork():
    from nomem.store import SQLiteStore

    store = SQLiteStore(":memory:")
    if not store.fts_enabled:
        pytest.skip("FTS5 not available in this SQLite build")
    store.add_message("u", "s", "user", "I am flying to Tokyo in March")
    store.add_message("u", "s", "user", "The weather is nice today")
    hits = store.search("u", "s", "tokyo trip", k=5)
    assert any("Tokyo" in h.content for h in hits)


# -- M2: archival retrieval ---------------------------------------------------


def test_retrieval_recalls_a_message_dropped_from_the_window():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=300)
    mem.add_user("chartreuse is my favorite color")
    for i in range(40):
        mem.add_user(f"unrelated filler message number {i:02d}")

    # Without a query the early fact is pushed out of the recent window.
    assert "chartreuse" not in mem.build_context().text

    # A matching query pulls it back in via the archival tier.
    result = mem.build_context(query="what is my favorite color chartreuse")
    assert "chartreuse" in result.text
    assert result.tokens_used <= result.max_tokens
    assert any(e.action == "retrieved" and e.tier == "archival" for e in result.log)


def test_retrieval_stays_within_budget():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=150)
    for i in range(60):
        mem.add_user(f"tokyo trip planning note number {i}")
    result = mem.build_context(query="tokyo trip planning")
    assert result.tokens_used <= 150
    assert len(result.text) <= 150


def test_retrieval_does_not_duplicate_a_recent_message():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=1000)  # everything fits in the window
    mem.add_user("pangolins are my favorite animal")
    mem.add_assistant("noted")
    result = mem.build_context(query="pangolins favorite animal")
    assert result.text.count("pangolins are my favorite animal") == 1


def test_no_query_means_no_archival_work():
    mem = make(max_tokens=1000)
    mem.add_user("hello world")
    result = mem.build_context()
    assert not any(e.tier == "archival" for e in result.log)
