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


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        make(max_tokens=0)
    with pytest.raises(ValueError):
        make(archival_ratio=1.5)
    with pytest.raises(ValueError):
        make(archival_ratio=-0.1)
    with pytest.raises(ValueError):
        make(retrieval_top_k=-1)
    with pytest.raises(ValueError):
        make(archival_max_ratio=1.2)


def test_budget_holds_at_full_archival_ratio():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=120, archival_ratio=1.0)
    for i in range(30):
        mem.add_user(f"tokyo trip note number {i}")
    result = mem.build_context(query="tokyo trip note")
    assert result.tokens_used <= 120
    assert len(result.text) <= 120


def test_recent_message_stays_a_conversation_turn():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=1000)  # everything fits in the window
    mem.add_user("pangolins are my favorite animal")
    mem.add_assistant("noted")
    result = mem.build_context(query="pangolins favorite animal")
    # The recent message is presented as a normal conversational turn...
    assert {"role": "user", "content": "pangolins are my favorite animal"} in result.messages
    # ...not relocated into a retrieved-context system block.
    assert not any(
        m["role"] == "system" and "pangolins" in m["content"] for m in result.messages
    )


def test_retrieval_skip_is_logged_when_fts_unavailable():
    mem = make(max_tokens=200)
    mem.add_user("tokyo trip")
    mem.store.fts_enabled = False  # simulate a SQLite build without FTS5
    result = mem.build_context(query="tokyo")
    assert any(
        e.action == "skipped" and e.tier == "archival" and "FTS5" in e.detail
        for e in result.log
    )


def test_close_leaves_a_shared_store_open():
    from nomem.store import SQLiteStore

    store = SQLiteStore(":memory:")
    alice = make(db=store, user_id="alice")
    bob = make(db=store, user_id="bob")
    bob.add_user("still here")
    alice.close()  # must not close the store out from under bob
    assert "still here" in bob.build_context().text
    store.close()


# -- retrieval / recent-window interaction ------------------------------------


_CONVO = [
    ("user", "I want to refactor the billing module before the release."),
    ("assistant", "Sounds good. What is the current pain point there?"),
    ("user", "The invoice generator is coupled to the PDF renderer."),
    ("assistant", "Decoupling those would help. Any deadline?"),
    ("user", "Ship it by Friday. Also the staging DB keeps timing out."),
    ("assistant", "Timeouts often come from connection pool exhaustion."),
    ("user", "Right, let us park that and talk about the release notes."),
    ("assistant", "Happy to help draft them. What changed this cycle?"),
    ("user", "Mostly bug fixes and the new export button."),
    ("assistant", "I will summarise those into three bullets."),
]
_QUERY = "what is coupled to the PDF renderer"


def _kept_turn_numbers(result) -> list[int]:
    """1-based positions of ``_CONVO`` turns rendered as conversation turns."""
    turns = [m["content"] for m in result.messages if m["role"] != "system"]
    return [i for i, (_, text) in enumerate(_CONVO, start=1) if text in turns]


def test_retrieval_never_leaves_a_hole_in_the_recent_window():
    """A retrieved message must not be one the window would have shown anyway.

    Retrieval reserves a slice of the budget but usually underspends it, and
    the rollover grows the recent window past whatever was assumed when
    candidates were filtered. If those two disagree, a message gets quoted in
    the retrieved block *and* skipped as a turn — leaving a gap mid-transcript
    with the missing turn hoisted out of order above it.
    """
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    for budget in range(60, 720, 5):
        mem = make(max_tokens=budget)
        for role, text in _CONVO:
            mem.add(role, text)
        result = mem.build_context(query=_QUERY)
        kept = _kept_turn_numbers(result)
        if kept:
            assert kept == list(range(kept[0], kept[-1] + 1)), (
                f"non-contiguous window at max_tokens={budget}: {kept}"
            )
        assert result.tokens_used <= budget
        assert len(result.text) <= budget


def test_retrieval_stands_down_when_the_match_still_fits_in_the_window():
    """If the window can hold the match, it stays a turn and isn't retrieved."""
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=1000)  # comfortably fits the whole conversation
    for role, text in _CONVO:
        mem.add(role, text)
    result = mem.build_context(query=_QUERY)
    assert _kept_turn_numbers(result) == list(range(1, len(_CONVO) + 1))
    assert not any(
        m["role"] == "system" and "Relevant earlier context" in m["content"]
        for m in result.messages
    )


def test_retrieval_still_fires_for_a_message_outside_the_window():
    """Standing down must not mean never retrieving — the M2 promise holds."""
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=300)
    mem.add_user("chartreuse is my favorite color")
    for i in range(40):
        mem.add_user(f"unrelated filler message number {i:02d}")
    result = mem.build_context(query="what is my favorite color chartreuse")
    assert any(
        m["role"] == "system" and "chartreuse" in m["content"] for m in result.messages
    )
    assert result.tokens_used <= result.max_tokens


def test_build_context_reads_the_message_list_once(monkeypatch):
    """Settling retrieval must stay in memory, not re-scan the store."""
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=200)
    for i in range(30):
        mem.add_user(f"tokyo trip planning note number {i:02d}")

    calls = []
    original = mem.store.recent_messages
    monkeypatch.setattr(
        mem.store,
        "recent_messages",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    mem.build_context(query="tokyo trip planning")
    assert len(calls) == 1


# -- flexible archival slice ---------------------------------------------------

def _long_match_setup(**kw):
    mem = make(max_tokens=400, facts_ratio=0, **kw)
    mem.add_user("the pangolin sanctuary " + "details " * 12 + "end")  # ~120 tokens
    for i in range(40):
        mem.add_user(f"filler chat {i}")
    return mem


def test_long_match_stretches_the_archival_slice():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    long_text = _long_match_setup().store.search("default", "default", "pangolin sanctuary", 1)[0].content
    # Fixed 30% slice (120 tok): the match plus its block header doesn't fit.
    fixed = _long_match_setup(archival_max_ratio=0.3).build_context(query="pangolin sanctuary")
    assert long_text not in fixed.text
    # Default: the slice may stretch to 60%, taking tokens from the window.
    res = _long_match_setup().build_context(query="pangolin sanctuary")
    assert long_text in res.text
    assert res.tokens_used <= 400
    assert any("slice stretched" in e.detail for e in res.log)


def test_stretching_never_displaces_matches_that_fit_the_slice():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    def build(**kw):
        mem = make(max_tokens=400, facts_ratio=0, **kw)
        mem.add_user("pangolin sanctuary " * 9 + "visit")   # long, ranks first
        mem.add_user("pangolin sanctuary opens at nine")     # short
        for i in range(40):
            mem.add_user(f"filler chat {i}")
        return mem.build_context(query="pangolin sanctuary")
    fixed, flex = build(archival_max_ratio=0.3), build()
    assert "pangolin sanctuary opens at nine" in fixed.text
    assert "pangolin sanctuary opens at nine" in flex.text
    assert flex.tokens_used <= 400


def test_budget_holds_when_the_slice_stretches_to_everything():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    mem = make(max_tokens=150, archival_ratio=0.1, archival_max_ratio=1.0)
    for i in range(30):
        mem.add_user(f"tokyo trip note number {i} " + "x" * (i % 7) * 5)
    result = mem.build_context(query="tokyo trip note")
    assert result.tokens_used <= 150
    assert len(result.text) <= 150


def test_no_stretch_for_a_turn_found_only_as_a_neighbour():
    if not _fts_available():
        pytest.skip("FTS5 not available in this SQLite build")
    from nomem import ApproxCounter
    mem = NoMem(max_tokens=120, core="You are a concise travel assistant.", counter=ApproxCounter())
    for role, text in [
        ("user", "Hi! I'm planning a trip to Tokyo."),
        ("assistant", "Exciting! When are you thinking of going?"),
        ("user", "Early March, for about a week, budget around $2000."),
        ("assistant", "Great — cherry blossoms start late March, so you may be a touch early."),
        ("user", "Good to know. What neighborhoods should I stay in?"),
        ("assistant", "Shinjuku and Shibuya are lively; Asakusa is more traditional."),
        ("user", "Let's go with Shinjuku. Any food I must try?"),
    ]:
        mem.add(role, text)
    res = mem.build_context(query="where am I staying and what's my budget?")
    assert not any("slice stretched" in e.detail for e in res.log)
    assert "What neighborhoods should I stay in?" in res.text
