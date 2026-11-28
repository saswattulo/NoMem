"""Score fusion and retrieval hardening."""

import warnings

import pytest

from nomem import ApproxCounter, NoMem
from nomem.embed import Embedder, HashingEmbedder
from nomem.retrieval import RetrievalConfig, _fuse_scores
from nomem.store import SQLiteStore, _fts_query


def mem(**kw):
    return NoMem(counter=ApproxCounter(), **kw)


def test_score_fusion_is_default_and_legacy_is_rrf():
    assert RetrievalConfig().fusion == "score"
    legacy = RetrievalConfig.legacy()
    assert legacy.fusion == "rrf" and legacy.length_penalty == 0 and not legacy.context_same_role
    assert legacy.weights["vector"] == 0.15


@pytest.mark.parametrize("bad", [
    {"fusion": "max"}, {"pool": 0}, {"rrf_k": -1}, {"weights": {"keyword": -1}},
    {"weights": {"vector": "x"}}, {"length_penalty": -0.1}, {"length_pivot": 0},
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        RetrievalConfig(**bad)


def test_fuse_scores_uses_score_gaps_not_just_ranks():
    # Keyword #1 is far ahead of #2; RRF would treat them as near-equal.
    scored = {"keyword": [(1, 10.0), (2, 1.0)], "vector": [(2, 0.3), (1, 0.29)]}
    fused = dict(_fuse_scores(scored, {"keyword": 1.0, "vector": 1.0}, 60))
    assert fused[1] > fused[2]
    # A strong cosine beats a weak keyword-only hit.
    scored = {"keyword": [(1, 10.0), (2, 6.0)], "vector": [(2, 0.9)]}
    fused = dict(_fuse_scores(scored, {"keyword": 1.0, "vector": 1.0}, 60))
    assert fused[2] > fused[1]


def test_fuse_scores_ignores_nan_and_negative():
    scored = {"vector": [(1, float("nan")), (2, -0.5), (3, 0.5)], "entity": [(4, None)]}
    fused = dict(_fuse_scores(scored, {"vector": 1.0, "entity": 0.5}, 60))
    assert fused.get(1, 0) == 0 and fused[2] == 0 and fused[3] > 0 and fused[4] > 0


def test_long_turn_does_not_win_on_size():
    m = mem()
    filler = " ".join(["Here are some general tips about dogs, dog food, dog toys and dog walkers."] * 40)
    m.add("assistant", filler)
    short = m.add("user", "My dog is a golden retriever called Max.")
    assert m.search("what breed is my dog?", 1)[0].id == short.id


def test_search_never_returns_another_sessions_message():
    st = SQLiteStore(":memory:")
    a = mem(db=st, session_id="a")
    a.add("user", "I live in Berlin.")
    b = mem(db=st, session_id="b")
    b.add("user", "hello there")
    hits = b.search_hits("where do I live?", 5)
    assert all("Berlin" not in h.message.content for h in hits)
    assert [h.message.content for h in a.search_hits("where do I live?", 5)][0] == "I live in Berlin."


@pytest.mark.parametrize("q", ["", "   ", "???", "the and of", "a" * 10000, 'he said "AND" OR NOT near(',
                               "NEAR dog*", "émoji 🐶 dog", "\x00dog"])
def test_odd_queries_do_not_raise(q):
    m = mem(embedder=HashingEmbedder())
    for i in range(5):
        m.add("user", f"the dog is named Rex {i}")
    assert len(m.search(q, 5)) <= 5
    m.build_context(query=q)


def test_long_query_is_capped():
    q = " ".join(f"word{i}" for i in range(1000)) + " word5 word5"
    match = _fts_query(q, max_terms=64)
    assert match.count(" OR ") == 63
    assert _fts_query("dog dog dog") == '"dog"'


def test_k_edge_cases():
    m = mem(embedder=HashingEmbedder())
    m.add("user", "the dog is named Rex")
    assert m.search("dog", 0) == [] and m.search("dog", -3) == []
    assert len(m.search("dog", 10**6)) == 1


class Broken(Embedder):
    def encode(self, texts):
        raise RuntimeError("model down")


def test_failing_embedder_keeps_messages_and_keyword_search():
    m = mem(embedder=Broken())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msg = m.add("user", "the dog is named Rex")
        hits = m.search("dog", 5)
    assert [h.id for h in hits] == [msg.id]
    assert any("embedder failed" in str(x.message) for x in w)
    assert any("vector ranker skipped" in str(x.message) for x in w)
    assert m.store.annotations([msg.id])  # annotation still ran


def test_mixed_vector_sizes_are_skipped_not_fatal():
    st = SQLiteStore(":memory:")
    old = mem(db=st, embedder=HashingEmbedder(dim=256))
    for i in range(3):
        old.add("user", f"the dog is named Rex {i}")
    new = mem(db=st, embedder=HashingEmbedder(dim=512))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new.add("user", "the dog sleeps a lot")
        hits = new.search_hits("dog sleeps", 5)
    assert hits and hits[0].message.content == "the dog sleeps a lot"
    assert any("not 512-dimensional" in str(x.message) for x in w)


def test_vector_cache_is_not_shared_by_recycled_store_ids():
    emb = HashingEmbedder()
    for word in ("apple", "banana", "cherry"):
        m = mem(embedder=emb)
        m.add("user", f"I like {word}")
        hits = m.search_hits(f"{word} fruit", 3)
        assert [h.message.content for h in hits] == [f"I like {word}"]
        m.close()
        del m


class NanEmbedder(Embedder):
    def encode(self, texts):
        return [[float("nan")] * 8 for _ in texts]


def test_nan_vectors_do_not_poison_results():
    m = mem(embedder=NanEmbedder())
    msg = m.add("user", "the dog is named Rex")
    assert [h.id for h in m.search("dog", 5)] == [msg.id]


def test_forget_drops_cached_vectors():
    m = mem(embedder=HashingEmbedder())
    gone = m.add("user", "my locker code is 4471")
    m.add("user", "the weather is nice")
    assert gone.id in [h.id for h in m.search("locker code", 5)]  # fills the vector cache
    m.forget([gone.id])
    assert gone.id not in [h.id for h in m.search("locker code", 5)]
    assert gone.id not in [i for i, _ in m.embedder.similar(m.store, m.user_id, m.session_id, "locker code", 5)]


@pytest.mark.parametrize("q", [
    "What was the name of the hotel you recommended in Lisbon?",
    "You mentioned a book about habits, which one was it?",
    "In our previous conversation, what did the list's third item say?",
    "Which ratio have you suggested for the dilution?",
    "I'm going back to our last chat about chess. What was your move?",
])
def test_assistant_cue_fires_on_questions_about_the_assistants_words(q):
    from nomem.annotate import asks_about_assistant
    assert asks_about_assistant(q)


@pytest.mark.parametrize("q", [
    "Can you suggest a good hotel in Lisbon?",
    "Remind me what my sister's name is.",
    "What did I say about my new job?",
    "We talked about my trip, where am I staying?",
    "When did Caroline go to the LGBTQ support group?",
    "What do you recommend for dinner tonight?",
])
def test_assistant_cue_does_not_fire_on_requests_or_user_facts(q):
    from nomem.annotate import asks_about_assistant
    assert not asks_about_assistant(q)


def test_assistant_cue_lifts_the_reply_to_a_matching_question():
    mem = NoMem(counter=ApproxCounter())
    mem.add("user", "Can you recommend a romantic restaurant in Rome for dinner?")
    reply = mem.add("assistant", "For a romantic evening I'd book Roscioli, near Campo de' Fiori. "
                    "It's cosy, the carbonara is famous and the wine list is long. " * 3)
    mem.add("user", "My sister lives in Rome and loves dinner parties.")
    mem.add("user", "We had a lovely dinner at a restaurant last week, very romantic.")
    mem.add("user", "Rome in May is my favourite trip so far.")
    q = "Which romantic restaurant in Rome did you recommend for dinner?"

    def rank(cfg):
        m = NoMem(counter=ApproxCounter(), db=mem.store, retrieval=cfg)
        return [h.id for h in m.search(q, 10)].index(reply.id)

    assert rank(RetrievalConfig()) <= 1
    assert rank(RetrievalConfig(assistant_cue_boost=1.0)) > rank(RetrievalConfig())


def test_assistant_cue_boost_must_be_positive():
    with pytest.raises(ValueError):
        RetrievalConfig(assistant_cue_boost=0)
