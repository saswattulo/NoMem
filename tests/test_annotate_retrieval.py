from datetime import datetime

import pytest

from nomem import ApproxCounter, NoMem
from nomem import annotate as A
from nomem.store import SQLiteStore

REF = datetime(2023, 5, 8, 13, 56).timestamp()


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def mk(**kw) -> NoMem:
    return NoMem(max_tokens=kw.pop("max_tokens", 2000), counter=ApproxCounter(), **kw)


def fts() -> bool:
    return SQLiteStore(":memory:").fts_enabled


# -- annotate -----------------------------------------------------------------

def test_relative_and_absolute_dates_resolve_against_message_time():
    days = lambda text: [(_day(a), _day(b)) for a, b in A.parse_times(text, REF)]
    assert days("I went there yesterday") == [("2023-05-07", "2023-05-07")]
    assert days("We met 2 weeks ago") == [("2023-04-24", "2023-04-30")]
    assert days("It was on 25 April 2023") == [("2023-04-25", "2023-04-25")]
    assert days("see you May 20th") == [("2023-05-20", "2023-05-20")]
    assert days("back in March") == [("2023-03-01", "2023-03-31")]
    assert days("last Friday was fun") == [("2023-05-05", "2023-05-05")]
    assert days("no dates here") == []


def test_slots_are_conservative_first_person_facts():
    s = {(x.attribute, x.value, x.negated) for x in A.extract_slots(
        "My name is Saswat and I live in Bangalore. I'm severely allergic to peanuts.")}
    assert s == {("name", "Saswat", False), ("location", "Bangalore", False),
                 ("allergy", "peanuts", False)}
    assert A.extract_slots("Do you live in Paris?") == []           # questions are not facts
    assert A.extract_slots("I am not allergic to cats.") == []       # negation blocks
    neg = A.extract_slots("I don't like coffee anymore.")
    assert [(x.attribute, x.value, x.negated) for x in neg] == [("likes", "coffee", True)]
    moved = A.extract_slots("I moved to Berlin last month.")
    assert [(x.attribute, x.value) for x in moved] == [("location", "Berlin")]


def test_entities_skip_sentence_fillers_and_speaker_prefix():
    assert set(A.entities("Caroline: Hey Mel! I went to the LGBTQ center in San Francisco")) == {
        "mel", "san francisco", "lgbtq"}


def test_importance_ranks_facts_above_filler():
    assert A.importance("Thanks!") == 0.0
    assert A.importance("I'm allergic to peanuts, remember that.", A.extract_slots(
        "I'm allergic to peanuts")) > A.importance("The weather is ok today.")


def test_near_duplicate_hash():
    a = A.simhash("I really love hiking in the mountains on weekends")
    b = A.simhash("I really love hiking in the mountains on the weekends")
    c = A.simhash("The quarterly report is due next Tuesday")
    assert A.hamming(a, b) < A.hamming(a, c)


# -- write path ---------------------------------------------------------------

def test_timestamp_backdates_a_message():
    mem = mk()
    m = mem.add_user("hello", timestamp=REF)
    assert m.created_at == REF
    assert mem.store.recent_messages("default", "default")[0].created_at == REF


def test_new_single_value_supersedes_old_one_and_keeps_history():
    mem = mk()
    mem.add_user("I live in Paris.", timestamp=REF)
    mem.add_user("I moved to Berlin.", timestamp=REF + 86400)
    current = {(f["attribute"], f["value"]) for f in mem.facts()}
    assert current == {("location", "Berlin")}
    hist = mem.fact_history()
    paris = next(f for f in hist if f["value"] == "Paris")
    assert paris["valid_to"] == REF + 86400 and paris["closed_by"] is not None


def test_multi_value_slots_accumulate_and_negation_closes_one():
    mem = mk()
    mem.add_user("I love coffee.")
    mem.add_user("I love tea.")
    mem.add_user("I don't like coffee anymore.")
    assert {f["value"] for f in mem.facts() if f["attribute"] == "likes"} == {"tea"}


def test_assistant_turns_do_not_write_user_facts():
    mem = mk()
    mem.add_assistant("I love helping with travel plans.")
    assert mem.facts() == []


def test_annotation_can_be_turned_off():
    mem = mk(annotate=False)
    mem.add_user("I live in Paris.")
    assert mem.facts() == []
    assert mem.store.annotations([1]) == {}


# -- retrieval ----------------------------------------------------------------

def test_search_reports_which_rankers_found_each_hit():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk()
    mem.add_user("My dentist appointment got moved.")
    mem.add_user("unrelated chatter about the weather")
    hits = mem.search_hits("when is the dentist appointment?", 3)
    assert hits[0].message.content.startswith("My dentist")
    assert any(r.startswith("keyword#") for r in hits[0].reasons)


def test_search_collapses_exact_duplicates():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk()
    for _ in range(4):
        mem.add_user("the launch code is purple banana")
    mem.add_user("purple is a colour")
    hits = mem.search_hits("purple banana launch code", 5)
    contents = [h.message.content for h in hits]
    assert contents.count("the launch code is purple banana") == 1
    assert any(r.startswith("repeats") for r in hits[0].reasons)


def test_time_ranker_finds_messages_about_a_date():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk(retrieval_top_k=3)
    mem.add_user("We had the team offsite.", timestamp=REF)
    for i in range(20):
        mem.add_user(f"filler message {i}", timestamp=REF + 86400 * (i + 3))
    hits = mem.search_hits("what happened on 8 May 2023 offsite", 3)
    assert hits[0].message.content == "We had the team offsite."
    assert any(r.startswith("time#") for r in hits[0].reasons)


def test_retrieved_memories_carry_their_date():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk(max_tokens=120)
    mem.add_user("I adopted a puppy named Rex.", timestamp=REF)
    for i in range(30):
        mem.add_user(f"unrelated filler number {i}")
    res = mem.build_context(query="what is my puppy called?")
    assert "[8 May 2023] user: I adopted a puppy named Rex." in res.text
