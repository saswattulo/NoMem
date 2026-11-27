"""Contradiction handling: newer memories that change older ones are linked, not resolved."""

from datetime import datetime

import pytest

from nomem import ApproxCounter, ConflictConfig, HashingEmbedder, NoMem
from nomem import annotate as A
from nomem import conflicts as C
from nomem.store import SQLiteStore

DAY = 86400
T0 = datetime(2026, 1, 5, 9, 0).timestamp()
FILLER = ["The weather was lovely today.", "I watched a documentary about whales.",
          "Work has been busy lately.", "Coffee first, then emails.", "The game last night was close."]


def mk(**kw) -> NoMem:
    return NoMem(max_tokens=kw.pop("max_tokens", 400), counter=ApproxCounter(), **kw)


def fts() -> bool:
    return SQLiteStore(":memory:").fts_enabled


def chat(mem: NoMem, old: str, new: str, gap_days: int = 20, filler: int = 30):
    t = T0
    a = mem.add_user(old, timestamp=t)
    for i in range(filler):
        t += 600
        mem.add_user(FILLER[i % len(FILLER)] + f" ({i})", timestamp=t)
    b = mem.add_user(new, timestamp=T0 + gap_days * DAY)
    for i in range(filler):
        mem.add_user(FILLER[i % len(FILLER)] + f" [{i}]", timestamp=T0 + gap_days * DAY + 600 * (i + 1))
    return a, b


def linked(mem, a, b):
    return [c for c in mem.conflicts() if (c["older_id"], c["newer_id"]) == (a.id, b.id)]


# -- detection -----------------------------------------------------------------

def test_slot_update_is_linked_with_both_texts_and_dates():
    mem = mk()
    a, b = chat(mem, "I live in Pune.", "I moved to Bangalore last week!")
    [c] = linked(mem, a, b)
    assert c["kind"] == "slot" and c["detail"] == "location: Pune -> Bangalore"
    assert c["older"] == "I live in Pune." and c["newer"].startswith("I moved to Bangalore")
    assert c["older_at"] < c["newer_at"] and c["confidence"] >= 0.9
    assert mem.conflicts(a.id) == mem.conflicts(b.id) == [c]


def test_same_frame_with_a_different_number_is_a_conflict():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk()
    a, b = chat(mem, "I have 2 kids.", "I have 3 kids.")
    [c] = linked(mem, a, b)
    assert c["kind"] == "frame" and c["detail"] == "2 -> 3"


@pytest.mark.parametrize("old,new,detail", [
    ("My flight leaves at 9am on Friday.", "Actually my flight got moved to 11am.", "flight: 09:00 -> 11:00"),
    ("The wedding is on June 5th.", "The wedding has been postponed to August 12th.", "wedding: 2026-06-05 -> 2026-08-12"),
    ("I drive a Honda Civic.", "I sold the Honda, I now drive a Tesla.", "drive: honda civic -> tesla"),
])
def test_update_cue_with_a_shared_topic_and_new_value_is_a_conflict(old, new, detail):
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk()
    a, b = chat(mem, old, new)
    [c] = linked(mem, a, b)
    assert c["kind"] == "update" and c["detail"] == detail


@pytest.mark.parametrize("x,y", [
    ("I ran 5 km this morning.", "I ran 10 km today, new record!"),          # two runs, not a state
    ("I have a dog named Rex.", "I went to the park with Rex yesterday."),    # no change
    ("My sister lives in Lisbon.", "My brother lives in Madrid."),           # different people
    ("The meeting is at 3pm.", "Lunch is at 1pm."),                           # different topics
    ("I visited Paris in 2019.", "Now I'm visiting Rome, finally!"),          # past event vs plan
    ("I like sushi.", "I also like ramen."),                                  # multi-valued
])
def test_lookalikes_are_not_flagged(x, y):
    mem = mk()
    a, b = chat(mem, x, y)
    assert not linked(mem, a, b)


def test_different_speakers_never_conflict():
    mem = mk()
    a, b = chat(mem, "Caroline: The party starts at 8pm.", "Melanie: Actually the party starts at 9pm now.")
    assert not linked(mem, a, b)


def test_an_earlier_timestamp_added_later_is_not_marked_newer():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk()
    mem.add_user("My flight leaves at 9am.", timestamp=T0 + 10 * DAY)
    late_import = mem.add_user("Actually my flight got moved to 11am.", timestamp=T0)
    assert not any(c["newer_id"] == late_import.id for c in mem.conflicts())


def test_correction_prefix_is_not_a_speaker_name():
    assert A.speaker_of("Correction: my cat is called Muffin.") is None
    assert A.speaker_of("Caroline: hi") == "Caroline"


def test_generic_slot_opinions_are_not_shown_as_conflicts():
    mem = mk()
    a, b = chat(mem, "My family is my motivation.", "My family is awesome to me.")
    assert not linked(mem, a, b)
    mem = mk()
    a, b = chat(mem, "My address is 12 Baker Street.", "My new address is 5 Elm Road.")
    [c] = linked(mem, a, b)
    assert c["detail"] == "address: 12 Baker Street -> 5 Elm Road"


def test_disabled_detection_records_and_shows_nothing():
    mem = mk(conflicts=ConflictConfig(enabled=False))
    chat(mem, "I live in Pune.", "I moved to Bangalore last week!")
    assert mem.conflicts() == []
    assert "CONFLICT" not in mem.build_context(query="where do I live? Pune").text
    assert mem.facts()[-1]["value"] == "Bangalore"  # supersession itself still works


def test_works_with_an_embedder():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk(embedder=HashingEmbedder())
    a, b = chat(mem, "The party starts at 8pm.", "Change of plans: the party starts at 9pm instead.")
    assert linked(mem, a, b)


# -- surfacing ---------------------------------------------------------------

def test_retrieving_the_newer_memory_quotes_the_older_one_with_its_date():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk(facts_ratio=0)
    chat(mem, "My flight leaves at 9am on Friday.", "Actually my flight got moved to 11am.", filler=40)
    mem.add_user("The flight number is BA117.", timestamp=T0 + 30 * DAY)  # retrievable, unrelated value
    for i in range(30):
        mem.add_user(f"filler after {i}", timestamp=T0 + 31 * DAY + i)
    res = mem.build_context(query="when does my flight leave?")
    assert "CONFLICT" in res.text
    assert "9am" in res.text and "11am" in res.text
    assert "5 Jan 2026" in res.text and "25 Jan 2026" in res.text
    assert any(e.action == "conflict" for e in res.log)


def test_both_sides_retrieved_point_at_each_other_without_requoting():
    if not fts():
        pytest.skip("FTS5 not available")
    mem = mk(facts_ratio=0)
    chat(mem, "The wedding is on June 5th.", "The wedding has been postponed to August 12th.")
    text = mem.build_context(query="when is the wedding?").text
    assert "see the 25 Jan 2026 memory in this list" in text
    assert "see the 5 Jan 2026 memory in this list" in text


def test_state_card_shows_the_value_it_replaced():
    mem = mk()
    chat(mem, "I live in Pune.", "I moved to Bangalore last week!")
    text = mem.build_context(query="where do I live?").text
    assert "location: Bangalore (since 25 Jan 2026; before that Pune, said 5 Jan 2026)" in text


def test_conflict_markers_respect_the_budget():
    for budget in (40, 60, 90, 140, 200):
        mem = NoMem(max_tokens=budget, counter=ApproxCounter(), core="Be brief.")
        chat(mem, "I live in Pune near the old fort and the river.",
             "I moved to Bangalore last week, the flat is near Indiranagar metro!")
        res = mem.build_context(query="where do I live? Pune or Bangalore")
        assert res.tokens_used <= budget


def test_show_per_memory_zero_hides_markers_but_keeps_links():
    mem = mk(facts_ratio=0, conflicts=ConflictConfig(show_per_memory=0))
    a, b = chat(mem, "I live in Pune.", "I moved to Bangalore last week!")
    assert linked(mem, a, b)
    assert "CONFLICT" not in mem.build_context(query="Pune").text


# -- pieces -------------------------------------------------------------------

def test_values_and_frames():
    v = C.sentence_values("Wow, Dave, the party starts at 9pm on 12 June", T0)
    assert v.time == {"21:00"} and v.date == {"2026-06-12"} and "dave" not in v.ent
    assert C.frame("I have 2 kids.") == C.frame("I have 3 kids now.")
    assert C.frame("I ran 5 km.") is None  # an event, not a state
    assert C.frame("Is it at 5pm?") is None
