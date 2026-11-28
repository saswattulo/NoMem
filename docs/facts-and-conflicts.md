# Facts and contradictions

## Slot facts

When a user says something about themselves, NoMem records it as a *slot
fact*: a subject, an attribute and a value, with the time it became true and
the message that said it.

| Attribute | Example statement | Kind |
| --- | --- | --- |
| `name` | "My name is Saswat." | single value |
| `age` | "I'm 29 years old." | single value |
| `location` | "I live in Pune." / "I moved to Bangalore." | single value |
| `job` | "I work as a backend engineer." | single value |
| `employer` | "I work at Acme." | single value |
| `diet` | "I'm vegetarian." | single value |
| `my <thing>` | "My favourite band is Radiohead." / "My dog is called Max." | single value |
| `allergy` | "I'm allergic to peanuts." | many values |
| `likes` / `dislikes` | "I love hiking." / "I can't stand olives." | many values |

The patterns are deliberately conservative: questions are skipped, negations
("I don't live in...") are not recorded as facts, and assistant turns never
write facts about the user unless a speaker is named. Precision matters more
than recall here, because a wrong fact on the card is worse than a missing
one; the message itself is still stored and searchable either way.

```python
mem.add_user("I live in Pune.")
mem.add_user("I moved to Bangalore last week!")

mem.facts()          # current: location = Bangalore
mem.fact_history()   # every fact, including Pune with valid_to set
mem.facts("user")    # filter by subject
```

A new value for a single-valued attribute closes the old row (`valid_to` and
`closed_by` are set) instead of deleting it, so you can always ask what used
to be true. Saying "I don't like X" closes a `likes = X` fact.

### Pinned facts

For facts the patterns can't know, pin them yourself:

```python
mem.remember("preferred name", "Sas")
mem.remember("timezone", "Asia/Kolkata")
```

Pinned facts join the Known-facts card in every context for that user.

### The Known-facts card

`build_context()` puts a short card after the core persona, query-relevant
facts first, using at most `facts_ratio` of the budget:

```text
Known facts:
- location: Bangalore (since 20 Mar 2026; before that Pune, said 1 Mar 2026)
- job: backend engineer (since 2 Feb 2026)
- preferred name: Sas
```

The card finds facts by meaning, not just by shared words: "what do I do for
a living?" matches `job` with no keyword in common.

## Contradictions

People change their minds, addresses and flight times. NoMem doesn't pick a
winner or rewrite anything. When a newer memory changes an older one, both
are kept, linked as a *conflict*, and shown together with their dates, so the
agent's own model decides what is true now.

Three deterministic detectors run on every write:

| Detector | Catches | Example |
| --- | --- | --- |
| `slot` | A single-valued fact gets a new value, or a like is negated | "I live in Pune" ... "I moved to Bangalore" |
| `frame` | The same sentence with a different number, time, date or name | "I have 2 kids" ... "I have 3 kids" |
| `update` | An update cue ("moved", "postponed", "now", "actually", "instead" ...) plus a shared topic word and a changed value of the same type, from the same speaker | "My flight leaves at 9am" ... "Actually my flight got moved to 11am" |

Recency only decides direction: the later memory is marked as the update.
A retrieved memory then reads:

```text
- [5 Jan 2026] user: My flight leaves at 9am on Friday. [CONFLICT: changed on 25 Jan 2026 (flight: 09:00 -> 11:00); user said "Actually my flight got moved to 11am."]
```

`mem.conflicts()` lists every pair:

```python
[{
  "older_id": 1, "newer_id": 3,
  "older": "My flight leaves at 9am on Friday.",
  "newer": "Actually my flight got moved to 11am.",
  "older_at": 1767571200.0, "newer_at": 1769299200.0,
  "kind": "update", "detail": "flight: 09:00 -> 11:00",
  "confidence": 0.6, "attribute": None, ...
}]
```

With an embedder, vector neighbours add candidates and raise confidence.

### How accurate is it?

The rules aim for precision:

- On real LoCoMo chat they flag 2 pairs in 5,882 turns.
- On a fresh held-out set of 25 updates and 25 look-alike controls, written
  before the latest rule changes and never tuned on, they catch **19 of 25
  with 0 false flags**.

Still missed: numbers written as words ("two cars"), ordinals ("the 7th
floor"), and brand names in lower camel case ("iPhone").

### Tuning

```python
from nomem import NoMem, ConflictConfig

mem = NoMem(conflicts=ConflictConfig(
    min_confidence=0.6,      # weakest link that is stored
    show_min_confidence=0.6, # weakest link shown as a marker
    show_per_memory=2,       # markers per memory (0 hides them)
    update=True,             # the update-cue detector
    frame=True,              # the frame detector
))

NoMem(conflicts=ConflictConfig(enabled=False))  # turn it off
```

All fields are listed in [Configuration](configuration.md#conflictconfig).
