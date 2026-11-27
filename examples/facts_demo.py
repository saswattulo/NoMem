"""Facts without an LLM: extraction, supersession, and a state card.

NoMem pulls first-person facts out of what the user says with deterministic
rules, keeps their history (a new value closes the old one instead of
deleting it), and puts the facts a question asks about into a small
"Known facts" card, even when the question shares no keyword with the
original message.

Run it:  python examples/facts_demo.py   (no API key, no model download)
"""

from datetime import datetime

from nomem import ApproxCounter, NoMem

day = lambda d: datetime(2026, 3, d, 9, 0).timestamp()

mem = NoMem(max_tokens=260, core="You are a helpful assistant.", counter=ApproxCounter())

mem.add_user("Hi! My name is Saswat and I live in Pune.", timestamp=day(1))
mem.add_user("I work as a backend engineer at Acme.", timestamp=day(1))
mem.add_user("I'm allergic to shellfish, remember that.", timestamp=day(2))
for i in range(25):
    mem.add_user(f"Some unrelated planning chatter, item {i}.", timestamp=day(3))
mem.add_user("Big news: I moved to Bangalore last week!", timestamp=day(20))

print("=== Current facts ===")
for f in mem.facts():
    print(f"  {f['attribute']}: {f['value']}")

print("\n=== History (nothing deleted) ===")
for f in mem.fact_history():
    status = "current" if f["valid_to"] is None else f"superseded {datetime.fromtimestamp(f['valid_to']):%d %b}"
    print(f"  {f['attribute']}: {f['value']}  [{status}]")

print("\n=== 'what do I do for a living?' (no shared keyword with the fact) ===")
result = mem.build_context(query="what do I do for a living?")
print(result.text)

print("\n=== 'remind me about Pune' (an old fact, flagged as superseded) ===")
result = mem.build_context(query="remind me about Pune")
print(result.text)
print()
print(result.explain())
