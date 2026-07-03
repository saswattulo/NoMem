"""Archival retrieval: recall a fact the sliding window already dropped.

An important detail is mentioned early, buried under later chit-chat, then
pulled back into context on demand when a query matches it.

Note: v0.1 retrieval is SQLite FTS5 *keyword* search, so the query has to
share a term with the stored fact (here, "peanut"). Semantic retrieval is a
later upgrade behind the same interface.

Run it:  python examples/retrieval_demo.py   (no API key, no model download)
"""

from nomem import NoMem

mem = NoMem(max_tokens=160, core="You are a helpful food assistant.")

# An important fact, stated early...
mem.add_user("Important: I'm severely allergic to peanuts.")

# ...then buried under lots of unrelated small talk.
chitchat = [
    "The weather has been lovely this week.",
    "I watched a great movie last night.",
    "My commute was smooth this morning.",
    "I reorganized my bookshelf yesterday.",
    "Thinking about repainting the kitchen.",
    "The neighbor's dog is adorable.",
    "I finally fixed my squeaky door.",
    "Coffee tastes better on weekends.",
    "I started a new book this week.",
    "The park was busy on Sunday.",
]
for line in chitchat:
    mem.add_user(line)
    mem.add_assistant("Nice!")

print("=== WITHOUT a query — the allergy has slid out of the window ===")
plain = mem.build_context()
print("allergy still in context" if "peanut" in plain.text else "allergy NOT in context (dropped)")
print()

print("=== WITH a matching query — retrieval brings it back ===")
result = mem.build_context(query="what snacks are safe given my peanut allergy?")
print(result.text)
print()
print("=== TRANSPARENCY LOG ===")
print(result.explain())
