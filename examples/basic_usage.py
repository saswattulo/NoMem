"""NoMem in ~15 lines: a hard token budget you can watch enforce itself.

Run it:  python examples/basic_usage.py
No API key, no model download, no external services.
"""

from nomem import NoMem

# A tiny budget so the sliding window is easy to see in action.
mem = NoMem(
    max_tokens=120,
    core="You are a concise travel assistant.",
    user_id="saswat",
    session_id="trip-planning",
)

conversation = [
    ("user", "Hi! I'm planning a trip to Tokyo."),
    ("assistant", "Exciting! When are you thinking of going?"),
    ("user", "Early March, for about a week, budget around $2000."),
    ("assistant", "Great — cherry blossoms start late March, so you may be a touch early."),
    ("user", "Good to know. What neighborhoods should I stay in?"),
    ("assistant", "Shinjuku and Shibuya are lively; Asakusa is more traditional."),
    ("user", "Let's go with Shinjuku. Any food I must try?"),
]

for role, text in conversation:
    mem.add(role, text)

result = mem.build_context(query="where am I staying and what's my budget?")

print("=== ASSEMBLED CONTEXT ===")
print(result.text)
print()
print("=== TRANSPARENCY LOG ===")
print(result.explain())
