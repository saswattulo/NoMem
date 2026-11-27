"""Stress test: scale, latency and budget invariants — no LLM, no network.

1. **Scale.** Ingest N synthetic messages (filler drawn from a fixed
   vocabulary, plus planted facts at random positions), then measure write
   throughput, query latency and whether each planted fact is retrieved.
2. **Invariants.** Hundreds of randomised configurations (budget, ratios,
   message sizes, queries) checking that the context never exceeds its
   budget, never repeats a message, and keeps the recent window contiguous.

Usage:
    python benchmarks/stress.py [--quick] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nomem import ApproxCounter, NoMem  # noqa: E402

FILLER = (
    "weather coffee meeting walk movie book music garden dinner lunch office "
    "train weekend morning evening friend family project email phone call "
    "park shop market city road river bike game show news story plan idea"
).split()

FACTS = [
    ("I'm severely allergic to peanuts.", "what am I allergic to?", "peanuts"),
    ("My sister Priya lives in Lisbon.", "where does my sister live?", "Lisbon"),
    ("I drive a blue Subaru Outback.", "what car do I drive?", "Subaru"),
    ("My dentist appointment is on the 14th of June.", "when is my dentist appointment?", "dentist"),
    ("I work as a nurse at St Mary's hospital.", "what is my job?", "nurse"),
    ("My cat is called Biscuit.", "what is my cat's name?", "Biscuit"),
    ("I prefer aisle seats on flights.", "which seat do I like on a flight?", "aisle"),
    ("My budget for the trip is 2000 dollars.", "what's my trip budget?", "2000"),
]


# Statements that update each other, so the invariants also cover
# conflict markers on retrieved memories.
UPDATES = [
    "I live in Pune.", "I moved to Bangalore last week!", "My flight leaves at 9am.",
    "Actually my flight got moved to 11am.", "I have 2 kids.", "I have 3 kids now.",
    "The party starts at 8pm.", "Change of plans: the party starts at 9pm instead.",
]


def _filler(rng: random.Random) -> str:
    return " ".join(rng.choice(FILLER) for _ in range(rng.randint(6, 18))).capitalize() + "."


def scale(n: int, seed: int = 0, mem_kwargs=None) -> dict:
    rng = random.Random(seed)
    mem = NoMem(max_tokens=800, counter=ApproxCounter(), core="You are helpful.", **(mem_kwargs or {}))
    positions = sorted(rng.sample(range(n), len(FACTS)))
    plant = dict(zip(positions, FACTS))
    t0 = time.perf_counter()
    for i in range(n):
        if i in plant:
            mem.add_user(plant[i][0])
        else:
            mem.add("user" if i % 2 == 0 else "assistant", _filler(rng))
    ingest_s = time.perf_counter() - t0

    lat, hits = [], 0
    for _ in range(3):
        for fact, q, needle in FACTS:
            t0 = time.perf_counter()
            res = mem.build_context(query=q)
            lat.append(time.perf_counter() - t0)
            assert res.tokens_used <= mem.max_tokens
    for fact, q, needle in FACTS:
        hits += needle in mem.build_context(query=q).text
    lat.sort()
    mem.close()
    return {
        "messages": n,
        "ingest_msgs_per_s": round(n / ingest_s),
        "query_p50_ms": round(1000 * lat[len(lat) // 2], 2),
        "query_p95_ms": round(1000 * lat[int(0.95 * len(lat))], 2),
        "planted_recall": f"{hits}/{len(FACTS)}",
    }


def invariants(trials: int, seed: int = 1, mem_kwargs=None) -> dict:
    rng = random.Random(seed)
    checked = 0
    for t in range(trials):
        max_tokens = rng.randint(20, 600)
        core = rng.choice(["", "You are a concise assistant.", "x " * rng.randint(1, 40)])
        mem = NoMem(
            max_tokens=max_tokens,
            core=core,
            counter=ApproxCounter(),
            retrieval_top_k=rng.randint(0, 8),
            archival_ratio=rng.choice([0.0, 0.1, 0.3, 0.5, 1.0]),
            facts_ratio=rng.choice([0.0, 0.1, 0.3]),
            summary_ratio=rng.choice([0.0, 0.0, 0.1, 0.3]),
            **(mem_kwargs or {}),
        )
        msgs = []
        for i in range(rng.randint(0, 60)):
            if rng.random() < 0.15:
                text = rng.choice(FACTS)[0]
            elif rng.random() < 0.15:
                text = rng.choice(UPDATES)
            elif rng.random() < 0.1 and msgs:
                text = rng.choice(msgs).content  # exact repeats
            else:
                text = _filler(rng) * rng.randint(1, 4)
            msgs.append(mem.add(rng.choice(["user", "assistant"]), text,
                                timestamp=1.7e9 + 3600 * i * rng.random()))
        query = rng.choice([None, "", "peanuts allergy", "sister Lisbon", "zzz", rng.choice(FACTS)[1],
                            "where do I live?", "when is my flight?", "party time"])
        res = mem.build_context(query=query)

        core_tokens = mem.counter.count_message("system", core) if core else 0
        if core_tokens <= max_tokens:
            assert res.tokens_used <= max_tokens, (t, res.tokens_used, max_tokens)
        # Working-window messages are contiguous newest turns, chronological.
        added = [e.ref for e in res.log if e.action == "added" and e.tier == "working"]
        ids_newest_first = [m.id for m in reversed(msgs)]
        retrieved = {e.ref for e in res.log if e.action == "retrieved"}
        window_order = [i for i in ids_newest_first if i not in retrieved]
        assert added == window_order[: len(added)], (t, "window not contiguous")
        # No message appears both as a recent turn and a retrieved memory.
        assert not (set(added) & retrieved), (t, "duplicate message")
        mem.close()
        checked += 1
    return {"invariant_trials": checked, "violations": 0}


def run(quick: bool = False, mem_kwargs=None) -> dict:
    sizes = [1_000, 10_000] if quick else [1_000, 10_000, 50_000]
    return {
        "scale": [scale(n, mem_kwargs=mem_kwargs) for n in sizes],
        "invariants": invariants(200 if quick else 500, mem_kwargs=mem_kwargs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = run(args.quick)
    print(json.dumps(out, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
