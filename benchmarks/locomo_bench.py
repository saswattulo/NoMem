"""LoCoMo retrieval benchmark — no LLM anywhere.

For every answerable LoCoMo question (categories 1-4) we ask NoMem two things:

1. **Search recall@k** — does the evidence turn appear in the top-k results of
   NoMem's archival search?
2. **Context recall** — does the evidence turn's text make it into the final
   token-budgeted context from ``build_context(query=...)``?

Both are averaged per question (fraction of that question's evidence turns
found) and broken down by LoCoMo category.

Usage:
    python benchmarks/locomo_bench.py [--data PATH] [--budget 1000] [--json OUT]
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nomem import ApproxCounter, NoMem  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "locomo10.json"


def _parse_session_time(s: str) -> float:
    # e.g. "1:56 pm on 8 May, 2023"
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").timestamp()


def _searcher(mem: NoMem):
    if hasattr(mem, "search"):
        return lambda q, k: mem.search(q, k)
    return lambda q, k: mem.store.search(mem.user_id, mem.session_id, q, k)


def ingest(mem: NoMem, conv: dict) -> tuple[dict, list[float]]:
    """Load one LoCoMo conversation; return dia_id -> message, and add latencies."""
    supports_ts = "timestamp" in inspect.signature(mem.add).parameters
    by_dia: dict = {}
    lat: list[float] = []
    n = 1
    while f"session_{n}" in conv:
        base = _parse_session_time(conv[f"session_{n}_date_time"])
        for i, turn in enumerate(conv[f"session_{n}"]):
            text = f"{turn['speaker']}: {turn['text']}"
            kwargs = {"timestamp": base + i * 30} if supports_ts else {}
            t0 = time.perf_counter()
            msg = mem.add("user", text, **kwargs)
            lat.append(time.perf_counter() - t0)
            by_dia[turn["dia_id"]] = msg
        n += 1
    return by_dia, lat


def run(data_path: Path = DEFAULT_DATA, budget: int = 1000, ks=(5, 10), mem_kwargs=None) -> dict:
    data = json.loads(Path(data_path).read_text())
    mem_kwargs = mem_kwargs or {}
    per_cat: dict = {c: {f"r@{k}": [] for k in ks} | {"ctx": []} for c in CATEGORIES}
    add_lat, search_lat, build_lat = [], [], []
    ctx_tokens = []

    for conv_i, sample in enumerate(data):
        mem = NoMem(max_tokens=budget, counter=ApproxCounter(),
                    session_id=f"conv{conv_i}", **mem_kwargs)
        by_dia, lat = ingest(mem, sample["conversation"])
        add_lat += lat
        search = _searcher(mem)
        for qa in sample["qa"]:
            cat = qa.get("category")
            if cat not in CATEGORIES:
                continue
            ev = [by_dia[e] for e in qa.get("evidence", []) if e in by_dia]
            if not ev:
                continue
            ev_ids = {m.id for m in ev}
            q = qa["question"]

            t0 = time.perf_counter()
            hits = search(q, max(ks))
            search_lat.append(time.perf_counter() - t0)
            hit_ids = [m.id for m in hits]
            for k in ks:
                found = len(ev_ids & set(hit_ids[:k]))
                per_cat[cat][f"r@{k}"].append(found / len(ev_ids))

            t0 = time.perf_counter()
            res = mem.build_context(query=q)
            build_lat.append(time.perf_counter() - t0)
            ctx_tokens.append(res.tokens_used)
            assert res.tokens_used <= budget, "budget exceeded"
            found = sum(1 for m in ev if m.content in res.text)
            per_cat[cat]["ctx"].append(found / len(ev))
        mem.close()

    def mean(xs):
        return round(100 * statistics.fmean(xs), 1) if xs else 0.0

    def pct(xs, p):
        xs = sorted(xs)
        return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 3)

    overall = {key: mean([v for c in per_cat.values() for v in c[key]]) for key in per_cat[1]}
    return {
        "questions": sum(len(c["ctx"]) for c in per_cat.values()),
        "budget": budget,
        "overall": overall,
        "by_category": {CATEGORIES[c]: {k: mean(v) for k, v in d.items()} for c, d in per_cat.items()},
        "latency_ms": {
            "add_p50": pct(add_lat, 0.5), "add_p95": pct(add_lat, 0.95),
            "search_p50": pct(search_lat, 0.5), "search_p95": pct(search_lat, 0.95),
            "build_p50": pct(build_lat, 0.5), "build_p95": pct(build_lat, 0.95),
        },
        "avg_context_tokens": round(statistics.fmean(ctx_tokens)) if ctx_tokens else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if not args.data.exists():
        sys.exit(f"LoCoMo data not found at {args.data}; see benchmarks/README.md")
    out = run(args.data, args.budget)
    print(json.dumps(out, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
