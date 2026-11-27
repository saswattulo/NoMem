"""LongMemEval retrieval benchmark — no LLM anywhere.

`LongMemEval <https://github.com/xiaowu0162/LongMemEval>`_ (ICLR 2025) gives
each of its 500 questions its own chat history ("haystack") of ~50 sessions
and ~500 turns, mixing the sessions that hold the answer with unrelated
ShareGPT / UltraChat sessions. Evidence is labelled at two levels:
``answer_session_ids`` (sessions) and ``has_answer`` (turns).

The official QA score needs an LLM judge, so NoMem uses the benchmark's own
retrieval metrics instead. For every question we load its haystack into a
fresh NoMem, ask the question at the question's own date, and measure:

1. **Turn recall@k** — fraction of the question's evidence turns in NoMem's
   top-k search results.
2. **Session recall@k** — the ranked turns are collapsed to sessions (a
   session ranks where its best turn ranks); fraction of evidence sessions in
   the top-k sessions. ``all@k`` is the share of questions with *every*
   evidence session in the top k (LongMemEval's ``recall_all``).
3. **Context recall** — fraction of evidence turns whose text makes it into
   ``build_context(query=...)`` at the given token budget.

Abstention questions (ids ending ``_abs``) are excluded: their "evidence" is
the false premise, not an answer. Turn metrics skip the few questions that
have no turn-level labels.

The clock is set to each question's ``question_date`` while it is asked, so
"last weekend" or "two months ago" resolve the way the dataset intends and
recency is measured from the question, not from today.

Usage:
    python benchmarks/longmemeval_bench.py [--data PATH] [--limit N] [--budget 1000]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nomem.core  # noqa: E402
from nomem import ApproxCounter, NoMem  # noqa: E402

DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
TYPES = ("single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning")
SEARCH_POOL = 100  # turns fetched per question to rank sessions from


def _parse_date(s: str) -> float:
    # e.g. "2023/05/20 (Sat) 02:21"
    return datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").timestamp()


@contextmanager
def _clock(now: float):
    """Make NoMem's reads see ``now`` as the current time (benchmark only)."""
    real = nomem.core.time
    nomem.core.time = SimpleNamespace(time=lambda: now)
    try:
        yield
    finally:
        nomem.core.time = real


def ingest(mem: NoMem, item: dict) -> tuple[dict, list, list[float]]:
    """Load one haystack; return message id -> session id, evidence messages, latencies."""
    session_of: dict = {}
    evidence = []
    lat: list[float] = []
    for sid, date, turns in zip(item["haystack_session_ids"], item["haystack_dates"],
                                item["haystack_sessions"]):
        base = _parse_date(date)
        for i, turn in enumerate(turns):
            t0 = time.perf_counter()
            msg = mem.add(turn["role"], turn["content"], timestamp=base + i * 30)
            lat.append(time.perf_counter() - t0)
            session_of[msg.id] = sid
            if turn.get("has_answer"):
                evidence.append(msg)
    return session_of, evidence, lat


def select(data: list, limit: int | None) -> list:
    """Answerable questions; with ``limit``, an even spread across the file."""
    items = [x for x in data if not x["question_id"].endswith("_abs")]
    if limit and limit < len(items):
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def run(data_path: Path = DEFAULT_DATA, budget: int = 1000, ks=(5, 10),
        limit: int | None = None, mem_kwargs=None, progress: bool = False) -> dict:
    data = json.loads(Path(data_path).read_text())
    items = select(data, limit)
    mem_kwargs = mem_kwargs or {}
    keys = [f"turn r@{k}" for k in ks] + [f"session r@{k}" for k in ks] \
        + [f"session all@{k}" for k in ks] + ["ctx"]
    per_type: dict = {t: {key: [] for key in keys} for t in TYPES}
    add_lat, search_lat, build_lat, ctx_tokens = [], [], [], []
    n_turns = 0
    t_start = time.perf_counter()

    for qi, item in enumerate(items):
        mem = NoMem(max_tokens=budget, counter=ApproxCounter(),
                    session_id=item["question_id"], **mem_kwargs)
        session_of, evidence, lat = ingest(mem, item)
        add_lat += lat
        n_turns += len(lat)
        row = per_type[item["question_type"]]
        q, now = item["question"], _parse_date(item["question_date"])

        with _clock(now):
            t0 = time.perf_counter()
            hits = mem.search(q, SEARCH_POOL)
            search_lat.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            res = mem.build_context(query=q)
            build_lat.append(time.perf_counter() - t0)
        ctx_tokens.append(res.tokens_used)
        assert res.tokens_used <= budget, "budget exceeded"

        hit_ids = [m.id for m in hits]
        sessions: list = []
        for mid in hit_ids:
            s = session_of[mid]
            if s not in sessions:
                sessions.append(s)
        gold = set(item["answer_session_ids"])
        for k in ks:
            found = len(gold & set(sessions[:k]))
            row[f"session r@{k}"].append(found / len(gold))
            row[f"session all@{k}"].append(float(found == len(gold)))
        if evidence:
            ev_ids = {m.id for m in evidence}
            for k in ks:
                row[f"turn r@{k}"].append(len(ev_ids & set(hit_ids[:k])) / len(ev_ids))
            row["ctx"].append(sum(1 for m in evidence if m.content in res.text) / len(evidence))
        mem.close()
        if progress and (qi + 1) % 50 == 0:
            print(f"  longmemeval {qi + 1}/{len(items)} "
                  f"({time.perf_counter() - t_start:.0f}s)", file=sys.stderr)

    def mean(xs):
        return round(100 * statistics.fmean(xs), 1) if xs else 0.0

    def pct(xs, p):
        xs = sorted(xs)
        return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 3) if xs else 0.0

    return {
        "questions": len(items),
        "turns_ingested": n_turns,
        "budget": budget,
        "overall": {key: mean([v for t in per_type.values() for v in t[key]]) for key in keys},
        "by_type": {t: {key: mean(v) for key, v in d.items()} | {"n": len(d[f"session r@{ks[0]}"])}
                    for t, d in per_type.items()},
        "latency_ms": {
            "add_p50": pct(add_lat, 0.5), "add_p95": pct(add_lat, 0.95),
            "search_p50": pct(search_lat, 0.5), "search_p95": pct(search_lat, 0.95),
            "build_p50": pct(build_lat, 0.5), "build_p95": pct(build_lat, 0.95),
        },
        "avg_context_tokens": round(statistics.fmean(ctx_tokens)) if ctx_tokens else 0,
        "seconds": round(time.perf_counter() - t_start, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--limit", type=int, help="evaluate an even spread of N questions")
    ap.add_argument("--embedder", help="model2vec (potion-base-8M), hashing, or a Model2Vec model id/path")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if not args.data.exists():
        sys.exit(f"LongMemEval data not found at {args.data}; see benchmarks/README.md")
    mem_kwargs = {}
    if args.embedder:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_all import make_embedder
        mem_kwargs["embedder"] = make_embedder(args.embedder)
    out = run(args.data, args.budget, limit=args.limit, mem_kwargs=mem_kwargs, progress=True)
    print(json.dumps(out, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
