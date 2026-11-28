"""Run Mem0 (mem0ai, open-source) through the same LoCoMo and LongMemEval harness.

This is a measured, like-for-like baseline: the same data, the same questions,
the same evidence labels and the same metrics as ``locomo_bench.py`` and
``longmemeval_bench.py``. Only the memory layer changes.

Two Mem0 modes:

* ``--infer off`` (default, no API key): Mem0 stores each message verbatim with
  ``add(..., infer=False)`` and answers with its own ``search``. No LLM is
  called. This measures Mem0's retrieval stack on raw turns, which is what
  NoMem does too, so the metrics line up exactly.
* ``--infer on``: Mem0's real pipeline, where an LLM extracts facts on every
  ``add``. Needs ``OPENAI_API_KEY`` (or another provider configured in
  ``--llm``). Extracted facts are not turns, so turn/session recall is scored
  by mapping each fact back to the message it was extracted from, and context
  recall by checking whether a fact from an evidence turn reached the context.
  This mode costs money and takes hours on LongMemEval; it is opt-in.

With ``infer=False`` Mem0's search is still its own hybrid: cosine on the
local Qdrant store plus Qdrant's BM25 index over spaCy-lemmatised text, fused
by normalised score addition. Its third signal, entity boosts, is filled only
by the LLM pipeline, so it is off here. Install ``mem0ai[nlp]`` and
``python -m spacy download en_core_web_sm``, or the BM25 half runs on raw text.

Embedders:

* ``potion`` — the same ``minishlab/potion-base-8M`` NoMem uses, so the
  embedding model is held constant.
* ``bge-small`` — ``BAAI/bge-small-en-v1.5`` through fastembed, a stronger
  local transformer encoder, as a stand-in for Mem0's default OpenAI
  ``text-embedding-3-small``.
* ``openai`` — Mem0's default (needs ``OPENAI_API_KEY``).

``--system nomem`` runs NoMem through this same search-then-fill path, so the
two systems are compared with identical context assembly.

"Context" for Mem0 is built the way an app using Mem0 would: take ``search``
results in rank order and add them until the same token budget (1,000
``ApproxCounter`` tokens) is full.

Usage:
    pip install "mem0ai[nlp]" fastembed && python -m spacy download en_core_web_sm
    python benchmarks/mem0_bench.py --embedder potion
    python benchmarks/mem0_bench.py --system nomem --embedder potion
    python benchmarks/mem0_bench.py --embedder bge-small --longmemeval-limit 100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from nomem import ApproxCounter  # noqa: E402

LOCOMO = ROOT / "benchmarks" / "data" / "locomo10.json"
LME = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
LME_TYPES = ("single-session-user", "single-session-assistant", "single-session-preference",
             "multi-session", "knowledge-update", "temporal-reasoning")
SEARCH_POOL = 100


class _Embed:
    """Adapter giving Mem0 any NoMem-style or fastembed encoder."""

    def __init__(self, name: str):
        self.name = name
        if name == "potion":
            from nomem.embed import Model2VecEmbedder
            enc = Model2VecEmbedder()
            self._encode = lambda texts: [list(map(float, v)) for v in enc.encode(texts)]
        elif name == "bge-small":
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._encode = lambda texts: [v.tolist() for v in model.embed(texts)]
        else:
            raise ValueError(name)
        self.dims = len(self._encode(["probe"])[0])
        self.config = type("Cfg", (), {"embedding_dims": self.dims})()

    def embed(self, text, memory_action=None):
        return self._encode([text.replace("\n", " ")])[0]

    def embed_batch(self, texts, memory_action="add"):
        return self._encode([t.replace("\n", " ") for t in texts])


class Mem0Runner:
    """One fresh Mem0 instance per conversation / haystack."""

    def __init__(self, embedder: str, infer: bool, llm: str):
        from mem0 import Memory  # imported lazily: optional dependency

        self._Memory = Memory
        self.infer = infer
        self.llm = llm
        self.embedder_name = embedder
        self.custom = None if embedder == "openai" else _Embed(embedder)
        self.dims = self.custom.dims if self.custom else 1536
        self.tmp = Path(tempfile.mkdtemp(prefix="mem0bench-"))
        self.n = 0
        self.llm_calls = 0

    def fresh(self):
        self.n += 1
        if not self.infer:
            os.environ.setdefault("OPENAI_API_KEY", "unused-infer-off")
        cfg = {
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": f"c{self.n}", "embedding_model_dims": self.dims,
                "path": str(self.tmp / f"q{self.n}"), "on_disk": False}},
            "history_db_path": str(self.tmp / f"h{self.n}.db"),
        }
        if self.custom is None:
            cfg["embedder"] = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
        else:
            cfg["embedder"] = {"provider": "openai", "config": {"embedding_dims": self.dims}}
        if self.infer:
            prov, _, model = self.llm.partition(":")
            cfg["llm"] = {"provider": prov, "config": {"model": model, "temperature": 0}}
        m = self._Memory.from_config(cfg)
        if self.custom is not None:
            m.embedding_model = self.custom
        if self.infer:
            gen = m.llm.generate_response

            def counted(*a, **kw):
                self.llm_calls += 1
                return gen(*a, **kw)
            m.llm.generate_response = counted
        return m

    def add(self, m, role, text, ts, uid):
        """Store one turn; return the ids of the memories it produced."""
        meta = {"ts": ts}
        res = m.add([{"role": role, "content": text}], user_id=uid, metadata=meta, infer=self.infer)
        return [r["id"] for r in res.get("results", []) if r.get("event") in ("ADD", "UPDATE")]

    def search(self, m, q, k, uid):
        res = m.search(q, top_k=k, filters={"user_id": uid}, threshold=0.0)
        return res["results"] if isinstance(res, dict) else res

    def clock(self, now):
        return contextlib.nullcontext()  # Mem0 OSS search has no notion of "now"

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class NoMemRunner:
    """NoMem behind the same interface, so both systems fill the budget the same way."""

    llm_calls = 0

    def __init__(self, embedder: str | None, vector_weight: float | None = None):
        from nomem import NoMem
        self._NoMem = NoMem
        self.kw = {}
        if vector_weight is not None:
            from nomem.retrieval import RetrievalConfig
            cfg = RetrievalConfig()
            cfg.weights["vector"] = vector_weight
            self.kw["retrieval"] = cfg
        if embedder == "potion":
            from nomem.embed import Model2VecEmbedder
            self.kw["embedder"] = Model2VecEmbedder()
        elif embedder == "bge-small":
            from nomem.embed import Embedder

            class _BGE(Embedder):
                model_name = "BAAI/bge-small-en-v1.5"

                def __init__(self):
                    super().__init__()
                    from fastembed import TextEmbedding
                    self._m = TextEmbedding(model_name=self.model_name)

                def encode(self, texts):
                    return [v.tolist() for v in self._m.embed(list(texts))]
            self.kw["embedder"] = _BGE()
        elif embedder == "openai":
            raise SystemExit("NoMem has no OpenAI embedder in this harness")
        self.n = 0

    def fresh(self):
        self.n += 1
        return self._NoMem(counter=ApproxCounter(), session_id=f"s{self.n}", **self.kw)

    def add(self, m, role, text, ts, uid):
        return [m.add(role, text, timestamp=ts).id]

    def search(self, m, q, k, uid):
        return [{"id": h.id, "memory": h.content} for h in m.search(q, k)]

    def clock(self, now):
        # NoMem ranks by recency: ask at the question's own date, as longmemeval_bench does
        from longmemeval_bench import _clock
        return _clock(now)

    def close(self):
        pass


def _context(hits, budget, counter):
    used, parts = 0, []
    for h in hits:
        t = counter.count(h["memory"]) + 1
        if used + t > budget:
            continue
        parts.append(h)
        used += t
    return parts, used


def _pct(xs, p):
    xs = sorted(xs)
    return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 3) if xs else 0.0


def _mean(xs):
    return round(100 * statistics.fmean(xs), 1) if xs else 0.0


def run_locomo(r: Mem0Runner, budget=1000, ks=(5, 10)) -> dict:
    data = json.loads(LOCOMO.read_text())
    counter = ApproxCounter()
    per_cat = {c: {f"r@{k}": [] for k in ks} | {"ctx": []} for c in CATEGORIES}
    add_lat, search_lat, ctx_tok = [], [], []
    for ci, sample in enumerate(data):
        conv, uid = sample["conversation"], f"conv{ci}"
        m = r.fresh()
        origin: dict = {}   # memory id -> dia_id it came from
        n = 1
        while f"session_{n}" in conv:
            base = datetime.strptime(conv[f"session_{n}_date_time"].strip(),
                                     "%I:%M %p on %d %B, %Y").timestamp()
            for i, turn in enumerate(conv[f"session_{n}"]):
                t0 = time.perf_counter()
                ids = r.add(m, "user", f"{turn['speaker']}: {turn['text']}", base + i * 30, uid)
                add_lat.append(time.perf_counter() - t0)
                for mid in ids:
                    origin[mid] = turn["dia_id"]
            n += 1
        for qa in sample["qa"]:
            cat = qa.get("category")
            ev = set(qa.get("evidence", [])) & set(origin.values())
            if cat not in CATEGORIES or not ev:
                continue
            t0 = time.perf_counter()
            hits = r.search(m, qa["question"], SEARCH_POOL, uid)
            search_lat.append(time.perf_counter() - t0)
            ranked = []
            for h in hits:
                d = origin.get(h["id"])
                if d and d not in ranked:
                    ranked.append(d)
            for k in ks:
                per_cat[cat][f"r@{k}"].append(len(ev & set(ranked[:k])) / len(ev))
            parts, used = _context(hits, budget, counter)
            ctx_tok.append(used)
            got = {origin.get(h["id"]) for h in parts}
            per_cat[cat]["ctx"].append(len(ev & got) / len(ev))
        print(f"  mem0 locomo conv {ci + 1}/{len(data)}", file=sys.stderr)
    return {
        "questions": sum(len(c["ctx"]) for c in per_cat.values()),
        "overall": {k: _mean([v for c in per_cat.values() for v in c[k]]) for k in per_cat[1]},
        "by_category": {CATEGORIES[c]: {k: _mean(v) for k, v in d.items()} for c, d in per_cat.items()},
        "latency_ms": {"add_p50": _pct(add_lat, .5), "add_p95": _pct(add_lat, .95),
                       "search_p50": _pct(search_lat, .5), "search_p95": _pct(search_lat, .95)},
        "avg_context_tokens": round(statistics.fmean(ctx_tok)) if ctx_tok else 0,
    }


def run_longmemeval(r: Mem0Runner, limit=100, budget=1000, ks=(5, 10)) -> dict:
    data = json.loads(LME.read_text())
    items = [x for x in data if not x["question_id"].endswith("_abs")]
    if limit and limit < len(items):
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    counter = ApproxCounter()
    keys = [f"turn r@{k}" for k in ks] + [f"session r@{k}" for k in ks] \
        + [f"session all@{k}" for k in ks] + ["ctx"]
    per = {t: {k: [] for k in keys} for t in LME_TYPES}
    add_lat, search_lat, ctx_tok = [], [], []
    t_start, n_turns = time.perf_counter(), 0
    fmt = "%Y/%m/%d (%a) %H:%M"
    for qi, item in enumerate(items):
        uid = item["question_id"]
        m = r.fresh()
        origin: dict = {}  # memory id -> (session id, turn key)
        ev_keys = set()
        for sid, date, turns in zip(item["haystack_session_ids"], item["haystack_dates"],
                                    item["haystack_sessions"]):
            base = datetime.strptime(date.strip(), fmt).timestamp()
            for i, turn in enumerate(turns):
                key = (sid, i)
                t0 = time.perf_counter()
                ids = r.add(m, turn["role"], turn["content"], base + i * 30, uid)
                add_lat.append(time.perf_counter() - t0)
                n_turns += 1
                for mid in ids:
                    origin[mid] = key
                if turn.get("has_answer"):
                    ev_keys.add(key)
        with r.clock(datetime.strptime(item["question_date"].strip(), fmt).timestamp()):
            t0 = time.perf_counter()
            hits = r.search(m, item["question"], SEARCH_POOL, uid)
            search_lat.append(time.perf_counter() - t0)
        turns_ranked, sessions = [], []
        for h in hits:
            key = origin.get(h["id"])
            if key is None:
                continue
            if key not in turns_ranked:
                turns_ranked.append(key)
            if key[0] not in sessions:
                sessions.append(key[0])
        row = per[item["question_type"]]
        gold = set(item["answer_session_ids"])
        for k in ks:
            found = len(gold & set(sessions[:k]))
            row[f"session r@{k}"].append(found / len(gold))
            row[f"session all@{k}"].append(float(found == len(gold)))
        parts, used = _context(hits, budget, counter)
        ctx_tok.append(used)
        if ev_keys:
            for k in ks:
                row[f"turn r@{k}"].append(len(ev_keys & set(turns_ranked[:k])) / len(ev_keys))
            got = {origin.get(h["id"]) for h in parts}
            row["ctx"].append(len(ev_keys & got) / len(ev_keys))
        if (qi + 1) % 10 == 0:
            print(f"  mem0 longmemeval {qi + 1}/{len(items)} "
                  f"({time.perf_counter() - t_start:.0f}s)", file=sys.stderr)
    return {
        "questions": len(items), "turns_ingested": n_turns,
        "overall": {k: _mean([v for t in per.values() for v in t[k]]) for k in keys},
        "by_type": {t: {k: _mean(v) for k, v in d.items()} | {"n": len(d["session r@5"])}
                    for t, d in per.items()},
        "latency_ms": {"add_p50": _pct(add_lat, .5), "add_p95": _pct(add_lat, .95),
                       "search_p50": _pct(search_lat, .5), "search_p95": _pct(search_lat, .95)},
        "avg_context_tokens": round(statistics.fmean(ctx_tok)) if ctx_tok else 0,
        "seconds": round(time.perf_counter() - t_start, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="mem0", choices=["mem0", "nomem"],
                    help="nomem runs NoMem through this same search-then-fill path")
    ap.add_argument("--embedder", default="potion", choices=["potion", "bge-small", "openai", "none"])
    ap.add_argument("--vector-weight", type=float, help="NoMem only: vector ranker RRF weight")
    ap.add_argument("--infer", default="off", choices=["off", "on"])
    ap.add_argument("--llm", default="openai:gpt-4.1-mini",
                    help="provider:model for --infer on (Mem0 LLM provider names)")
    ap.add_argument("--longmemeval-limit", type=int, default=100, help="0 = all 470")
    ap.add_argument("--skip-locomo", action="store_true")
    ap.add_argument("--skip-longmemeval", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    logging.getLogger("mem0").setLevel(logging.ERROR)
    if (args.infer == "on" or args.embedder == "openai") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("--infer on / --embedder openai need OPENAI_API_KEY")

    if args.system == "nomem":
        r = NoMemRunner(None if args.embedder == "none" else args.embedder, args.vector_weight)
    else:
        r = Mem0Runner(args.embedder, args.infer == "on", args.llm)
    out = {"system": args.system, "vector_weight": args.vector_weight, "embedder": args.embedder, "infer": args.infer}
    try:
        if not args.skip_locomo and LOCOMO.exists():
            out["locomo"] = run_locomo(r)
        if not args.skip_longmemeval and LME.exists():
            out["longmemeval"] = run_longmemeval(r, args.longmemeval_limit or None)
    finally:
        r.close()
    out["llm_calls"] = r.llm_calls
    print(json.dumps(out, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
