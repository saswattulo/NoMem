"""NoMem vs Mem0 on the same retrieval questions, with no LLM on either side.

Mem0's normal write path calls an LLM to extract facts. This script runs
Mem0's own engine with ``infer=False`` instead: every turn is stored verbatim
and found again with Mem0's hybrid search (vector + BM25 + spaCy entity
boosts, Qdrant). That is the fairest LLM-free comparison: same turns in, same
questions, same scoring. Mem0 gets a stronger encoder than NoMem's default
(``BAAI/bge-small-en-v1.5``, 33M parameters, via fastembed), and NoMem is run
three ways: keyword rules only, + Model2Vec (its default vector ranker), and
+ the same bge-small encoder.

Metrics (higher is better):

* ``r@5`` / ``r@10``: share of evidence turns in the top 5 / 10 results.
* ``ctx``: share of evidence turns that reach a 1,000-token context. For
  NoMem that is ``build_context(query)``. Mem0 has no context builder, so its
  ranked results are packed best-first until the next one would not fit,
  which is how its search results are usually put into a prompt.
* ``add`` / ``search`` p50 latency, in ms, on this machine.

Setup::

    pip install mem0ai qdrant-client fastembed spacy nomem-ai[embed]
    python -m spacy download en_core_web_sm
    python benchmarks/mem0_compare.py --json benchmarks/results/mem0_compare.json

No API key is needed. Mem0 insists on an LLM config, so a dummy OpenAI key
is set; with ``infer=False`` it is never called.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from nomem import ApproxCounter, Embedder, NoMem  # noqa: E402

import locomo_bench as LB  # noqa: E402
import longmemeval_bench as LME  # noqa: E402

BUDGET = 1000
BGE = "BAAI/bge-small-en-v1.5"
COUNTER = ApproxCounter()


class FastEmbedEmbedder(Embedder):
    """The same fastembed encoder Mem0 uses here, for an equal-encoder run."""

    def __init__(self, model: str = BGE):
        from fastembed import TextEmbedding

        self.model_name = model
        self._model = TextEmbedding(model_name=model)

    def encode(self, texts):
        return [list(map(float, v)) for v in self._model.embed(list(texts))]


# -- systems ----------------------------------------------------------------
# Each system: reset(), add(role, text, ts) -> id, search(q, k) -> [ids],
# context(q) -> text (or None), texts: id -> stored text.


class NoMemSystem:
    def __init__(self, name, embedder=None):
        self.name, self.embedder, self.mem = name, embedder, None

    def reset(self, scope):
        if self.mem:
            self.mem.close()
        self.mem = NoMem(max_tokens=BUDGET, counter=COUNTER, session_id=scope,
                         embedder=self.embedder)

    def add(self, role, text, ts):
        return self.mem.add(role, text, timestamp=ts).id

    def search(self, q, k):
        return [m.id for m in self.mem.search(q, k)]

    def context(self, q):
        return self.mem.build_context(query=q).text


class Mem0System:
    def __init__(self, threshold=0.1):
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        os.environ.setdefault("OPENAI_API_KEY", "sk-unused-infer-false")
        from mem0 import Memory

        self.tmp = tempfile.mkdtemp(prefix="mem0cmp-")
        self.name = "Mem0 (infer=False, bge-small)"
        self.threshold = threshold
        self.m = Memory.from_config({
            "embedder": {"provider": "fastembed", "config": {"model": BGE, "embedding_dims": 384}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "cmp", "embedding_model_dims": 384,
                "path": os.path.join(self.tmp, "qdrant"), "on_disk": False}},
            "history_db_path": os.path.join(self.tmp, "history.db"),
        })
        import mem0
        self.version = getattr(mem0, "__version__", "?")
        self.texts = {}

    def reset(self, scope):
        self.m.reset()
        self.scope = scope
        self.texts = {}

    def add(self, role, text, ts):
        r = self.m.add([{"role": role, "content": text}], user_id=self.scope, infer=False)
        mid = r["results"][0]["id"]
        self.texts[mid] = text
        return mid

    def search(self, q, k):
        r = self.m.search(q, filters={"user_id": self.scope}, top_k=k, threshold=self.threshold)
        return [x["id"] for x in r["results"]]

    def context(self, q):
        ids = self.search(q, 30)
        out, used = [], 0
        for mid in ids:
            line = self.texts[mid]
            cost = COUNTER.count_message("user", line)
            if used + cost > BUDGET:
                break
            out.append(line)
            used += cost
        return "\n".join(out)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# -- benchmarks -------------------------------------------------------------


def _pct(xs, p=0.5):
    xs = sorted(xs)
    return round(1000 * xs[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else 0.0


def _mean(xs):
    return round(100 * statistics.fmean(xs), 1) if xs else 0.0


def locomo(system, data_path, limit_convs=None):
    data = json.loads(Path(data_path).read_text())[:limit_convs]
    r5, r10, ctx, add_lat, search_lat = [], [], [], [], []
    for ci, sample in enumerate(data):
        system.reset(f"conv{ci}")
        conv, by_dia, texts = sample["conversation"], {}, {}
        n = 1
        while f"session_{n}" in conv:
            base = LB._parse_session_time(conv[f"session_{n}_date_time"])
            for i, turn in enumerate(conv[f"session_{n}"]):
                text = f"{turn['speaker']}: {turn['text']}"
                t0 = time.perf_counter()
                mid = system.add("user", text, base + i * 30)
                add_lat.append(time.perf_counter() - t0)
                by_dia[turn["dia_id"]] = mid
                texts[mid] = text
            n += 1
        for qa in sample["qa"]:
            if qa.get("category") not in LB.CATEGORIES:
                continue
            ev = {by_dia[e] for e in qa.get("evidence", []) if e in by_dia}
            if not ev:
                continue
            t0 = time.perf_counter()
            hits = system.search(qa["question"], 10)
            search_lat.append(time.perf_counter() - t0)
            r5.append(len(ev & set(hits[:5])) / len(ev))
            r10.append(len(ev & set(hits[:10])) / len(ev))
            text = system.context(qa["question"])
            ctx.append(sum(1 for e in ev if texts[e] in text) / len(ev))
        print(f"  {system.name}: LoCoMo conv {ci + 1}/{len(data)}", file=sys.stderr, flush=True)
    return {"questions": len(r5), "r@5": _mean(r5), "r@10": _mean(r10), "ctx": _mean(ctx),
            "add_p50_ms": _pct(add_lat), "search_p50_ms": _pct(search_lat)}


def longmemeval(system, data_path, limit):
    items = LME.select(json.loads(Path(data_path).read_text()), limit)
    turn5, sess5, ctx, add_lat, search_lat = [], [], [], [], []
    for qi, item in enumerate(items):
        system.reset(item["question_id"])
        session_of, evidence, texts = {}, set(), {}
        for sid, date, turns in zip(item["haystack_session_ids"], item["haystack_dates"],
                                    item["haystack_sessions"]):
            base = LME._parse_date(date)
            for i, turn in enumerate(turns):
                t0 = time.perf_counter()
                mid = system.add(turn["role"], turn["content"], base + i * 30)
                add_lat.append(time.perf_counter() - t0)
                session_of[mid], texts[mid] = sid, turn["content"]
                if turn.get("has_answer"):
                    evidence.add(mid)
        q, now = item["question"], LME._parse_date(item["question_date"])
        with LME._clock(now):
            t0 = time.perf_counter()
            hits = system.search(q, 50)
            search_lat.append(time.perf_counter() - t0)
            text = system.context(q)
        sessions = []
        for mid in hits:
            if session_of[mid] not in sessions:
                sessions.append(session_of[mid])
        gold = set(item["answer_session_ids"])
        sess5.append(len(gold & set(sessions[:5])) / len(gold))
        if evidence:
            turn5.append(len(evidence & set(hits[:5])) / len(evidence))
            ctx.append(sum(1 for e in evidence if texts[e] in text) / len(evidence))
        if (qi + 1) % 10 == 0:
            print(f"  {system.name}: LongMemEval {qi + 1}/{len(items)}", file=sys.stderr, flush=True)
    return {"questions": len(items), "session r@5": _mean(sess5), "turn r@5": _mean(turn5),
            "ctx": _mean(ctx), "add_p50_ms": _pct(add_lat), "search_p50_ms": _pct(search_lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="nomem,nomem-m2v,nomem-bge,mem0")
    ap.add_argument("--locomo", type=Path, default=LB.DEFAULT_DATA)
    ap.add_argument("--longmemeval", type=Path, default=LME.DEFAULT_DATA)
    ap.add_argument("--longmemeval-limit", type=int, default=50)
    ap.add_argument("--locomo-convs", type=int, default=None)
    ap.add_argument("--mem0-threshold", type=float, default=0.1)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out = {}
    for name in args.systems.split(","):
        if name == "nomem":
            system = NoMemSystem("NoMem (rules only)")
        elif name == "nomem-m2v":
            from nomem import Model2VecEmbedder
            system = NoMemSystem("NoMem + Model2Vec", Model2VecEmbedder())
        elif name == "nomem-bge":
            system = NoMemSystem("NoMem + bge-small", FastEmbedEmbedder())
        elif name == "mem0":
            system = Mem0System(args.mem0_threshold)
        else:
            sys.exit(f"unknown system {name}")
        row = {}
        if args.locomo.exists():
            row["locomo"] = locomo(system, args.locomo, args.locomo_convs)
        if args.longmemeval.exists() and args.longmemeval_limit:
            row["longmemeval"] = longmemeval(system, args.longmemeval, args.longmemeval_limit)
        if isinstance(system, Mem0System):
            row["mem0_version"] = system.version
            system.close()
        out[system.name] = row
        print(json.dumps({system.name: row}), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
