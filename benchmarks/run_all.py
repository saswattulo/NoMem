"""Run every benchmark and print a compact one-line-per-metric summary.

Usage:
    python benchmarks/run_all.py [--label NAME] [--quick] [--embedder model2vec|hashing|MODEL]

Writes full results to benchmarks/results/<label>.json (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import locomo_bench  # noqa: E402
import stress  # noqa: E402


def make_embedder(name: str):
    from nomem.embed import HashingEmbedder, Model2VecEmbedder

    if name == "hashing":
        return HashingEmbedder()
    if name == "model2vec":
        return Model2VecEmbedder()
    return Model2VecEmbedder(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--embedder", help="model2vec (potion-base-8M), hashing, or a Model2Vec model id/path")
    ap.add_argument("--vector-weight", type=float, help="override the vector ranker's RRF weight")
    args = ap.parse_args()

    mem_kwargs = {}
    if args.embedder:
        mem_kwargs["embedder"] = make_embedder(args.embedder)
    if args.vector_weight is not None:
        from nomem.retrieval import RetrievalConfig
        cfg = RetrievalConfig()
        cfg.weights["vector"] = args.vector_weight
        mem_kwargs["retrieval"] = cfg

    out = {"label": args.label, "embedder": args.embedder, "vector_weight": args.vector_weight}
    if locomo_bench.DEFAULT_DATA.exists():
        out["locomo"] = locomo_bench.run(mem_kwargs=mem_kwargs)
    else:
        print("(LoCoMo data missing, skipping; see benchmarks/README.md)")
    out["stress"] = stress.run(args.quick, mem_kwargs=mem_kwargs)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.label}.json").write_text(json.dumps(out, indent=2))

    if "locomo" in out:
        lo = out["locomo"]
        o = lo["overall"]
        print(f"LoCoMo ({lo['questions']} q, {lo['budget']} tok): "
              f"search R@5 {o['r@5']}  R@10 {o['r@10']}  context recall {o['ctx']}")
        for cat, d in lo["by_category"].items():
            print(f"  {cat:<12} R@5 {d['r@5']:>5}  R@10 {d['r@10']:>5}  ctx {d['ctx']:>5}")
        print(f"  latency ms: {lo['latency_ms']}  avg ctx tokens {lo['avg_context_tokens']}")
    for s in out["stress"]["scale"]:
        print(f"scale {s['messages']:>6}: ingest {s['ingest_msgs_per_s']}/s  "
              f"query p50 {s['query_p50_ms']}ms p95 {s['query_p95_ms']}ms  planted {s['planted_recall']}")
    inv = out["stress"]["invariants"]
    print(f"invariants: {inv['invariant_trials']} trials, {inv['violations']} violations")


if __name__ == "__main__":
    main()
