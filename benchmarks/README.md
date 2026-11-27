# Benchmarks

NoMem is LLM-free, so it is measured with metrics that need no LLM either.

| Script | What it measures |
| --- | --- |
| `locomo_bench.py` | Retrieval quality on [LoCoMo](https://github.com/snap-research/locomo): did the evidence turn for each question make it into the top-k search results, and into the token-budgeted context? |
| `stress.py` | Write and query latency at 1k / 10k / 50k messages, planted-fact recall at scale, and budget invariants under randomised configs. |
| `run_all.py` | Runs both and prints one summary line per metric (used to build `TRAJECTORY.md`). |

## Data

LoCoMo is CC BY-NC 4.0, so it is not committed. Fetch it once:

```bash
curl -L -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## Run

```bash
python benchmarks/run_all.py            # everything, ~1-2 minutes
python benchmarks/locomo_bench.py       # retrieval quality only
python benchmarks/stress.py --quick     # latency + invariants, smaller scale
```

Category 5 (adversarial, unanswerable) questions are excluded because they
have no evidence turn. All runs use the deterministic `ApproxCounter` so
numbers are reproducible.
