# Benchmarks

NoMem is LLM-free, so it is measured with metrics that need no LLM either. The one exception is `qa_bench.py`, which puts an LLM in the evaluation (never in NoMem) to produce the answer-accuracy score other memory layers publish.

| Script | What it measures |
| --- | --- |
| `locomo_bench.py` | Retrieval quality on [LoCoMo](https://github.com/snap-research/locomo): did the evidence turn for each question make it into the top-k search results, and into the token-budgeted context? |
| `longmemeval_bench.py` | Retrieval quality on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (`longmemeval_s`, 470 answerable questions, each with its own ~500-turn history): are the evidence sessions and turns in the top-k results, and in the budgeted context? A second benchmark so gains are not tuned to LoCoMo alone. |
| `stress.py` | Write and query latency at 1k / 10k / 50k messages, planted-fact recall at scale, and budget invariants under randomised configs. |
| `updates_bench.py` | Contradictions: does a newer memory that changes an older one get linked to it (40 tuned, 20 dev and 25 fresh held-out update cases), are both shown in context with a conflict marker, and how often do look-alike pairs (75 controls) or real LoCoMo chat get flagged by mistake? |
| `mem0_bench.py` | Mem0 (mem0ai OSS, `infer=False`, no LLM) and NoMem through the same LoCoMo and LongMemEval questions, with identical context packing. Results and the published-vs-measured caveats are in [`MEM0_COMPARISON.md`](MEM0_COMPARISON.md). |
| `qa_bench.py` | End-to-end answer accuracy on LoCoMo, the LLM-judged **J** score Mem0 publishes: an LLM answers each question from a system's context and an LLM judge grades it, with Mem0's own prompts and model. Runs NoMem (`build_context` and packed search), Mem0 OSS, the whole conversation and no memory through the same pipeline. The LLM is in the harness only; needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (the context stage, `--stage contexts`, does not). See [`MEM0_COMPARISON.md`](MEM0_COMPARISON.md#end-to-end-answer-accuracy-llm-judged-j). |
| `run_all.py` | Runs all four (LongMemEval on an even spread of 100 questions) and prints one summary line per metric (used to build `TRAJECTORY.md`). |

## Data

Neither dataset is committed (LoCoMo is CC BY-NC 4.0; LongMemEval is 265 MB).
Fetch them once; each benchmark is skipped if its file is missing:

```bash
curl -L -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
curl -L -o benchmarks/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

## Run

```bash
python benchmarks/run_all.py            # everything, ~1-2 minutes
python benchmarks/locomo_bench.py       # retrieval quality only
python benchmarks/longmemeval_bench.py  # all 470 LongMemEval questions, ~20 minutes
python benchmarks/run_all.py --longmemeval-limit 0  # everything, full LongMemEval
python benchmarks/stress.py --quick     # latency + invariants, smaller scale
python benchmarks/run_all.py --embedder model2vec   # same, with the vector ranker
python benchmarks/run_all.py --embedder hashing --vector-weight 0.5  # ablations
```

`--embedder model2vec` needs `pip install nomem[embed]` and downloads
`minishlab/potion-base-8M` (~30 MB) from Hugging Face once.

LongMemEval's official score is QA accuracy judged by an LLM, so NoMem
reports the benchmark's retrieval metrics instead: turn recall@k (evidence
turns marked `has_answer`), session recall@k (turns collapsed to their
sessions, best turn first) and `all@k` (every evidence session in the top k).
Its 30 abstention questions are excluded, and each question is asked at its
own `question_date`, so relative dates and recency resolve as the dataset
intends.

LoCoMo category 5 (adversarial, unanswerable) questions are excluded because they
have no evidence turn. All runs use the deterministic `ApproxCounter` so
numbers are reproducible.
