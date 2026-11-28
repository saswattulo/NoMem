# Benchmarks

NoMem is LLM-free, so it is measured with metrics that need no LLM either:
did the evidence for a question come back from search, and did it reach the
budgeted context? Every number here comes from `python benchmarks/run_all.py`
(add `--embedder model2vec` for the vector column), and every step that led
to it is in
[`benchmarks/TRAJECTORY.md`](https://github.com/saswattulo/NoMem/blob/main/benchmarks/TRAJECTORY.md).

## Results

| | v0.1 (FTS5 only) | v0.2+ (hybrid, rules only) | + Model2Vec |
| --- | --- | --- | --- |
| LoCoMo search recall@5 | 50.9% | 56.8% | 58.6% |
| LoCoMo context recall (1,000 tokens) | 52.5% | 58.1% | 59.9% |
| LongMemEval session recall@5 | | 89.9% | 92.1% |
| LongMemEval turn recall@5 | | 52.7% | 55.9% |
| LongMemEval context recall (1,000 tokens) | | 50.0% | 53.2% |
| Query latency at 50k messages (p50) | 190 ms | 3-5 ms | 7-10 ms |
| Planted facts recalled at 50k messages | 7/8 | 8/8 | 8/8 |
| Updated facts caught, fresh held-out (false flags) | | 19/25 (0/25) | 19/25 (0/25) |
| Write throughput | | ~4,000 msg/s | ~1,500 msg/s |
| Model calls per write | 0 | 0 | 1 embedding (no generative model) |

## What each benchmark measures

**[LoCoMo](https://github.com/snap-research/locomo)**: 10 long multi-session
conversations (5,882 turns) and 1,531 answerable questions in four
categories (single-hop, multi-hop, temporal, open-domain). For each question:
is the evidence turn in the top 5 search results, and in a 1,000-token
context from `build_context(query)`? The adversarial category (no evidence
turn) is excluded.

**[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** (`longmemeval_s`):
470 answerable questions, each over its own history of about 500 turns. Is
the evidence session in the top 5 sessions, the evidence turn in the top 5
turns, and the evidence turn in a 1,000-token context? Each question is asked
at its own `question_date`, so "last weekend" resolves as the dataset
intends. Its official score is QA accuracy judged by an LLM, so NoMem
reports the retrieval metrics instead.

**Stress** (`stress.py`): write and query latency at 1k, 10k and 50k
messages, recall of 8 planted facts at scale, and the budget invariant
(`tokens_used <= max_tokens`) over 500 randomised configurations: 0
violations.

**Updated facts** (`updates_bench.py`): does a newer memory that changes an
older one get linked to it, are both shown in context with a conflict
marker, and how often do look-alike pairs or real chat get flagged by
mistake? The held-out set was written before the last round of rule changes
and measured once.

## Running them

```bash
pip install -e ".[embed]"
curl -L -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
curl -L -o benchmarks/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

python benchmarks/run_all.py                         # ~1-2 minutes
python benchmarks/run_all.py --embedder model2vec    # with the vector ranker
python benchmarks/run_all.py --longmemeval-limit 0   # all 470 LongMemEval questions
python benchmarks/mem0_bench.py --system nomem --embedder potion   # NoMem vs Mem0
```

All runs use the deterministic `ApproxCounter`, so numbers are reproducible.

## Where it is weak

- **Context recall is below turn recall@10** (50.0% vs 74.5% on
  LongMemEval): `retrieval_top_k=5` caps how many matches are considered for
  the context.
- **Preference and multi-session questions** on LongMemEval (about 35-40%
  context recall) need evidence from several turns that share few words with
  the question.
- **Contradiction rules miss** numbers written as words, ordinals and some
  brand names.
