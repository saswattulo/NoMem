# NoMem vs Mem0

Measured 2026-11-28 with Mem0 OSS `mem0ai` 2.2.1 on 4 CPUs, using
`benchmarks/mem0_bench.py`. Same data, same questions, same evidence labels,
same metrics as `locomo_bench.py` and `longmemeval_bench.py`.

## Two kinds of numbers, not comparable to each other

| | What it measures | Needs an LLM? |
| --- | --- | --- |
| **Mem0's published scores** (vendor blog, Apr 2026, updated Sep 2026): LoCoMo **92.5**, LongMemEval **94.4**, BEAM-1M 64.1 | QA accuracy: an LLM answers from retrieved memories, another LLM judges the answer. Managed platform, which the blog says "includes proprietary optimizations not available in the open-source SDK". Judge/answer models not stated. Previous algorithm: 71.4 / 67.8. | Yes: extraction on every write, answering and judging at eval time |
| **This harness** | Retrieval: did the evidence turn come back in the top k, and did it reach a 1,000-token context? | No |

NoMem cannot produce a number like 92.5 without an LLM answering and judging
(that LLM would sit in the eval, not in NoMem). So the honest comparison is
the measured one below: both systems on the same retrieval questions. Mem0's
published figures are reported as published, not reproduced.

## Measured: Mem0 with `infer=False` vs NoMem

`infer=False` makes Mem0 store each turn verbatim and skip its LLM, so both
systems hold exactly the same text. Mem0 still uses its own hybrid search:
cosine over Qdrant plus Qdrant BM25 over spaCy-lemmatised text, fused by
normalised score addition. (Its third signal, entity boosts, is only filled
by the LLM pipeline.) The real Mem0 product also rewrites turns into facts
with an LLM; that mode is in the script (`--infer on`) but needs an API key
and was not run.

Context here is built the same way for both: search results packed best-first
into 1,000 `ApproxCounter` tokens. NoMem's own `build_context` reserves part
of the budget for recent turns, so its numbers are shown separately.

### LoCoMo (1,531 questions, 5,882 turns)

| System | R@5 | R@10 | Context recall | add p50 | search p50 |
| --- | --- | --- | --- | --- | --- |
| NoMem, keyword rules only | 56.8 | 64.0 | 72.4 | 0.5 ms | 5 ms |
| NoMem + potion-base-8M (default) | **58.6** | 66.0 | 73.6 | 0.6 ms | 6 ms |
| Mem0 + potion-base-8M | 55.9 | 64.2 | 73.6 | 10 ms | 72 ms |
| NoMem + bge-small-en-v1.5 | **59.5** | 67.0 | 74.4 | 67 ms | 49 ms |
| Mem0 + bge-small-en-v1.5 | 59.1 | 67.4 | **75.5** | 93 ms | 127 ms |
| NoMem `build_context` (its real API, 30-80% for archival) | | | 59.9 | | |

### LongMemEval (`longmemeval_s`, even spread of 100 answerable questions, ~49k turns)

| System | Session R@5 | all@5 | Turn R@5 | Turn R@10 | Context recall | add p50 / p95 | search p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoMem, keyword rules only | 89.1 | 81.0 | 55.2 | 74.6 | 68.9 | 2.6 / 41 ms | 7 / 22 ms |
| NoMem + potion-base-8M | **92.8** | **87.0** | 57.1 | 75.6 | 69.1 | 3.1 / 55 ms | 13 / 35 ms |
| Mem0 + potion-base-8M | 92.2 | 86.0 | **69.0** | **79.2** | **76.5** | 13 / 55 ms | 66 / 219 ms |
| NoMem `build_context` (+ potion) | | | | | 55.8 | | |

Turn R@5 by question type (NoMem + potion vs Mem0 + potion): single-session
user 78.6 vs 78.6, single-session assistant 63.6 vs 90.9, preference 41.7 vs
41.7, multi-session 42.8 vs 51.3, knowledge update 59.4 vs 81.2, temporal
59.3 vs 71.0. The 100-question spread has 6-27 questions per type, so
per-type gaps are noisy.

### What the numbers say

1. **On LoCoMo NoMem matches or beats Mem0's retrieval** at the same encoder
   (58.6 vs 55.9 R@5 with potion; 59.5 vs 59.1 with bge-small), and is
   10-20x faster to write and search.
2. **On LongMemEval Mem0 ranks individual turns better** (69.0 vs 57.1 turn
   R@5) while sessions are a tie (92.8 vs 92.2). Ablation: NoMem with only
   its vector ranker gets turn R@5 **74.6** and context 81.9 on LongMemEval,
   but only 37.8 R@5 on LoCoMo; raising the vector RRF weight to 1.0 gives
   58.3 / 56.2. So the gap is the fusion, not the encoder: rank-based RRF
   lets a strong keyword match on a long, wordy turn beat a turn that is
   semantically right. Mem0's score-based fusion (cosine + normalised BM25)
   keeps both benchmarks good. Trying score-based fusion in
   `nomem/retrieval.py` is the clearest next step.
3. **NoMem's own `build_context` gives up 14-21 points of context recall**
   on these benchmarks (59.9 vs 73.6 on LoCoMo; 55.8 vs 69.1 on LongMemEval)
   because it keeps recent turns in the budget. That is by design for a live
   agent, but it matters when quoting numbers next to Mem0's packed results.

## Cost and latency

| | NoMem | Mem0 OSS, `infer=False` (as measured) | Mem0 as normally used (`infer=True`) |
| --- | --- | --- | --- |
| Model calls per write | 0 (one local embedding with `[embed]`) | 1 local embedding + spaCy | 1 LLM call + embeddings |
| Write latency | 0.5-3 ms p50 | 10-93 ms p50 | One LLM round trip (typically about 1-3 s) per write |
| Search latency | 5-13 ms p50 | 66-127 ms p50 | Vendor reports about 0.9-1.1 s p50 end to end (platform) |
| API cost | $0 | $0 | Extraction system prompt alone is ~8.4k tokens per call |
| Context per query | ≤ 1,000 tokens (configurable) | same budget here | Vendor reports ~7k tokens per query |

Rough `infer=True` bill to reproduce here (estimate, not run): LoCoMo is
5,882 writes x ~10k input tokens ≈ 60M tokens; full `longmemeval_s` is
231,606 writes ≈ 2.3B tokens. At about $0.40 per million input tokens (a
small GPT-4.1-mini-class model; check current pricing) that is roughly $25
for LoCoMo and $900 for LongMemEval before prompt caching, plus hours of
wall time.

## Reproduce

```bash
pip install -e ".[embed]" "mem0ai[nlp]" fastembed
python -m spacy download en_core_web_sm
python benchmarks/mem0_bench.py --system mem0  --embedder potion
python benchmarks/mem0_bench.py --system nomem --embedder potion
python benchmarks/mem0_bench.py --system mem0  --embedder bge-small --skip-longmemeval
# real Mem0 with LLM extraction (paid):
OPENAI_API_KEY=... python benchmarks/mem0_bench.py --system mem0 --infer on --embedder openai
```

LongMemEval runs the same 100-question spread as `run_all.py`
(`--longmemeval-limit 0` for all 470). Mem0 + potion takes about 12 minutes
for those 100 questions; Mem0 + bge-small about 4x longer.
