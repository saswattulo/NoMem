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
| NoMem, keyword rules only | 59.6 | 67.8 | 77.7 | 0.5 ms | 10 ms* |
| NoMem + potion-base-8M (default) | **62.3** | **70.3** | **79.0** | 0.7 ms | 8 ms* |
| Mem0 + potion-base-8M | 55.9 | 64.2 | 73.6 | 10 ms | 72 ms |
| NoMem + bge-small-en-v1.5 | **63.9** | **73.1** | **82.9** | 64 ms | 50 ms |
| Mem0 + bge-small-en-v1.5 | 59.1 | 67.4 | 75.5 | 93 ms | 127 ms |
| NoMem 0.3 (rank fusion) + potion, for reference | 58.6 | 66.0 | 73.6 | 0.6 ms | 6 ms |
| NoMem `build_context` (its real API, 30-80% for archival) | | | 63.4 | | |

NoMem rows re-measured after score fusion (step 12 in `TRAJECTORY.md`);
Mem0 rows are unchanged from the first run. *Measured with other benchmarks
running on the same 4 CPUs, so NoMem latencies here are pessimistic.

### LongMemEval (`longmemeval_s`, even spread of 100 answerable questions, ~49k turns)

| System | Session R@5 | all@5 | Turn R@5 | Turn R@10 | Context recall | add p50 / p95 | search p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoMem, keyword rules only | 88.2 | 80.0 | 72.0 | 79.0 | 79.4 | 6 / 81 ms* | 14 / 48 ms* |
| NoMem + potion-base-8M | **92.5** | **87.0** | **78.7** | **85.7** | **85.3** | 2.4 / 47 ms | 11 / 18 ms |
| Mem0 + potion-base-8M | 92.2 | 86.0 | 69.0 | 79.2 | 76.5 | 13 / 55 ms | 66 / 219 ms |
| NoMem 0.3 (rank fusion) + potion, for reference | 92.8 | 87.0 | 57.1 | 75.6 | 69.1 | 3.1 / 55 ms | 13 / 35 ms |
| NoMem `build_context` (+ potion) | | | | | 77.7 | | |

Turn R@5 by question type (NoMem + potion vs Mem0 + potion): single-session
user 82.1 vs 78.6, single-session assistant 90.9 vs 90.9, preference 66.7 vs
41.7, multi-session 63.5 vs 51.3, knowledge update 90.6 vs 81.2, temporal
82.1 vs 71.0. The 100-question spread has 6-27 questions per type, so
per-type gaps are noisy. The fusion settings were tuned on these 100
questions; on 124 held-out LongMemEval questions NoMem + potion gets 76.1
turn R@5 (0.3: 55.6). Mem0 was not run on the held-out set.

Single-session-assistant questions (answer inside an assistant reply), all
56 of them, same harness and packing (step 13 in `TRAJECTORY.md`):

| System | Turn R@5 | Turn R@10 | Context recall |
| --- | --- | --- | --- |
| NoMem + potion, before the assistant cue | 82.1 | 91.1 | |
| NoMem + potion | **89.3** | **94.6** | **85.7** |
| Mem0 + potion | 85.7 | 91.1 | 78.6 |

### What the numbers say

1. **On LoCoMo NoMem beats Mem0's retrieval** at the same encoder
   (62.3 vs 55.9 R@5 with potion; 63.9 vs 59.1 with bge-small), and is
   10-20x faster to write and search. Even keyword-only NoMem (59.6) is
   ahead of Mem0 with potion.
2. **On LongMemEval NoMem now ranks individual turns better too** (77.7 vs
   69.0 turn R@5; sessions a tie, 92.5 vs 92.2). NoMem 0.3 was at 57.1: its
   rank-based RRF let a long, wordy turn that repeats a query word beat a
   turn that is semantically right. Score fusion (BM25 relative to the
   best hit, squared, plus raw cosine) and a mild long-turn penalty fixed
   it. Single-session-assistant questions, where Mem0 led (90.9 vs 81.8 on
   the spread), are now level on the spread and ahead on all 56 (89.3 vs
   85.7): a query cue spots questions about what the assistant said and
   favours assistant turns.
3. **NoMem's own `build_context` gives up about 11-16 points of context
   recall** on these benchmarks (63.4 vs 79.0 on LoCoMo; 77.7 vs 85.3 on LongMemEval)
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
