# NoMem vs Mem0

Measured 2026-11-28 with Mem0 OSS `mem0ai` 2.2.1 on 4 CPUs, using
`benchmarks/mem0_bench.py`. Same data, same questions, same evidence labels,
same metrics as `locomo_bench.py` and `longmemeval_bench.py`.

## Two kinds of numbers, not comparable to each other

| | What it measures | Needs an LLM? |
| --- | --- | --- |
| **Mem0's published scores** (vendor blog, Apr 2026, updated Sep 2026): LoCoMo **92.5**, LongMemEval **94.4**, BEAM-1M 64.1 | QA accuracy: an LLM answers from retrieved memories, another LLM judges the answer. Managed platform, which the blog says "includes proprietary optimizations not available in the open-source SDK". Judge/answer models not stated. Previous algorithm: 71.4 / 67.8. | Yes: extraction on every write, answering and judging at eval time |
| **This harness** | Retrieval: did the evidence turn come back in the top k, and did it reach a 1,000-token context? | No |

A number like 92.5 needs an LLM to answer and another to judge. That LLM
sits in the evaluation, never in NoMem: `benchmarks/qa_bench.py` (section
[End-to-end answer accuracy](#end-to-end-answer-accuracy-llm-judged-j) below)
runs NoMem, Mem0 OSS and two reference contexts through Mem0's own LoCoMo
answer and judge prompts. Until that has been run with an API key, the
measured comparison is the retrieval one: both systems on the same questions.

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
| NoMem + potion-base-8M | **92.5** | **87.0** | **77.7** | **85.7** | **86.3** | 2.5 / 48 ms | 11 / 29 ms |
| Mem0 + potion-base-8M | 92.2 | 86.0 | 69.0 | 79.2 | 76.5 | 13 / 55 ms | 66 / 219 ms |
| NoMem 0.3 (rank fusion) + potion, for reference | 92.8 | 87.0 | 57.1 | 75.6 | 69.1 | 3.1 / 55 ms | 13 / 35 ms |
| NoMem `build_context` (+ potion) | | | | | 75.7 | | |

Turn R@5 by question type (NoMem + potion vs Mem0 + potion): single-session
user 82.1 vs 78.6, single-session assistant 81.8 vs 90.9, preference 66.7 vs
41.7, multi-session 63.5 vs 51.3, knowledge update 90.6 vs 81.2, temporal
82.1 vs 71.0. The 100-question spread has 6-27 questions per type, so
per-type gaps are noisy. The fusion settings were tuned on these 100
questions; on 124 held-out LongMemEval questions NoMem + potion gets 76.1
turn R@5 (0.3: 55.6). Mem0 was not run on the held-out set.

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
   it. Mem0 still leads on single-session-assistant questions (90.9 vs
   81.8), where the evidence sits in long assistant answers that the
   length penalty discounts.
3. **NoMem's own `build_context` gives up about 11-16 points of context
   recall** on these benchmarks (63.4 vs 79.0 on LoCoMo; 75.7 vs 86.3 on LongMemEval)
   because it keeps recent turns in the budget. That is by design for a live
   agent, but it matters when quoting numbers next to Mem0's packed results.

## End-to-end answer accuracy (LLM-judged J)

`benchmarks/qa_bench.py` scores LoCoMo the way Mem0 does: an LLM answers each
of the 1,540 category 1-4 questions from the context a system builds, and an
LLM judge labels the answer CORRECT or WRONG. J is the share labelled
CORRECT. NoMem still uses no LLM; the models sit only in the harness.

What is held fixed, so the systems differ only in their context:

| | Setting |
| --- | --- |
| Answer prompt | Mem0's `ANSWER_PROMPT` (from mem0's `evaluation/prompts.py`), verbatim, sent as the system message |
| Judge prompt | Mem0's `ACCURACY_PROMPT` (from `evaluation/metrics/llm_judge.py`), verbatim, JSON mode |
| Models | `gpt-4o-mini` answers and judges, temperature 0 (Mem0's paper setting); `--provider anthropic` uses `claude-haiku-5-5` |
| Questions | All 1,540 LoCoMo questions in categories 1-4 (adversarial excluded, as in Mem0's paper) |
| Context | 1,000 or 2,000 `ApproxCounter` tokens, every memory shown with its date |

Contexts compared:

| Name | Context |
| --- | --- |
| `nomem` | `NoMem.build_context(query=q)`, the real API: facts card, retrieved memories and the recent turns, in one budget |
| `nomem-search` | `NoMem.search` results packed best-first into the budget |
| `mem0` | Mem0 OSS 2.2.1, `infer=False`, its own hybrid search, packed the same way |
| `full` | The whole conversation with session dates (~20k tokens): Mem0's "full-context" reference |
| `none` | No memory: the floor |

### Measured so far (no LLM yet)

The answer and judge stage needs an `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`,
which this environment did not have, so **no J score has been measured
yet**. The context stage has been run for all five systems. As a rough,
LLM-free signal, the table shows how often every word of the gold answer
appears somewhere in the context. It undercounts temporal questions, whose
gold answers are usually worked out from a date ("the day before 8 May"),
and it is not a substitute for J.

| Context (potion-base-8M) | Tokens | Gold answer in context | single-hop | multi-hop | temporal | open-domain | Retrieval p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nomem-search`, 1,000 | 995 | **48.8%** | **65.6** | **26.6** | 35.8 | **9.4** | **8.5 ms** |
| `mem0`, 1,000 | 995 | 45.0% | 59.6 | 23.8 | **36.4** | 8.3 | 39.7 ms |
| `nomem` (`build_context`), 1,000 | 978 | 35.5% | 55.8 | 13.1 | 10.3 | 8.3 | 5.3 ms |
| `nomem-search`, 2,000 | 1,991 | **55.2%** | **70.7** | **32.3** | 47.4 | **12.5** | |
| `mem0`, 2,000 | 1,995 | 52.3% | 66.0 | 30.9 | 47.4 | 11.5 | 39.6 ms |
| `nomem` (`build_context`), 2,000 | 1,979 | 39.1% | 61.5 | 14.9 | 10.9 | 8.3 | |
| `full` | 20,059 | 75.8% | 87.8 | 63.8 | 71.7 | 19.8 | |
| `none` | 0 | 0.5% | | | | | |

Two things already show. Packed the same way, NoMem's context contains the
answer more often than Mem0's at both budgets. And NoMem's own
`build_context` is well behind its packed search on LoCoMo (35.5% vs 48.8%),
most of all on temporal and multi-hop questions: it spends part of the
budget on the latest turns, which rarely hold the answer to a question about
months ago, and those recent turns carry no date. That is the same gap the
retrieval tables show (63.4% vs 79.0% context recall), and the J run will
show how much of it reaches the final answer.

### Published LoCoMo J scores, for reference

From Mem0's paper (arXiv 2504.19413, gpt-4o-mini answering, mean of 10
runs). These were measured by Mem0, not here.

| System | J | Tokens per query |
| --- | --- | --- |
| Full context | 72.90 | ~26,000 |
| Mem0 with graph (Mem0g) | 68.44 | |
| Mem0 | 66.88 | ~1,800 (memories) |
| Zep | 65.99 | |
| Best RAG (k=2, 256-token chunks) | 60.97 | |
| LangMem | 58.10 | |
| OpenAI memory | 52.90 | |
| A-Mem | 48.38 | |

Mem0 per category: single-hop 67.13, multi-hop 51.15, open-domain 72.93,
temporal 55.51. Mem0's current headline, **92.5**, is its managed platform
with a newer algorithm, and the answer and judge models are not stated, so
it is not a like-for-like target. LoCoMo J also moves by several points with
the answer prompt and the judge model alone, which is why this harness uses
Mem0's prompts and model and runs the `full` and `none` references next to
every system: a run should land `full` near 73 before its other numbers are
compared with the paper.

### Run it

```bash
pip install -e ".[embed]" openai                       # + "mem0ai[nlp]" fastembed for mem0
python benchmarks/qa_bench.py --stage contexts \
    --systems nomem,nomem-search,mem0,full,none         # no LLM, ~5 minutes
OPENAI_API_KEY=... python benchmarks/qa_bench.py \
    --systems nomem,nomem-search,mem0,full,none         # answer + judge
OPENAI_API_KEY=... python benchmarks/qa_bench.py --budget 2000 --systems nomem,nomem-search,mem0
python benchmarks/qa_bench.py --provider mock --convs 2 --limit 15   # offline smoke test
```

Contexts are saved to `benchmarks/results/` and every LLM reply is cached
there (`qa_cache.jsonl`), so a re-run or an interrupted run only pays for
calls not yet made. Cost estimate with gpt-4o-mini: the four memory systems
at 1,000 tokens are about 1,540 x 4 x ~2.5k tokens ≈ 15M input tokens
(≈ $2.50); `full` adds about 31M (≈ $5). Each row of the output
(`qa_<provider>_<model>_<setup>_rows.jsonl`) keeps the context, answer and
verdict, so any score can be checked question by question.

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
