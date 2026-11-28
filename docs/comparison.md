# NoMem vs Mem0

Measured on 2026-11-28 with Mem0 OSS (`mem0ai` 2.2.1) on 4 CPUs, using
`benchmarks/mem0_bench.py`: the same data, questions, evidence labels and
metrics as NoMem's own LoCoMo and LongMemEval benchmarks. The full write-up,
with every table, is
[`benchmarks/MEM0_COMPARISON.md`](https://github.com/saswattulo/NoMem/blob/main/benchmarks/MEM0_COMPARISON.md).

## Summary

--8<-- "docs/_comparison_table.md"

## How it was measured

Mem0 normally calls an LLM on every write to rewrite turns into facts. With
`infer=False` it stores each turn verbatim and skips the LLM, so both
systems hold exactly the same text, and Mem0 still uses its own hybrid
search (cosine over Qdrant plus BM25 over spaCy-lemmatised text). Both are
given the same embedding model in each row.

Context recall is measured by packing each system's search results,
best-first, into 1,000 tokens. NoMem's own `build_context()` also keeps
recent turns in the budget, which is what a live agent wants, so it scores
lower on these benchmarks (63.4% on LoCoMo, 75.7% on LongMemEval) than the
packed figure (79.0% and 86.3%).

## What the numbers say

1. **On LoCoMo, NoMem matches or beats Mem0's retrieval** with the same
   encoder, and writes and searches 10 to 20 times faster.
2. **On LongMemEval NoMem now also ranks individual turns better**: 77.7%
   vs 69.0% turn recall@5, sessions a tie. NoMem 0.3 was at 57.1%; its
   rank-based fusion let a long, wordy turn that repeats a query word beat a
   turn that is right in meaning. Score fusion plus a mild long-turn penalty
   closed that gap (checked on 124 held-out questions: 55.6% to 76.1%).
3. **Mem0 as normally used costs an LLM call per write.** NoMem never does.

## Mem0's published scores, and answer accuracy

Mem0 reports 92.5 on LoCoMo and 94.4 on LongMemEval for its managed
platform, and its paper reports 66.88 on LoCoMo for the open algorithm
(full-context reference: 72.90). Those are **QA accuracy (J)**: an LLM
answers from retrieved memories and another LLM judges the answer. They
measure something different from the retrieval recall above.

`benchmarks/qa_bench.py` measures the same thing for NoMem, with Mem0's own
answer and judge prompts and model (gpt-4o-mini), and runs Mem0 OSS, the
whole conversation and no memory through the identical pipeline. The LLM
sits only in the benchmark, never in NoMem. The J scores have not been run
yet (they need an API key). The context stage has, and as an LLM-free
signal it already shows NoMem's packed context holding the gold answer more
often than Mem0's:

| LoCoMo, 1,540 questions | Gold answer in context, 1,000 tokens | 2,000 tokens |
| --- | --- | --- |
| NoMem search, packed | **48.8%** | **55.2%** |
| Mem0 OSS (`infer=False`), packed | 45.0% | 52.3% |
| NoMem `build_context()` | 35.5% | 39.1% |
| Whole conversation (~20k tokens) | 75.8% | |

`build_context()` trails because it keeps the latest turns in the budget,
which helps a live agent but rarely holds the answer to a LoCoMo question.
Details, the published J table and the run commands are in
[`MEM0_COMPARISON.md`](https://github.com/saswattulo/NoMem/blob/main/benchmarks/MEM0_COMPARISON.md#end-to-end-answer-accuracy-llm-judged-j).

## Reproduce

```bash
pip install -e ".[embed]" "mem0ai[nlp]" fastembed
python -m spacy download en_core_web_sm
python benchmarks/mem0_bench.py --system mem0  --embedder potion
python benchmarks/mem0_bench.py --system nomem --embedder potion

# answer accuracy (LLM in the benchmark only; needs OPENAI_API_KEY)
python benchmarks/qa_bench.py --systems nomem,nomem-search,mem0,full,none
```
