| Same questions, no LLM on either side | NoMem | Mem0 (OSS 2.2.1, `infer=False`) |
| --- | --- | --- |
| LoCoMo search recall@5 (potion-base-8M) | **58.6%** | 55.9% |
| LoCoMo search recall@5 (bge-small) | **59.5%** | 59.1% |
| LongMemEval session recall@5 | **92.8%** | 92.2% |
| LongMemEval turn recall@5 | 57.1% | **69.0%** |
| Write latency, p50 | **0.6 ms** | 10 ms |
| Search latency, p50 | **6 ms** | 72 ms |
| LLM calls per write as normally used | **0** | 1 (Mem0's default `infer=True`) |
