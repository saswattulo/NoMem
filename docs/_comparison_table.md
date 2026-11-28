| Same questions, no LLM on either side | NoMem | Mem0 (OSS 2.2.1, `infer=False`) |
| --- | --- | --- |
| LoCoMo search recall@5 (potion-base-8M) | **62.3%** | 55.9% |
| LoCoMo search recall@5 (bge-small) | **63.9%** | 59.1% |
| LongMemEval session recall@5 | **92.5%** | 92.2% |
| LongMemEval turn recall@5 | **78.7%** | 69.0% |
| Write latency, p50 | **0.7 ms** | 10 ms |
| Search latency, p50 | **8 ms** | 72 ms |
| LLM calls per write as normally used | **0** | 1 (Mem0's default `infer=True`) |
