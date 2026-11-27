# Benchmark trajectory

Every change on the LLM-free roadmap is followed by `python benchmarks/run_all.py`.
LoCoMo: 1,531 answerable questions (categories 1-4), 1,000-token context, `ApproxCounter`.
"R@k" = evidence turn in the top-k search results; "ctx" = evidence turn in the final budgeted context.
Scale rows use synthetic chat with 8 planted facts; latency is per `build_context(query=...)`.

| Step | R@5 | R@10 | ctx | multi-hop ctx | temporal ctx | open-dom ctx | single-hop ctx | query p50 @10k | query p50 @50k | planted @50k | invariants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0. Baseline (main: FTS5 only) | 50.9 | 58.4 | 52.5 | 25.0 | 60.8 | 27.9 | 61.2 | 34.9 ms | 189.6 ms | 7/8 | 500/500 |
| 1. Read only as many recent turns as the budget can hold | 50.9 | 58.4 | 52.5 | 25.0 | 60.8 | 27.9 | 61.2 | 0.8 ms | 1.6 ms | 7/8 | 500/500 |
| 2a. Ablation: keyword only through the new fusion path | 50.9 | 58.4 | 52.5 | 25.0 | 60.8 | 27.9 | 61.2 | | | | |
| 2b. Ablation: + importance prior | 53.2 | 59.6 | 54.5 | 26.8 | 64.9 | 28.9 | 62.5 | | | | |
| 2c. Ablation: keyword + entity ranker (no df cap) | 40.0 | 54.9 | 41.7 | 17.3 | 51.8 | 25.7 | 47.6 | | | | |
| 2d. Ablation: keyword + entity ranker, entities in >2% of turns ignored | 51.2 | 58.6 | 52.8 | 24.8 | 61.2 | 28.2 | 61.6 | | | | |
| 2e. Ablation: keyword + time ranker | 52.5 | 59.5 | 54.1 | 24.6 | 61.9 | 28.4 | 63.6 | | | | |
| 2f. Ablation: keyword + neighbouring-turn ranker | 53.4 | 62.0 | 54.8 | 25.8 | 60.4 | 27.6 | 65.2 | | | | |
| 2. Annotations + hybrid retrieval (all rankers, RRF, importance, dedup, dated recall) | 56.5 | 63.1 | 57.7 | 27.5 | 65.3 | 29.9 | 67.8 | 1.1 ms | 2.4 ms | 7/8 | 500/500 |
| 3a. Ablation: + slot ranker at weight 1.0, any subject | 54.8 | 62.4 | 55.9 | 26.4 | 65.0 | 28.9 | 65.1 | | | | |
| 3b. Ablation: + slot ranker, subject-filtered, weight 0.5 | 56.3 | 63.2 | 57.3 | 27.4 | 65.0 | 28.2 | 67.5 | | | | |
| 3c. Ablation: + state card listing every current fact | 56.3 | 63.2 | 56.6 | 26.6 | 63.7 | 27.6 | 66.9 | | | | |
| 3. Slot ranker + query-relevant state card + superseded-fact notes | 56.3 | 63.2 | 57.3 | 27.4 | 65.0 | 28.2 | 67.4 | 2.2 ms | 2.7 ms | **8/8** | 500/500 |
| 4a. Ablation: extractive summary slice on (summary_ratio 0.1) | 56.3 | 63.2 | 56.4 | 26.5 | 63.7 | 27.6 | 66.7 | | | | |
| 4. Extractive compactor, opt-in (summary_ratio default 0); invariants now randomise facts/summary ratios | 56.3 | 63.2 | 57.3 | 27.4 | 65.0 | 28.2 | 67.4 | 1.3 ms | 4.2 ms | 8/8 | 500/500 |
