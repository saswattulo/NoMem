# Benchmark trajectory

Every change on the LLM-free roadmap is followed by `python benchmarks/run_all.py`.
LoCoMo: 1,531 answerable questions (categories 1-4), 1,000-token context, `ApproxCounter`.
"R@k" = evidence turn in the top-k search results; "ctx" = evidence turn in the final budgeted context.
Scale rows use synthetic chat with 8 planted facts; latency is per `build_context(query=...)`.

| Step | R@5 | R@10 | ctx | multi-hop ctx | temporal ctx | open-dom ctx | single-hop ctx | query p50 @10k | query p50 @50k | planted @50k | invariants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0. Baseline (main: FTS5 only) | 50.9 | 58.4 | 52.5 | 25.0 | 60.8 | 27.9 | 61.2 | 34.9 ms | 189.6 ms | 7/8 | 500/500 |
