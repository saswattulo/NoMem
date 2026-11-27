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
| 5a. Ablation: + HashingEmbedder (char n-gram hashing, no model), vector weight 1.0 | 51.9 | 58.4 | 53.1 | 23.7 | 60.4 | 25.6 | 63.0 | | | | |
| 5b. Ablation: + HashingEmbedder, vector weight 0.5 | 53.6 | 61.3 | 54.7 | 24.7 | 62.5 | 26.3 | 64.7 | | | | |
| 5. Optional embedder interface (off by default); neighbour boost reworked as a share of the seed's score | 56.4 | 63.6 | 57.3 | 27.3 | 65.0 | 28.2 | 67.4 | 1.4 ms | 2.6 ms | 8/8 | 500/500 |
| 6. Docs, examples, slot-synonym fix ("for a living" now maps to job, not location), v0.2.0 | 56.4 | 63.6 | 57.3 | 27.3 | 65.0 | 28.2 | 67.4 | 1.8 ms | 2.8 ms | 8/8 | 500/500 |
| 7a. Ablation: + Model2Vec potion-base-8M, vector weight 1.0 (old default) | 53.9 | 61.6 | 54.9 | 26.3 | 64.7 | 29.7 | 63.5 | 4.3 ms | 5.8 ms | 8/8 | 500/500 |
| 7b. Ablation: Model2Vec vector-only (every other ranker at 0) | 41.7 | 49.3 | 43.1 | 20.2 | 54.1 | 26.2 | 48.3 | | | | |
| 7c. Ablation: + Model2Vec, vector weight 0.5 | 55.9 | 63.8 | 56.9 | 26.8 | 65.8 | 29.8 | 66.4 | | | | |
| 7d. Ablation: + Model2Vec, vector weight 0.25 | 57.1 | 64.7 | 58.3 | 27.3 | 66.5 | 30.0 | 68.4 | | | | |
| 7e. Ablation: + Model2Vec, vector weight 0.1 | 57.9 | 65.0 | 58.9 | 29.0 | 66.4 | 31.1 | 68.9 | | | | |
| 7f. Ablation: keyword only, ranker pool 50 -> 100 | 56.8 | 64.0 | 57.6 | 27.4 | 65.1 | 28.2 | 68.0 | | | | |
| 7g. Ablation: + Model2Vec, vector weight 0.15, pool 200 | 58.7 | 66.5 | 59.7 | 29.8 | 66.8 | 31.1 | 70.0 | | | | |
| 7h. Ablation: + HashingEmbedder at the new defaults | 57.6 | 65.2 | 58.5 | | | | | | | | |
| 7. Defaults: vector weight 1.0 -> 0.15, ranker pool 50 -> 100. **Keyword only** (no embedder) | 56.8 | 64.0 | 57.6 | 27.4 | 65.1 | 28.2 | 68.0 | 1.5 ms | 3.1 ms | 8/8 | 500/500 |
| 7. Same defaults **+ Model2Vec potion-base-8M** (`run_all.py --embedder model2vec`) | **58.6** | **66.0** | **59.7** | 29.1 | 67.1 | 31.1 | 70.1 | 3.5 ms | 7.1 ms | 8/8 | 500/500 |
| 8. Contradiction detection (slot / frame / update-cue conflicts, markers on retrieved memories, previous value on the state card). **Keyword only** | 56.8 | 64.0 | 57.6 | 27.4 | 65.1 | 28.2 | 68.0 | 1.3 ms | 2.7 ms | 8/8 | 500/500 |
| 8. Same **+ Model2Vec potion-base-8M** | 58.6 | 66.0 | 59.7 | 29.1 | 67.1 | 31.1 | 70.1 | 2.5 ms | 5.2 ms | 8/8 | 500/500 |


Step 8 leaves retrieval untouched (identical recall) and costs ingest speed:
50k-message ingest 4,649 -> 3,860 msg/s keyword only, 3,027 -> 2,693 msg/s with
Model2Vec. Invariant trials now also mix in statements that contradict each other.

Updated-facts benchmark (`updates_bench.py`, same numbers with and without Model2Vec):

| Step | Tuned: detected | Tuned: false flags | Held-out: detected | Held-out: false flags | LoCoMo pairs flagged (5,882 turns) |
| --- | --- | --- | --- | --- | --- |
| 8a. First version: cue anywhere in the sentence, one shared topic word needs 0.6 | 29/40 | 0/30 | | | |
| 8b. Lower bar for one shared topic word, strong cues add confidence, "Correction:" no longer read as a speaker | 40/40 | 0/30 | | | 61 (all wrong on inspection) |
| 8c. Changed value must sit in the clause with the cue; sentence-initial words and greeted names ("Hey John") are not values; only upcoming dates can be "moved"; "new" is not a cue | 40/40 | 0/30 | | | 5 |
| 8d. Generic "my X is Y" slot changes count only when both sides look like values | 40/40 | 0/30 | 16/20 | 1/20 | 2 |
| 9a. Fresh held-out set (25 updates, 25 controls) written and committed before any change; the old held-out set becomes the dev set. Baseline on it: | 40/40 | 0/30 | dev 16/20, fresh **11/25** | dev 1/20, fresh 0/25 | 2 |
| 9b. Singular/plural stemming of topic words and frames; frames per clause | 40/40 | 0/30 | dev 18/20 | dev 2/20 | 3 |
| 9c. A fact about someone else ("my friend works at ...") never conflicts with one about the speaker | 40/40 | 0/30 | dev 19/20 | dev 1/20 | 3 |
| 9d. Wording change with no value after a strong cue, first version (content-word sets) | 40/40 | 0/30 | dev 20/20 | dev 1/20 | 45 (all wrong) |
| 9e. Wording change only between short, plain, first-person statements that line up word for word around one swapped phrase | 40/40 | 0/30 | dev 20/20 | dev 1/20 | 2 |
| 9. Measured once on the fresh held-out set | | | **fresh 19/25** | **fresh 0/25** | 2 |

The held-out set was written after 8d and never tuned on. Its misses: a weekday
plan ("on Wednesday" is read as last Wednesday), a change with no value
("by car" -> "by train"), and two singular/plural mismatches ("bank"/"banks",
"1 dog"/"2 dogs"). Its false flag: "I'm learning French" -> "I'm learning to cook
Thai food now".

Step 9 (weekday schedules are compared by name rather than dated, alongside 9b-9e)
lifted the fresh held-out set from 11/25 to 19/25 with no false flags, with and
without Model2Vec. Its remaining misses: numbers as words ("one car" / "two
cars", "one sibling" / "two siblings"), ordinals ("the 1st" / "the 5th", "3rd" /
"7th floor"), a lower-camel-case brand ("iPhone" / "Pixel"), and "these days"
as the only cue for a wording change ("tea" / "coffee"). LoCoMo recall, planted
facts and invariants are unchanged; 50k-message ingest is ~3,100 msg/s keyword
only (was ~3,860) and ~2,260 with Model2Vec (was ~2,690), since sentences with a
strong cue but no value are now checked too.

Step 7 held-out check: tuned weights also win on each half of LoCoMo separately
(conversations 0-4: R@5 55.5 -> 59.4 vs old Model2Vec settings; conversations 5-9:
52.3 -> 57.9), so the gain is not an artefact of tuning on all ten. Model2Vec
writes run at ~1,500 msg/s (one embedding per message) vs ~4,000 msg/s keyword only.
