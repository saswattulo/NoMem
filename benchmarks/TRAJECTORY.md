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

## LongMemEval (step 10)

Step 10 adds `longmemeval_bench.py` as a second retrieval benchmark, so later
changes are not tuned to LoCoMo alone. No defaults were changed for it: these
are the step 9 defaults, measured as they are. All 470 answerable questions of
`longmemeval_s` (231,606 turns in total, one fresh NoMem per question),
1,000-token context, `ApproxCounter`, each question asked at its own date.
"Session R@k" = share of evidence sessions among the top-k sessions; "all@5" =
questions with every evidence session in the top 5; "turn R@k" = share of
evidence turns among the top-k turns; "ctx" = evidence turn text in the final
context.

| Config | Session R@5 | Session R@10 | all@5 | Turn R@5 | Turn R@10 | ctx | search p50 | add p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keyword only | 89.9 | 94.7 | 82.1 | 52.7 | 74.5 | 41.1 | 6.8 ms | 2.4 / 38.3 ms |
| + Model2Vec potion-base-8M | **92.1** | **95.8** | **85.1** | **55.9** | **77.2** | **44.2** | 11.3 ms | 2.7 / 52.2 ms |

By question type (turn R@5 / ctx, keyword only -> + Model2Vec):

| Type | n | Session R@5 | Turn R@5 | ctx |
| --- | --- | --- | --- | --- |
| single-session-user | 64 | 98.4 -> 100.0 | 85.9 -> 90.6 | 78.1 -> 82.8 |
| single-session-assistant | 56 | 100.0 -> 100.0 | 76.8 -> 76.8 | 16.1 -> 19.6 |
| single-session-preference | 30 | 73.3 -> 73.3 | 40.0 -> 38.3 | 36.7 -> 35.0 |
| multi-session | 121 | 83.7 -> 88.1 | 35.1 -> 39.2 | 31.4 -> 34.5 |
| knowledge-update | 72 | 98.6 -> 99.3 | 54.2 -> 52.5 | 44.7 -> 43.1 |
| temporal-reasoning | 127 | 86.2 -> 88.8 | 44.1 -> 51.2 | 41.7 -> 47.8 |

Where NoMem is weak on LongMemEval (not fixed here, reported only):

- **Context recall trails search recall by a lot** (44.2 vs turn R@10 77.2).
  `build_context` takes only `retrieval_top_k=5` hits into a slice of 30% of
  the budget (~300 tokens at 1,000), and LongMemEval turns are long. Worst on
  single-session-assistant: the evidence is found (turn R@10 95) but the
  assistant turn rarely fits (ctx 20). At a 4,000-token budget ctx only rises
  to 50.2 (keyword only, 100-question spread), because the top-5 cap still binds.
- **Preference questions** ("can you suggest...") are the weakest type: turn
  R@5 38-40, session R@5 73. The evidence is something the user said about
  themselves in other words; Model2Vec does not close the gap.
- **Multi-session questions** need several sessions at once: only 64.5%
  (72% with vectors) get every evidence session into the top 5, and turn R@5
  is 35-39.
- **Temporal-reasoning** turn R@5 is 44-51; vectors help most here (+7).
- **Writes on long turns are slow.** Ingest runs at ~130 turns/s here vs
  ~2,800 msg/s on short synthetic chat. Profiling 3 haystacks before step 9:
  contradiction detection was ~55% of `add()` time on long turns
  (`conflicts.sentence_values` / `clause_values`), then annotation (date
  parsing, simhash). Step 9's rules tripled the slow tail here: add p95
  12.4 -> 38.3 ms keyword only (13.4 -> 52.2 ms with Model2Vec), with no
  change to LoCoMo's add p95 (~1.5 ms), whose turns are short.
- **A crash, now fixed:** `add()` raised `ValueError` on text like "Rome was
  founded 4000 years ago" (the year left datetime's range). Out-of-range
  relative dates are now skipped.

Measured on main after step 9 (merged). Recall is identical to a run on step 8;
context recall moves by up to 3.4 points per type because conflict markers now
take some of the budget. LoCoMo, stress, invariants and the updated-facts
benchmark are unchanged by step 10 (same numbers as step 9, with and without
Model2Vec).

## Flexible archival slice (step 11)

Before step 11 retrieved memories got a fixed 30% of the post-core budget, and
a match that didn't fit was skipped, even when the recent window held only
older, unrelated turns. LongMemEval showed the cost: the right turn was often
found but too long to fit (single-session-assistant: turn R@10 93, ctx 16).

Now the slice can stretch, up to `archival_max_ratio` (default 0.8) of the
budget, taking tokens from the oldest turns of the recent window. `max_tokens`
is still a hard cap (0 violations in 500 randomised trials, which now also
vary `archival_max_ratio`). Rules:

1. The fixed slice is filled first exactly as before; stretching only adds
   matches that it passed over, best first, so it never displaces one.
2. Only for direct matches: never for a turn found only as the neighbour of
   another match (`context` ranker), and never for a turn the window already
   shows with the fixed slice (that would just move it).
3. Every stretch is logged: `slice stretched 31 -> 55 tok, taken from the recent window`.

Rule 2 came from the README demo, where a first version stretched for a
neighbour-only turn and dropped a more useful recent one. It costs nothing on
either benchmark.

LoCoMo is flat for any ceiling from 0.45 to 0.8. 0.6 was proposed first (it
leaves 40% of the budget for recent conversation); the default is 0.8 by
Saswat's choice, after seeing that LongMemEval gains more from it.

| Ablation (keyword only) | LoCoMo ctx | LongMemEval ctx (100-q spread) |
| --- | --- | --- |
| 11a. Fixed slice (`archival_max_ratio=0.3`, old behaviour) | 57.6 | 42.0 |
| 11b. Stretch in rank order, one pass, ceiling 0.6 | 58.1 | 43.7 (single-session-user 75 -> 61: a long wrong match at rank 1 crowded out short right ones) |
| 11c. Two passes (fill fixed slice first, then stretch), ceiling 0.45 | 58.1 | 48.2 |
| 11d. Two passes, ceiling 0.6 | 58.1 | 49.9 |
| 11e. Two passes, ceiling 0.8 | 58.1 | 54.4 |
| 11f. Two passes, ceiling 0.6, stretch only for the top half of ranks | 57.6 | 44.2 |
| 11g. Two passes, ceiling 0.6, direct matches only (rule 2) | 58.1 | 49.9 |
| 11. Two passes, ceiling 0.8, direct matches only (rule 2) | **58.1** | **54.4** |

Full runs at the step 11 defaults (search recall is unchanged; only what
reaches the context moves):

| Benchmark | Keyword only: ctx before -> 0.6 -> **0.8** | + Model2Vec: ctx before -> 0.6 -> **0.8** |
| --- | --- | --- |
| LoCoMo (1,531 q) | 57.6 -> 58.1 -> **58.1** | 59.7 -> 59.9 -> **59.9** |
| LongMemEval (470 q) | 41.1 -> 47.7 -> **50.0** | 44.2 -> 51.1 -> **53.2** |
| single-session-user | 78.1 -> 84.4 -> 85.9 | 82.8 -> 89.8 -> 89.8 |
| single-session-assistant | 16.1 -> 41.1 -> 57.1 | 19.6 -> 41.1 -> 60.7 |
| single-session-preference | 36.7 -> 40.0 -> 40.0 | 35.0 -> 38.3 -> 38.3 |
| multi-session | 31.4 -> 34.4 -> 34.8 | 34.5 -> 38.1 -> 38.1 |
| knowledge-update | 44.7 -> 52.1 -> 52.8 | 43.1 -> 51.9 -> 50.5 |
| temporal-reasoning | 41.7 -> 44.1 -> 44.1 | 47.8 -> 50.8 -> 50.8 |

Stress, invariants and the updated-facts benchmark are unchanged. Context
recall is still well below turn R@10 (50.0 vs 74.5): `retrieval_top_k=5` still
caps how many matches are considered. At 0.8 a long match can leave only 20%
of the budget for recent conversation; set `archival_max_ratio=0.6` to keep more.

## Score fusion (step 12)

PR #8 found that LongMemEval's turn-level gap to Mem0 came from fusion:
rank-only RRF let long turns that repeat a query word outvote a short turn
that is semantically right. Step 12 replaces it with score fusion and adds a
long-turn penalty. Tuned on LoCoMo plus the usual 100-question LongMemEval
spread, then checked once on a **held-out** set of 124 LongMemEval questions
(every third answerable question not in the 100).

Search R@5, Model2Vec potion-base-8M unless marked:

| Step | LoCoMo R@5 | LME turn R@5 | LME session R@5 |
| --- | --- | --- | --- |
| 12a. Before (RRF, vector 0.15, context 0.2) | 58.6 | 57.1 | 92.8 |
| 12b. Score fusion: BM25 / best BM25 + raw cosine, vector 1.0 | 59.7 | 67.2 | 92.8 |
| 12c. + BM25 share squared | 60.7 | 69.4 | 92.5 |
| 12d. + context lift only for same-role neighbours | 60.7 | 70.6 | 92.5 |
| 12e. + long-turn penalty `(400/len)**0.3` | 60.7 | 77.7 | 92.5 |
| 12f. Ablation: penalty 0.5 (single-session-assistant 81.8 -> 72.7) | 60.7 | 78.3 | 92.9 |
| 12g. Ablation: penalty with pivot 800 chars | 60.7 | 73.7 | 91.9 |
| 12h. Ablation: cosine min-max normalised instead of raw | 57.4 | 79.2 | 95.3 |
| 12i. Ablation: vector weight 1.5 (held-out 76.5 vs 76.1: not adopted) | 61.4 | 79.8 | 94.1 |
| 12j. Ablation: no context lift | 56.4 | 69.6 | 92.8 |
| 12. + context weight 0.2 -> 0.35 (**new defaults**) | **62.3** | **77.7** | 92.5 |
| 12. Same, keyword rules only (no embedder) | 59.6 | 72.0 | 88.2 |

Held-out 124 LongMemEval questions (never used for tuning), turn R@5:

| | Before | Step 12 |
| --- | --- | --- |
| + Model2Vec | 55.6 | **76.1** |
| Keyword only | 52.1 | **71.4** |
| Step 12 without the long-turn penalty | | 71.0 |

Full `run_all.py` at the new defaults:

| Benchmark | Keyword only: before -> after | + Model2Vec: before -> after |
| --- | --- | --- |
| LoCoMo R@5 | 56.8 -> **59.6** | 58.6 -> **62.3** |
| LoCoMo R@10 | 64.0 -> 67.8 | 66.0 -> 70.3 |
| LoCoMo context recall | 58.1 -> 60.8 | 59.9 -> 63.4 |
| LongMemEval turn R@5 | 55.2 -> **72.0** | 57.1 -> **77.7** |
| LongMemEval turn R@10 | 74.6 -> 79.0 | 75.6 -> 85.7 |
| LongMemEval session R@5 | 89.1 -> 88.2 | 92.8 -> 92.5 |
| LongMemEval context recall | 54.4 -> 70.6 | 55.8 -> 75.7 |
| single-session-user turn R@5 | 85.7 -> 82.1 | 78.6 -> 82.1 |
| single-session-assistant | 63.6 -> 81.8 | 63.6 -> 81.8 |
| single-session-preference | 41.7 -> 50.0 | 41.7 -> 66.7 |
| multi-session | 40.5 -> 53.1 | 42.8 -> 63.5 |
| knowledge-update | 59.4 -> 87.5 | 59.4 -> 90.6 |
| temporal-reasoning | 50.6 -> 76.5 | 59.3 -> 82.1 |
| search p50 (LME) | 6.8 -> 6.7 ms | 11.4 -> 11.0 ms |

All 470 LongMemEval questions (`--longmemeval-limit 0`), before (step 11)
-> after:

| | Keyword only | + Model2Vec |
| --- | --- | --- |
| Turn R@5 | 52.7 -> **72.4** | 55.9 -> **78.5** |
| Session R@5 | 89.9 -> 89.7 | 92.1 -> **93.3** |
| Context recall | 50.0 -> **70.3** | 53.2 -> **76.5** |
| single-session-user turn R@5 | 91.4 | 94.5 |
| single-session-assistant | 83.9 | 82.1 |
| single-session-preference | 44.4 | 51.1 |
| multi-session | 58.3 | 69.2 |
| knowledge-update | 86.6 | 91.7 |
| temporal-reasoning | 69.8 | 76.6 |
| search p50 / p95 | 5.3 / 7.2 ms | 9.0 / 15.5 ms |

Stress (planted 8/8 at every scale), 500/500 invariants and the
updated-facts benchmark are unchanged. Session R@5 dips slightly (0.3-0.9);
LoCoMo multi-hop keyword-only dips 27.3 -> 26.7. `RetrievalConfig.legacy()`
restores the old ranking.

Hardening in the same step (tests in `tests/test_retrieval_robust.py`): the
slot ranker no longer returns messages from another session of the same
user; a failing embedder no longer loses the message or fails search; vectors
of another size are skipped with a warning; NaN vectors are neutralised; the
vector cache no longer keys on `id(store)` (reused after garbage collection);
queries keep their 64 longest distinct words; `RetrievalConfig` is validated.

## Assistant cue (step 13)

After step 12, Mem0 still led on single-session-assistant questions (90.9 vs
81.8 turn R@5 on the 100-question spread): the answer sits in an assistant
reply, often a long one, and the long-turn penalty plus the same-role
context rule let the user's question (which shares more of the query's
words) rank above it.

Tried first, and dropped: indexing every turn over 800 characters a second
time as ~400-character passages (keyword and vector), with a turn scored by
the better of itself and its best passage, the passage exempt from the
length penalty. On all 470 questions it cost the user-fact categories much
more than it gained (turn R@5 78.5 -> 64-76 depending on weight; preference
51 -> 33-44, multi-session 69 -> 48-64), because LongMemEval's haystacks are
full of long ShareGPT replies whose passages then match. Even applied only
to questions about the assistant, it gained at most 1 of the 56
single-session-assistant questions, while tripling the vector rows and FTS
entries per long turn.

What worked is a query cue (`annotate.asks_about_assistant`): past-tense
verbs after "you" ("you recommended", "did you suggest") or "our previous /
last conversation". When it fires, assistant turns' scores are multiplied by
`assistant_cue_boost` (1.6) and the reply next to a matching user turn gets
the context lift despite its role. "Remind me" and "we talked about" alone
don't fire it, since they also ask for the user's own facts. It fires on
51 of the 56 single-session-assistant questions and on none of LongMemEval's
other 414 answerable questions or LoCoMo's 1,986.

Tuned (boost 1.6; 1.6-2.0 is flat) on the 100-question spread, then checked
on the 124 held-out questions and on all 470:

| Turn R@5, + Model2Vec | Spread (100) | Held-out (124) | All 470 |
| --- | --- | --- | --- |
| single-session-assistant, before | 81.8 | 80.0 | 82.1 |
| single-session-assistant, after | **90.9** | **86.7** | **89.3** |
| Overall, before | 77.7 | 76.1 | 78.5 |
| Overall, after | **78.7** | **76.9** | **79.3** |

Keyword only, all 470: single-session-assistant 83.9 -> 87.5, overall
72.4 -> 72.9 (the spread is unchanged at 72.0). Every other question type,
LoCoMo (62.3 / 59.6 R@5), stress (planted 8/8, 0 of 500 invariant
violations) and the updated-facts benchmark are unchanged. LongMemEval
context recall through `build_context` rises 75.7 -> 77.7; packed search
results (`mem0_bench.py --system nomem`) dip 86.3 -> 85.3, as the assistant
turns that now rank higher are longer and fewer fit in 1,000 tokens.

Mem0 (OSS 2.2.1, `infer=False`, potion) on all 56 single-session-assistant
questions through `mem0_bench.py`'s harness: turn R@5 85.7, context 78.6;
NoMem on the same harness: 89.3 and 85.7.
