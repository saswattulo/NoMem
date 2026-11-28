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

Now the slice can stretch, up to `archival_max_ratio` (default 0.6) of the
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

Defaults were picked on LoCoMo; LongMemEval was only used as a check.
LoCoMo is flat for any ceiling from 0.45 to 0.8, so 0.6 was kept as the
ceiling that still leaves 40% of the budget for the recent conversation.

| Ablation (keyword only) | LoCoMo ctx | LongMemEval ctx (100-q spread) |
| --- | --- | --- |
| 11a. Fixed slice (`archival_max_ratio=0.3`, old behaviour) | 57.6 | 42.0 |
| 11b. Stretch in rank order, one pass, ceiling 0.6 | 58.1 | 43.7 (single-session-user 75 -> 61: a long wrong match at rank 1 crowded out short right ones) |
| 11c. Two passes (fill fixed slice first, then stretch), ceiling 0.45 | 58.1 | 48.2 |
| 11d. Two passes, ceiling 0.6 | 58.1 | 49.9 |
| 11e. Two passes, ceiling 0.8 | 58.1 | 54.4 |
| 11f. Two passes, ceiling 0.6, stretch only for the top half of ranks | 57.6 | 44.2 |
| 11. Two passes, ceiling 0.6, direct matches only (rule 2) | **58.1** | **49.9** |

Full runs at the step 11 defaults (search recall is unchanged; only what
reaches the context moves):

| Benchmark | Keyword only: ctx before -> after | + Model2Vec: ctx before -> after |
| --- | --- | --- |
| LoCoMo (1,531 q) | 57.6 -> 58.1 | 59.7 -> 59.9 |
| LongMemEval (470 q) | 41.1 -> **47.7** | 44.2 -> **51.1** |
| single-session-user | 78.1 -> 84.4 | 82.8 -> 89.8 |
| single-session-assistant | 16.1 -> 41.1 | 19.6 -> 41.1 |
| single-session-preference | 36.7 -> 40.0 | 35.0 -> 38.3 |
| multi-session | 31.4 -> 34.4 | 34.5 -> 38.1 |
| knowledge-update | 44.7 -> 52.1 | 43.1 -> 51.9 |
| temporal-reasoning | 41.7 -> 44.1 | 47.8 -> 50.8 |

Stress, invariants and the updated-facts benchmark are unchanged. Context
recall is still well below turn R@10 (47.7 vs 74.5): `retrieval_top_k=5` still
caps how many matches are considered, and a ceiling of 0.8 would add another
~4.5 points on LongMemEval at the cost of recent conversation. Both are left
as knobs rather than retuned to LongMemEval.
