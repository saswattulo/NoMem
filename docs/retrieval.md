# Retrieval and vectors

## Searching directly

`build_context()` uses retrieval for you, but you can search on its own:

```python
mem.search("where did I live before?")        # list[Message], best first
for hit in mem.search_hits("where did I live before?", k=3):
    print(round(hit.score, 4), hit.reasons, hit.message.content)
```

```text
0.0273 ['keyword#1', 'slot#2', 'context#2'] I live in Pune.
0.0145 ['slot#1', 'context#1'] I moved to Bangalore last week!
```

`reasons` names every ranker that proposed the message and its rank there.

## How ranking works

Each ranker returns up to `pool` (100) candidates, and the fusion sums what
each ranker gives a candidate (`fusion="score"`, the default):

- **keyword**: `weight * (bm25 / best bm25) ** 2`. BM25 has no fixed scale, so
  it is taken relative to the query's best keyword hit, and squared so a hit
  far behind the best one counts for little.
- **vector**: `weight * cosine`. Cosine already has a fixed scale, so a weak
  semantic match stays weak even when it is the best one.
- **entity, time, slot**: these only rank, so they add
  `weight * (rrf_k + 1) / (rrf_k + rank + 1)`, worth `weight` at rank 1.

Rank-only fusion (`fusion="rrf"`, NoMem 0.3's default) treats the keyword #1
and #20 as nearly equal and let long turns that repeat a query word outvote a
short turn that is semantically right; score fusion took LongMemEval turn
recall@5 from 57.1% to 78.7% with Model2Vec (see [Benchmarks](benchmarks.md)).

Then:

- an **importance prior** multiplies the score by
  `1 - w/2 + w * importance` (`importance_weight`, default 0.3), where
  importance comes from rules such as numbers, names, dates and first-person
  statements;
- an optional **recency prior** (`recency_half_life_days`, off by default)
  favours newer memories;
- a **long-turn penalty** scales hits over 400 characters by
  `(400 / length) ** 0.3`: a pasted document or a 3,000-character answer
  matches many query words by size alone;
- an **assistant cue**: when the question asks about something the assistant
  said ("the restaurant you recommended", "in our previous conversation"),
  assistant turns' scores are multiplied by 1.6 (`assistant_cue_boost`), and
  the reply next to a matching user turn gets the context lift even though
  its role differs. Such answers sit in the assistant's reply, often a long
  one, while the user's question shares more of the query's words. The cue
  is a regex on past-tense verbs after "you" and on "our last/previous
  conversation"; "remind me" or "we talked about" alone don't fire it, since
  they also ask for the user's own facts;
- **near-duplicates** (SimHash within 3 bits) are collapsed to one;
- the **context** ranker adds the turns either side of the top 5 hits, so an
  answer that spans two turns comes in whole. Only neighbours said in the
  same role get it (two people talking; an assistant's reply to a user's
  fact is usually advice about it, not more facts).

Search is scoped to the session: facts are per user, but the slot ranker
only points at messages in the session being searched.

If the embedder raises, `add()` stores the message without a vector and
search falls back to the other rankers (both warn). Vectors of a different
size (from an earlier embedder) are skipped with a warning.

```python
from nomem import NoMem, RetrievalConfig

mem = NoMem(retrieval=RetrievalConfig(
    weights={"keyword": 1.0, "entity": 0.5, "time": 0.5,
             "vector": 1.0, "context": 0.35, "slot": 0.5},
    recency_half_life_days=30,   # favour the last month
))
```

The defaults are what the benchmarks back; see
[Benchmarks](benchmarks.md) and `benchmarks/TRAJECTORY.md` for every
ablation behind them.

## Optional vectors

The core is model-free. For semantic matching, pass an embedder (an
embedding model, never a generative one):

```python
from nomem import NoMem, Model2VecEmbedder

mem = NoMem(embedder=Model2VecEmbedder())   # minishlab/potion-base-8M
```

`Model2VecEmbedder` (`pip install nomem-ai[embed]`) is the recommended
default: about 8M parameters, CPU only, about 30 MB, fast enough to embed
every message on write (about 0.4 ms each). It downloads once from the
Hugging Face Hub; pass a local path to run fully offline.

On LoCoMo it lifts search recall@5 from 59.6% to 62.3%, and on LongMemEval
(100-question spread) turn recall@5 from 72.0% to 78.7%.

Its cosine is added to the keyword score at weight 1.0. Under the old
rank-only fusion it had to stay a light tie-breaker (0.15), because a
static embedding's ranking alone is weaker than BM25's; with scores, a weak
cosine stays small and only a strong one moves a result.

| Embedder | Needs | Use it for |
| --- | --- | --- |
| `Model2VecEmbedder(model=...)` | `model2vec` | The default vector ranker. |
| `SentenceTransformerEmbedder(model="BAAI/bge-small-en-v1.5")` | `sentence-transformers` | A stronger encoder (63.9% LoCoMo recall@5). |
| `HashingEmbedder()` | nothing | Character n-grams, no model. Offline use and tests. |
| your own `Embedder` subclass | | Implement `encode(texts) -> list[list[float]]`. |

```python
from nomem import Embedder

class MyEmbedder(Embedder):
    def encode(self, texts):
        return my_model.embed(texts)   # list of float vectors
```

Vectors are stored in the same SQLite file, next to the messages.
