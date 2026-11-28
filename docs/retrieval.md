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

Each ranker returns up to `pool` (100) candidates. Reciprocal Rank Fusion
gives each candidate `weight / (rrf_k + rank)` per ranker it appears in, and
sums them. Then:

- an **importance prior** multiplies the score by
  `1 - w/2 + w * importance` (`importance_weight`, default 0.3), where
  importance comes from rules such as numbers, names, dates and first-person
  statements;
- an optional **recency prior** (`recency_half_life_days`, off by default)
  favours newer memories;
- **near-duplicates** (SimHash within 3 bits) are collapsed to one;
- the **context** ranker adds the turns either side of the top 5 hits, so an
  answer that spans two turns comes in whole.

```python
from nomem import NoMem, RetrievalConfig

mem = NoMem(retrieval=RetrievalConfig(
    weights={"keyword": 1.0, "entity": 0.5, "time": 0.5,
             "vector": 0.15, "context": 0.2, "slot": 0.5},
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

On LoCoMo it lifts search recall@5 from 56.8% to 58.6%, and on LongMemEval
turn recall@5 from 52.7% to 55.9%.

The vector ranker is a light tie-breaker by default (`weights["vector"] =
0.15`). Static embeddings are weaker than keyword matching on their own
(41.7% recall@5 vector-only on LoCoMo) and pull recall *down* at equal weight.
With a stronger encoder, raise the weight.

| Embedder | Needs | Use it for |
| --- | --- | --- |
| `Model2VecEmbedder(model=...)` | `model2vec` | The default vector ranker. |
| `SentenceTransformerEmbedder(model="BAAI/bge-small-en-v1.5")` | `sentence-transformers` | A stronger encoder; raise `weights["vector"]`. |
| `HashingEmbedder()` | nothing | Character n-grams, no model. Offline use and tests. |
| your own `Embedder` subclass | | Implement `encode(texts) -> list[list[float]]`. |

```python
from nomem import Embedder

class MyEmbedder(Embedder):
    def encode(self, texts):
        return my_model.embed(texts)   # list of float vectors
```

Vectors are stored in the same SQLite file, next to the messages.
