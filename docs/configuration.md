# Configuration

## `NoMem(...)`

```python
NoMem(
    max_tokens=3000,
    core="",
    *,
    user_id="default",
    session_id="default",
    db=":memory:",
    counter=None,
    retrieval_top_k=5,
    archival_ratio=0.3,
    archival_max_ratio=0.8,
    retrieval=None,
    embedder=None,
    annotate=True,
    facts_ratio=0.1,
    summary_ratio=0.0,
    compactor=None,
    conflicts=None,
)
```

| Argument | Default | What it does |
| --- | --- | --- |
| `max_tokens` | `3000` | Hard cap on the assembled context. Never exceeded. |
| `core` | `""` | Fixed persona or rules, always included first. |
| `user_id` | `"default"` | Scope for facts and conflicts (shared across sessions). |
| `session_id` | `"default"` | Scope for messages. |
| `db` | `":memory:"` | SQLite path, `":memory:"`, or a shared `SQLiteStore`. |
| `counter` | best available | A `TokenCounter`. `OpenAICounter` if `tiktoken` is installed, else `ApproxCounter`. |
| `retrieval_top_k` | `5` | Memories considered per query. |
| `archival_ratio` | `0.3` | Budget share (after core) reserved for retrieved memories. Unused share rolls over. |
| `archival_max_ratio` | `0.8` | How far that share may stretch for a direct match that does not fit, taking tokens from the oldest recent turns. Set it to `archival_ratio` for a fixed slice. At 0.8 a long match can leave 20% for recent conversation; use 0.6 to keep more. |
| `retrieval` | `RetrievalConfig()` | Ranker weights and priors (below). |
| `embedder` | `None` | Optional vector ranker. `None` keeps NoMem model-free. |
| `annotate` | `True` | Rule-based annotation on write (entities, dates, facts). Turning it off also turns off facts and conflicts. |
| `facts_ratio` | `0.1` | Most of the budget the Known-facts card may use. `0` disables it. |
| `summary_ratio` | `0.0` | Budget share for an extractive summary of turns that just left the window. Off by default. |
| `compactor` | `ExtractiveCompactor()` | How dropped turns are summarised. `NullCompactor()` for none. |
| `conflicts` | `ConflictConfig()` | Contradiction detection (below). |

## `RetrievalConfig`

```python
from nomem import RetrievalConfig
```

| Field | Default | What it does |
| --- | --- | --- |
| `pool` | `100` | Candidates taken from each ranker. |
| `fusion` | `"score"` | `"score"` adds keyword and vector scores; `"rrf"` is rank-only Reciprocal Rank Fusion. |
| `rrf_k` | `60` | RRF constant (Cormack et al., 2009); also shapes the rank bonus of rank-only rankers. |
| `keyword_power` | `2.0` | BM25 share of the query's best hit is raised to this power (`"score"` only). |
| `weights` | `keyword 1.0, entity 0.5, time 0.5, vector 1.0, context 0.35, slot 0.5` | Weight of each ranker in the fusion. |
| `length_penalty` | `0.3` | Hits longer than `length_pivot` characters are scaled by `(pivot / length) ** penalty`. |
| `length_pivot` | `400` | Length (characters) where the penalty starts. |
| `context_same_role` | `True` | Only neighbours said in the same role as the hit get the context lift. |
| `assistant_cue_boost` | `1.6` | For a question about what the assistant said ("the hotel you recommended"), assistant turns' scores are multiplied by this, and a user turn's reply gets the context lift. `1.0` turns it off. |
| `max_query_terms` | `64` | Longest distinct query words sent to the keyword index. |
| `importance_weight` | `0.3` | Strength of the importance prior. |
| `recency_half_life_days` | `None` | Recency prior; `None` turns it off. |
| `near_dup_bits` | `3` | SimHash distance treated as a near-duplicate. |
| `context_span` | `1` | Neighbouring turns added either side of a top hit. |
| `context_seeds` | `5` | How many top hits get neighbours. |
| `entity_max_df` | `0.02` | Ignore entities that appear in more than this share of messages. |

`RetrievalConfig.legacy()` returns NoMem 0.3's retrieval (RRF, vector 0.15,
context 0.2, no length penalty, no assistant cue) for comparison.

## `ConflictConfig`

```python
from nomem import ConflictConfig
```

| Field | Default | What it does |
| --- | --- | --- |
| `enabled` | `True` | Run the detectors at all. |
| `frame` | `True` | The same-sentence, changed-value detector. |
| `update` | `True` | The update-cue detector. |
| `candidates` | `20` | Earlier messages checked per update sentence. |
| `min_confidence` | `0.6` | Weakest link that is stored. |
| `vector_min_similarity` | `0.55` | Cosine that counts as "same topic" when an embedder is set. |
| `max_per_message` | `3` | Most links a new message may create. |
| `show_min_confidence` | `0.6` | Weakest link shown as a `[CONFLICT: ...]` marker. |
| `show_per_memory` | `2` | Markers per retrieved memory; `0` hides them. |
| `quote_chars` | `90` | Longest quote of the other memory in a marker. |

## Sharing one database

Several `NoMem` objects can share one store, for example one per session:

```python
from nomem import NoMem, SQLiteStore

store = SQLiteStore("app.db")
alice_work = NoMem(user_id="alice", session_id="work", db=store)
alice_home = NoMem(user_id="alice", session_id="home", db=store)
```

Facts are shared between Alice's two sessions; messages are not. For a
server with many users, [`MemoryService`](server.md#python-memoryservice)
does this for you and adds a lock.
