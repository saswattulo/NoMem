"""End-to-end answer accuracy on LoCoMo: the LLM-judged "J" score Mem0 reports.

NoMem itself never calls an LLM. This harness puts one *around* it, in the
evaluation only, so NoMem can be scored the same way Mem0 and others publish
LoCoMo results:

1. **Context** (no LLM). For every answerable LoCoMo question (categories 1-4,
   1,540 questions) each system builds the context an app would hand its model:

   * ``nomem``        - ``NoMem.build_context(query=q)``, the real API (state
                        card + retrieved memories + recent turns, one budget).
   * ``nomem-search`` - ``NoMem.search`` results packed best-first into the
                        budget, each with its date. Same packing as ``mem0``.
   * ``mem0``         - Mem0 OSS with ``infer=False`` (verbatim turns, its own
                        hybrid search), packed the same way. Needs ``mem0ai``.
   * ``full``         - the whole conversation with session dates (~20k
                        tokens): the "full-context" reference from Mem0's paper.
   * ``none``         - no memory at all: the floor.

2. **Answer** (LLM). The answer model gets Mem0's own LoCoMo answer prompt
   (``ANSWER_PROMPT`` from mem0's ``evaluation/prompts.py``) with the context.

3. **Judge** (LLM). A judge model labels the answer CORRECT or WRONG with
   Mem0's ``ACCURACY_PROMPT`` (their ``evaluation/metrics/llm_judge.py``).
   J = % CORRECT. LoCoMo's token F1 is reported too (no LLM).

Defaults follow Mem0's paper and open evaluation code: ``gpt-4o-mini`` answers
and judges, temperature 0, judge in JSON mode. Every LLM call is cached on
disk (``benchmarks/results/qa_cache.jsonl``), so re-runs and interrupted runs
cost nothing for calls already made.

Usage:
    # contexts only, no API key needed (also writes the JSONL the LLM stage reads)
    python benchmarks/qa_bench.py --stage contexts --systems nomem,nomem-search,mem0,full,none

    # answer + judge (reuses the saved contexts)
    OPENAI_API_KEY=... python benchmarks/qa_bench.py --systems nomem,nomem-search,mem0,full,none
    ANTHROPIC_API_KEY=... python benchmarks/qa_bench.py --provider anthropic

    # offline smoke test of the whole pipeline (a fake LLM; the score means nothing)
    python benchmarks/qa_bench.py --provider mock --limit 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import string
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from nomem import ApproxCounter  # noqa: E402

LOCOMO = ROOT / "benchmarks" / "data" / "locomo10.json"
RESULTS = ROOT / "benchmarks" / "results"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
SYSTEMS = ("nomem", "nomem-search", "mem0", "full", "none")
SEARCH_POOL = 100
TIME_FMT = "%I:%M %p on %d %B, %Y"

# Verbatim from mem0ai/mem0 evaluation/prompts.py (ANSWER_PROMPT).
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    {context}

    Question: {question}

    Answer:
    """

# Verbatim from mem0ai/mem0 evaluation/metrics/llm_judge.py (ACCURACY_PROMPT).
JUDGE_PROMPT = """
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {generated_answer}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """


# --------------------------------------------------------------------------
# Stage 1: contexts (no LLM)
# --------------------------------------------------------------------------

def _turns(conv: dict):
    """Yield (session_no, session_ts, index, turn) for one LoCoMo conversation."""
    n = 1
    while f"session_{n}" in conv:
        base = datetime.strptime(conv[f"session_{n}_date_time"].strip(), TIME_FMT).timestamp()
        for i, turn in enumerate(conv[f"session_{n}"]):
            yield n, base, i, turn
        n += 1


def _stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%-d %B %Y, %-I:%M %p")


def _pack(items, budget, counter):
    """items: (ts, text) best first -> dated lines that fit the budget, and tokens used."""
    used, lines = 0, []
    for ts, text in items:
        line = f"[{_stamp(ts)}] {text}"
        t = counter.count(line) + 1
        if used + t > budget:
            continue
        lines.append(line)
        used += t
    return "\n".join(lines), used


def _full_context(conv: dict, counter) -> tuple[str, int]:
    parts, cur = [], None
    for n, base, _, turn in _turns(conv):
        if n != cur:
            parts.append(f"\n--- Session {n} ({conv[f'session_{n}_date_time'].strip()}) ---")
            cur = n
        parts.append(f"{turn['speaker']}: {turn['text']}")
    text = "\n".join(parts).strip()
    return text, counter.count(text)


def _questions(sample: dict):
    for qi, qa in enumerate(sample["qa"]):
        if qa.get("category") in CATEGORIES and "answer" in qa:
            yield qi, qa


def _nomem_embedder(name: str | None):
    if name in (None, "none"):
        return {}
    if name == "potion":
        from nomem.embed import Model2VecEmbedder
        return {"embedder": Model2VecEmbedder()}
    raise SystemExit(f"unknown --embedder {name}")


def build_contexts(systems, budget=1000, embedder="potion", limit_convs=None, limit_q=None):
    """Return context records for every (system, question). Pure retrieval, no LLM."""
    from nomem import NoMem

    data = json.loads(LOCOMO.read_text())[:limit_convs]
    counter = ApproxCounter()
    out = []
    emb_kw = _nomem_embedder(embedder) if {"nomem", "nomem-search"} & set(systems) else {}
    mem0 = None
    if "mem0" in systems:
        from mem0_bench import Mem0Runner
        import logging
        logging.getLogger("mem0").setLevel(logging.ERROR)
        mem0 = Mem0Runner(embedder if embedder != "none" else "potion", infer=False, llm="")

    def rec(system, ci, qi, qa, context, tokens, seconds):
        return {"system": system, "budget": budget, "conv": ci, "qid": f"{ci}-{qi}",
                "category": CATEGORIES[qa["category"]], "question": qa["question"],
                "gold": str(qa["answer"]), "context": context, "context_tokens": tokens,
                "retrieval_ms": round(1000 * seconds, 2)}

    try:
        for ci, sample in enumerate(data):
            conv = sample["conversation"]
            qs = list(_questions(sample))[:limit_q]
            t_conv = time.perf_counter()
            if "nomem" in systems or "nomem-search" in systems:
                mem = NoMem(max_tokens=budget, counter=ApproxCounter(),
                            session_id=f"conv{ci}", **emb_kw)
                for _, base, i, turn in _turns(conv):
                    mem.add("user", f"{turn['speaker']}: {turn['text']}", timestamp=base + i * 30)
                for qi, qa in qs:
                    if "nomem" in systems:
                        t0 = time.perf_counter()
                        res = mem.build_context(query=qa["question"])
                        out.append(rec("nomem", ci, qi, qa, res.text, res.tokens_used,
                                       time.perf_counter() - t0))
                    if "nomem-search" in systems:
                        t0 = time.perf_counter()
                        hits = mem.search(qa["question"], SEARCH_POOL)
                        text, used = _pack([(h.created_at, h.content) for h in hits], budget, counter)
                        out.append(rec("nomem-search", ci, qi, qa, text, used,
                                       time.perf_counter() - t0))
                mem.close()
            if mem0 is not None:
                m, uid, meta = mem0.fresh(), f"conv{ci}", {}
                for _, base, i, turn in _turns(conv):
                    text = f"{turn['speaker']}: {turn['text']}"
                    for mid in mem0.add(m, "user", text, base + i * 30, uid):
                        meta[mid] = (base + i * 30, text)
                for qi, qa in qs:
                    t0 = time.perf_counter()
                    hits = mem0.search(m, qa["question"], SEARCH_POOL, uid)
                    items = [meta.get(h["id"], (0.0, h["memory"])) for h in hits]
                    text, used = _pack(items, budget, counter)
                    out.append(rec("mem0", ci, qi, qa, text, used, time.perf_counter() - t0))
            if "full" in systems:
                text, used = _full_context(conv, counter)
                for qi, qa in qs:
                    out.append(rec("full", ci, qi, qa, text, used, 0.0))
            if "none" in systems:
                for qi, qa in qs:
                    out.append(rec("none", ci, qi, qa, "(no memories)", 0, 0.0))
            print(f"  contexts conv {ci + 1}/{len(data)} ({time.perf_counter() - t_conv:.0f}s)",
                  file=sys.stderr)
    finally:
        if mem0 is not None:
            mem0.close()
    return out


# --------------------------------------------------------------------------
# Stage 2: answer + judge (LLM, eval only)
# --------------------------------------------------------------------------

class LLM:
    """One chat model behind a tiny interface, with an on-disk response cache."""

    def __init__(self, provider: str, model: str, cache: Path | None):
        self.provider, self.model = provider, model
        self.usage = Counter()
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._cache_path = cache
        if cache and cache.exists():
            for line in cache.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self._cache[row["k"]] = row["v"]
                except (json.JSONDecodeError, KeyError):
                    continue
        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(max_retries=8, timeout=120)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(max_retries=8, timeout=120)
        elif provider != "mock":
            raise SystemExit(f"unknown provider {provider}")

    def __call__(self, prompt: str, role: str = "user", json_mode: bool = False) -> str:
        key = hashlib.sha256(json.dumps([self.provider, self.model, role, json_mode, prompt])
                             .encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self.usage["cached"] += 1
                return self._cache[key]
        text = self._call(prompt, role, json_mode)
        with self._lock:
            self._cache[key] = text
            if self._cache_path and self.provider != "mock":
                with self._cache_path.open("a") as f:
                    f.write(json.dumps({"k": key, "v": text}) + "\n")
        return text

    def _call(self, prompt, role, json_mode) -> str:
        if self.provider == "mock":
            return _mock(prompt)
        if self.provider == "openai":
            kw = {"response_format": {"type": "json_object"}} if json_mode else {}
            r = self._client.chat.completions.create(
                model=self.model, temperature=0.0,
                messages=[{"role": role, "content": prompt}], **kw)
            with self._lock:
                self.usage["calls"] += 1
                self.usage["input_tokens"] += r.usage.prompt_tokens
                self.usage["output_tokens"] += r.usage.completion_tokens
            return r.choices[0].message.content or ""
        # anthropic: the prompt goes in `system` when the caller asked for a
        # system message (Mem0 sends its answer prompt that way).
        kw = {"system": prompt, "messages": [{"role": "user", "content": "Answer the question."}]} \
            if role == "system" else {"messages": [{"role": "user", "content": prompt}]}
        r = self._client.messages.create(model=self.model, max_tokens=2048,
                                         output_config={"effort": "low"}, **kw)
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += r.usage.input_tokens
            self.usage["output_tokens"] += r.usage.output_tokens
        return "".join(b.text for b in r.content if b.type == "text")


def _mock(prompt: str) -> str:
    """Offline stand-in for an LLM so the pipeline can be tested without a key."""
    if "label an answer" in prompt:
        gold = re.findall(r"Gold answer: (.*)", prompt)[-1]
        gen = prompt.rsplit("Generated answer:", 1)[1].split("First, provide", 1)[0]
        ok = _tokens(gold) and set(_tokens(gold)) <= set(_tokens(gen))
        return json.dumps({"label": "CORRECT" if ok else "WRONG"})
    q = re.findall(r"Question: (.*)", prompt)[-1]
    ctx = prompt.split("avoids vague time references", 1)[-1].rsplit("Question:", 1)[0]
    lines = [ln for ln in ctx.splitlines() if ln.strip()]
    qt = set(_tokens(q))
    return max(lines, key=lambda ln: len(qt & set(_tokens(ln))), default="")


_LABEL = re.compile(r"\b(CORRECT|WRONG)\b")


def parse_label(text: str) -> bool:
    try:
        return str(json.loads(text).get("label", "")).upper() == "CORRECT"
    except (json.JSONDecodeError, AttributeError):
        m = _LABEL.findall(text.upper())
        return bool(m) and m[-1] == "CORRECT"


def _tokens(s: str) -> list[str]:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_and_judge(records, answerer: LLM, judge: LLM, workers=8):
    done = [0]
    t0 = time.perf_counter()

    def one(r):
        prompt = ANSWER_PROMPT.format(context=r["context"], question=r["question"])
        ans = answerer(prompt, role="system")
        verdict = judge(JUDGE_PROMPT.format(question=r["question"], gold_answer=r["gold"],
                                            generated_answer=ans), json_mode=True)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  answered {done[0]}/{len(records)} ({time.perf_counter() - t0:.0f}s)",
                  file=sys.stderr)
        return r | {"answer": ans, "verdict": verdict, "correct": parse_label(verdict),
                    "f1": token_f1(ans, r["gold"])}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, records))


def summarise(rows) -> dict:
    out = {}
    for system in dict.fromkeys(r["system"] for r in rows):
        rs = [r for r in rows if r["system"] == system]
        cats = {c: [r for r in rs if r["category"] == c] for c in CATEGORIES.values()}
        out[system] = {
            "questions": len(rs),
            "J": round(100 * statistics.fmean(r["correct"] for r in rs), 2),
            "F1": round(100 * statistics.fmean(r["f1"] for r in rs), 2),
            "by_category_J": {c: round(100 * statistics.fmean(r["correct"] for r in v), 2)
                              for c, v in cats.items() if v},
            "gold_in_context": round(100 * statistics.fmean(gold_in_context(r) for r in rs), 1),
            "avg_context_tokens": round(statistics.fmean(r["context_tokens"] for r in rs)),
            "retrieval_ms_p50": round(statistics.median(r["retrieval_ms"] for r in rs), 2),
        }
    return out


def gold_in_context(r) -> bool:
    """LLM-free proxy: every content word of the gold answer appears in the context."""
    gold = set(_tokens(r["gold"]))
    return bool(gold) and gold <= set(_tokens(r["context"]))


def context_summary(records) -> dict:
    """Stage-1 numbers that need no LLM: context size and the gold-in-context proxy."""
    out = {}
    for system in dict.fromkeys(r["system"] for r in records):
        rs = [r for r in records if r["system"] == system]
        out[system] = {
            "questions": len(rs),
            "gold_in_context": round(100 * statistics.fmean(gold_in_context(r) for r in rs), 1),
            "by_category": {c: round(100 * statistics.fmean(gold_in_context(r) for r in v), 1)
                            for c in CATEGORIES.values()
                            if (v := [r for r in rs if r["category"] == c])},
            "avg_context_tokens": round(statistics.fmean(r["context_tokens"] for r in rs)),
            "retrieval_ms_p50": round(statistics.median(r["retrieval_ms"] for r in rs), 2),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["contexts", "all"], default="all")
    ap.add_argument("--systems", default="nomem,nomem-search,full,none",
                    help=f"comma list from {','.join(SYSTEMS)}")
    ap.add_argument("--budget", type=int, default=1000, help="context tokens (ApproxCounter)")
    ap.add_argument("--embedder", default="potion", choices=["potion", "none"])
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "mock"])
    ap.add_argument("--answer-model")
    ap.add_argument("--judge-model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="questions per conversation (quick runs)")
    ap.add_argument("--convs", type=int, help="first N conversations only")
    ap.add_argument("--rebuild", action="store_true", help="ignore saved contexts")
    ap.add_argument("--json", type=Path, help="summary output (default under results/)")
    args = ap.parse_args()

    systems = [s for s in args.systems.split(",") if s]
    bad = set(systems) - set(SYSTEMS)
    if bad:
        sys.exit(f"unknown systems: {', '.join(sorted(bad))}")
    if not LOCOMO.exists():
        sys.exit(f"LoCoMo data not found at {LOCOMO}; see benchmarks/README.md")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Contexts are deterministic for a given setup; save them per system so the
    # slow ones (Mem0 ingest is ~3 min per conversation) are built once.
    tag = f"b{args.budget}-{args.embedder}" + (f"-c{args.convs}" if args.convs else "") \
        + (f"-l{args.limit}" if args.limit else "")
    records = []
    for s in systems:
        path = RESULTS / f"qa_contexts_{s}_{tag}.jsonl"
        if path.exists() and not args.rebuild:
            records += [json.loads(x) for x in path.read_text().splitlines()]
            continue
        recs = build_contexts([s], args.budget, args.embedder, args.convs, args.limit)
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        records += recs
    print(f"{len(records)} contexts for {', '.join(systems)}", file=sys.stderr)
    if args.stage == "contexts":
        print(json.dumps(context_summary(records), indent=2))
        return

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY (or use --provider anthropic / mock)")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY (or use --provider openai / mock)")
    default = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5", "mock": "mock"}[args.provider]
    am, jm = args.answer_model or default, args.judge_model or default
    cache = RESULTS / "qa_cache.jsonl"
    answerer = LLM(args.provider, am, cache)
    judge = answerer if jm == am else LLM(args.provider, jm, cache)

    t0 = time.perf_counter()
    rows = answer_and_judge(records, answerer, judge, args.workers)
    summary = {
        "provider": args.provider, "answer_model": am, "judge_model": jm,
        "budget": args.budget, "embedder": args.embedder,
        "systems": summarise(rows),
        "llm_usage": {"answer": dict(answerer.usage)} | ({"judge": dict(judge.usage)} if judge is not answerer else {}),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stem = f"qa_{args.provider}_{am}_{tag}"
    (RESULTS / f"{stem}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = args.json or RESULTS / f"{stem}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
