"""The LLM-judged QA harness: label parsing, LoCoMo F1, and the offline pipeline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import qa_bench  # noqa: E402


def test_parse_label_json_and_fallback():
    assert qa_bench.parse_label('{"label": "CORRECT"}')
    assert not qa_bench.parse_label('{"label": "WRONG"}')
    assert qa_bench.parse_label("Same date, so the answer is CORRECT")
    assert not qa_bench.parse_label("Different city. WRONG")
    assert not qa_bench.parse_label("")


def test_token_f1():
    assert qa_bench.token_f1("7 May 2023", "7 May 2023") == 1.0
    assert qa_bench.token_f1("The shell necklace", "a shell necklace") == 1.0
    assert qa_bench.token_f1("Paris", "London") == 0.0
    assert 0 < qa_bench.token_f1("she went to Paris in May", "Paris") < 1


def test_prompts_format():
    p = qa_bench.ANSWER_PROMPT.format(context="[1 May 2023, 1:00 PM] A: hi", question="Q?")
    assert "[1 May 2023, 1:00 PM] A: hi" in p and "Question: Q?" in p
    j = qa_bench.JUDGE_PROMPT.format(question="Q?", gold_answer="G", generated_answer="X")
    assert "Gold answer: G" in j and "Generated answer: X" in j


def test_pack_respects_budget():
    counter = qa_bench.ApproxCounter()
    items = [(1_700_000_000.0, "A: " + "word " * 50)] * 30
    text, used = qa_bench._pack(items, 300, counter)
    assert used <= 300 and text.count("\n") + 1 == len([x for x in text.splitlines() if x])


def test_mock_pipeline_end_to_end():
    recs = [
        {"system": "s", "category": "single-hop", "question": "Where did Ann move?",
         "gold": "Paris", "context": "[1 May 2023, 1:00 PM] Ann: I moved to Paris last week",
         "context_tokens": 12, "retrieval_ms": 1.0},
        {"system": "s", "category": "temporal", "question": "When did Bo leave?",
         "gold": "June 2022", "context": "(no memories)", "context_tokens": 0, "retrieval_ms": 0.0},
    ]
    llm = qa_bench.LLM("mock", "mock", cache=None)
    rows = qa_bench.answer_and_judge(recs, llm, llm, workers=2)
    assert [r["correct"] for r in rows] == [True, False]
    summary = qa_bench.summarise(rows)["s"]
    assert summary["J"] == 50.0 and summary["questions"] == 2
    json.dumps(summary)
