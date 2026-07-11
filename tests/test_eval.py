"""Tests for retrieval metrics + runner (pure, no API/GPU)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from sarcolit.eval import generation, judge, label_gold, metrics, retrieval


def test_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d"]
    assert metrics.recall_at_k(ranked, {"b"}, 1) == 0.0  # b not in top-1
    assert metrics.recall_at_k(ranked, {"b"}, 2) == 1.0  # b in top-2
    # Two relevant, one in top-2 → recall 0.5.
    assert metrics.recall_at_k(ranked, {"b", "z"}, 2) == 0.5


def test_reciprocal_rank() -> None:
    ranked = ["a", "b", "c"]
    assert metrics.reciprocal_rank(ranked, {"a"}) == 1.0  # rank 1
    assert metrics.reciprocal_rank(ranked, {"b"}) == 0.5  # rank 2
    assert metrics.reciprocal_rank(ranked, {"z"}) == 0.0  # not found


def test_ndcg_at_k() -> None:
    ranked = ["a", "b", "c"]
    # Relevant at rank 1 → perfect ranking → 1.0.
    assert metrics.ndcg_at_k(ranked, {"a"}, 3) == 1.0
    # Relevant at rank 2 → dcg = 1/log2(3), idcg = 1 → ndcg = 1/log2(3).
    assert math.isclose(metrics.ndcg_at_k(ranked, {"b"}, 3), 1 / math.log2(3))
    assert metrics.ndcg_at_k(ranked, {"z"}, 3) == 0.0  # not found


def test_evaluate_aggregates() -> None:
    # q1: relevant at rank 1; q2: relevant at rank 3.
    results = [
        (["x", "y", "z"], ["x"]),
        (["a", "b", "c"], ["c"]),
    ]
    m = metrics.evaluate(results, ks=(1, 3))
    assert m["recall@1"] == 0.5  # only q1 found at k=1
    assert m["recall@3"] == 1.0  # both found by k=3
    assert m["mrr"] == (1.0 + 1 / 3) / 2  # ranks 1 and 3
    assert 0.0 < m["ndcg@3"] <= 1.0


def test_empty_relevant_is_zero() -> None:
    assert metrics.recall_at_k(["a"], set(), 1) == 0.0
    assert metrics.ndcg_at_k(["a"], set(), 1) == 0.0


# --- runner (with a fake retriever, no GPU) --------------------------------


@dataclass
class _Hit:
    pmid: str


class _FakeRetriever:
    """Returns a fixed ranking per query keyed by the query string."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def search(self, query: str, k: int) -> list[_Hit]:
        return [_Hit(p) for p in self.rankings[query][:k]]

    def close(self) -> None:
        pass


def test_load_query_set_accepts_synthetic_and_gold(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rows = [
        {"query": "q1", "relevant_pmid": "111"},  # synthetic shape
        {"query": "q2", "relevant_pmids": ["222", "333"]},  # gold shape
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = retrieval.load_query_set(path)
    assert items[0]["relevant"] == ["111"]
    assert items[1]["relevant"] == ["222", "333"]


def test_run_retrieval_eval_with_fake_retriever(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rows = [
        {"query": "q1", "relevant_pmid": "a"},  # a ranked 1st
        {"query": "q2", "relevant_pmid": "z"},  # z ranked 3rd
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    fake = _FakeRetriever({"q1": ["a", "b", "c"], "q2": ["x", "y", "z"]})

    m, n = retrieval.run_retrieval_eval(path, ks=(1, 3), retriever=fake)
    assert n == 2
    assert m["recall@1"] == 0.5  # only q1's relevant is at rank 1
    assert m["recall@3"] == 1.0
    assert m["mrr"] == (1.0 + 1 / 3) / 2


# --- judge helpers (pure, no API) ------------------------------------------


def test_judge_format_sources_numbers_and_labels() -> None:
    sources = [
        {"pmid": "111", "title": "EMG in aging", "abstract": "Alpha."},
        {"pmid": "222", "title": "Grip cutoffs", "abstract": "Beta."},
    ]
    text = judge.format_sources(sources)
    assert "[1] (PMID: 111) EMG in aging\nAlpha." in text
    assert "[2] (PMID: 222) Grip cutoffs\nBeta." in text


def test_judge_parse_json_extracts_object() -> None:
    raw = 'Here is my verdict:\n{"faithful": true, "score": 0.8, "reasoning": "ok"}'
    parsed = judge._parse_json(raw)
    assert parsed["faithful"] is True
    assert parsed["score"] == 0.8


def test_judge_parse_json_ignores_trailing_text() -> None:
    # The failure that killed a run: valid JSON followed by extra prose.
    raw = '{"faithful": false, "score": 0.3}\n\nNote: some trailing explanation.'
    parsed = judge._parse_json(raw)
    assert parsed["faithful"] is False
    assert parsed["score"] == 0.3


# --- generation eval runner (fake pipeline + fake judge, no GPU/API) --------


@dataclass
class _AnsHit:
    pmid: str
    title: str
    abstract: str


@dataclass
class _Answer:
    answer: str
    sources: list[_AnsHit]


class _FakePipeline:
    def ask(self, query: str) -> _Answer:
        return _Answer(f"answer to {query}", [_AnsHit("111", "T", "A")])

    def close(self) -> None:
        pass


def test_run_generation_eval_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rows = [{"query": "q1", "relevant_pmid": "a"}, {"query": "q2", "relevant_pmid": "b"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # Fake judge: q1 faithful, q2 not.
    def fake_judge(q: str, a: str, s: list[dict]) -> dict:
        faithful = q == "q1"
        return {"faithful": faithful, "score": 1.0 if faithful else 0.2, "reasoning": ""}

    summary, results = generation.run_generation_eval(
        path, limit=2, out_path=tmp_path / "rep.json", pipeline=_FakePipeline(), judge=fake_judge
    )
    assert summary["n"] == 2
    assert summary["faithful_rate"] == 0.5
    assert summary["mean_faithfulness_score"] == (1.0 + 0.2) / 2
    assert results[0]["answer"] == "answer to q1"


# --- gold labelling parser (pure) ------------------------------------------


def test_parse_selection() -> None:
    assert label_gold._parse_selection("1 4 9", 20) == [0, 3, 8]
    assert label_gold._parse_selection("9,4,1", 20) == [0, 3, 8]  # commas + order
    assert label_gold._parse_selection("n", 20) == []  # none relevant
    assert label_gold._parse_selection("", 20) == []
    assert label_gold._parse_selection("1 1 2", 20) == [0, 1]  # dedup
    assert label_gold._parse_selection("21", 20) is None  # out of range
    assert label_gold._parse_selection("abc", 20) is None  # invalid
