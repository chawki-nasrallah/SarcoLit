"""Tests for prompt assembly and pipeline wiring (pure, no model/GPU)."""

from dataclasses import dataclass

import pytest

from sarcolit.generation import prompt


@dataclass
class FakeHit:
    pmid: str
    title: str
    abstract: str
    year: int | None


HITS = [
    FakeHit("111", "EMG in sarcopenia", "EMG detects motor unit loss.", 2023),
    FakeHit("222", "Grip strength cutoffs", "Cutoff is 27 kg for men.", None),
]


def test_format_context_numbers_and_labels_sources() -> None:
    ctx = prompt.format_context(HITS)
    assert "[1] (PMID: 111, 2023) EMG in sarcopenia" in ctx
    assert "EMG detects motor unit loss." in ctx
    # No year → no trailing comma-year.
    assert "[2] (PMID: 222) Grip strength cutoffs" in ctx


def test_build_messages_has_system_and_user_with_question() -> None:
    msgs = prompt.build_messages("What defines low grip strength?", HITS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "ONLY" in msgs[0]["content"]  # grounding instruction present
    assert "PMID" in msgs[0]["content"]  # citation instruction present
    assert "What defines low grip strength?" in msgs[1]["content"]
    assert "Cutoff is 27 kg for men." in msgs[1]["content"]  # context embedded


class FakeRetriever:
    def __init__(self, hits: list[FakeHit]) -> None:
        self.hits = hits
        self.requested_k: int | None = None

    def search(self, question: str, k: int = 5) -> list[FakeHit]:
        self.requested_k = k
        return self.hits

    def close(self) -> None:
        pass


class FakeGenerator:
    def __init__(self) -> None:
        self.seen_passages: list[FakeHit] | None = None

    def answer(self, question: str, passages: list[FakeHit]) -> str:
        self.seen_passages = passages
        return f"answer to: {question}"


def test_ask_pipeline_wires_retrieval_into_generation() -> None:
    # Importing rag pulls the ml stack; skip in CI's light env, run locally.
    pytest.importorskip("torch")
    pytest.importorskip("qdrant_client")
    from sarcolit.generation.rag import RagPipeline

    retriever = FakeRetriever(HITS)
    generator = FakeGenerator()
    pipeline = RagPipeline(retriever=retriever, generator=generator, k=3)

    result = pipeline.ask("What defines low grip strength?")

    assert result.answer == "answer to: What defines low grip strength?"
    assert result.sources == HITS  # retrieved abstracts returned alongside answer
    assert retriever.requested_k == 3  # pipeline k honoured
    assert generator.seen_passages == HITS  # generator grounded on retrieved hits
