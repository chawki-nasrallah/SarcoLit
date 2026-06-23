"""End-to-end RAG pipeline: question in, grounded cited answer out.

Step 5 of ``v0.2-rag-baseline`` — the single entry point that ties retrieval
(``Retriever``) and generation (``Generator``) together so a caller uses *one*
object instead of wiring the two halves by hand.

    pipeline = RagPipeline()
    result = pipeline.ask("What defines low grip strength in sarcopenia?")
    print(result.answer)        # grounded, PMID-cited text
    print(result.sources)       # the abstracts it was grounded in

CLI:  ``uv run python -m sarcolit.generation.rag "your question"``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from sarcolit.generation.generate import Generator
from sarcolit.retrieval.search import Retriever, SearchHit

DEFAULT_K = 5


@dataclass
class RagAnswer:
    """A grounded answer plus the abstracts it was built from."""

    question: str
    answer: str
    sources: list[SearchHit]


class RagPipeline:
    """Retriever + Generator behind one ``ask`` call.

    Both halves are injectable so the wiring can be tested with fakes (no GPU).
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        k: int = DEFAULT_K,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()
        self.k = k

    def ask(self, question: str, k: int | None = None) -> RagAnswer:
        """Retrieve the top-k abstracts, then generate a grounded answer."""
        hits = self.retriever.search(question, k=k or self.k)
        answer = self.generator.answer(question, hits)
        return RagAnswer(question=question, answer=answer, sources=hits)

    def close(self) -> None:
        self.retriever.close()


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: python -m sarcolit.generation.rag "your question"')
        return
    pipeline = RagPipeline()
    try:
        result = pipeline.ask(question)
        print(f"\n=== ANSWER ===\n{result.answer}\n")
        print("=== SOURCES ===")
        for hit in result.sources:
            print(f"  [PMID {hit.pmid}] ({hit.year}) {hit.title}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
