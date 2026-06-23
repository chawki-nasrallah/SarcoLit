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
from typing import TYPE_CHECKING

# Heavy imports (torch/transformers/qdrant) are deferred into __init__ so that
# importing this module — and printing a startup message — is instant. The slow
# `import transformers` otherwise runs silently before any output, making the
# CLI look frozen for ~30s on first launch.
if TYPE_CHECKING:
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
        if retriever is None or generator is None:
            # Imported here, not at module top, to keep startup responsive.
            from sarcolit.generation.generate import Generator
            from sarcolit.retrieval.search import Retriever
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


def _print_result(result: RagAnswer) -> None:
    print(f"\n=== ANSWER ===\n{result.answer}\n")
    print("=== SOURCES ===")
    for hit in result.sources:
        print(f"  [PMID {hit.pmid}] ({hit.year}) {hit.title}")


def _interactive(pipeline: RagPipeline) -> None:
    """Ask questions in a loop; models stay loaded between questions."""
    print("\nSarcoLit RAG - type a question, or blank line / Ctrl-C to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        _print_result(pipeline.ask(question))


def main() -> None:
    """One-shot if a question is given on the CLI, else interactive REPL."""
    question = " ".join(sys.argv[1:]).strip()
    # Flush so this shows *before* the slow torch/transformers import below,
    # otherwise the terminal looks frozen for ~30s on first launch.
    print(
        "Starting SarcoLit - loading libraries and models (first launch can take ~30-60s)...",
        flush=True,
    )
    pipeline = RagPipeline()
    print("Models loaded.", flush=True)
    try:
        if question:
            _print_result(pipeline.ask(question))
        else:
            _interactive(pipeline)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
