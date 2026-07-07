"""Run the generation (faithfulness) evaluation (Step 3 of v0.3-eval).

For each query: run the local RAG pipeline (``ask``) to get an answer + its
sources, then have Claude (Opus) judge whether the answer's claims are supported
by those sources. Aggregates a faithfulness rate + mean score, and saves a
per-answer report (so you can read *which* answers failed and why).

Only the judge uses the API; the RAG system under test stays local. The
pipeline and judge are injectable so the runner is unit-testable without a
GPU/API.

Run:  ``uv run --env-file .env python -m sarcolit.eval.generation --limit 15``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

QUERY_PATH = Path("evals/synthetic_queries.jsonl")
REPORT_PATH = Path("evals/reports/generation_baseline.json")


def load_queries(path: Path, limit: int | None = None, offset: int = 0) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    items = items[offset:]
    return items[:limit] if limit else items


def run_generation_eval(
    query_path: Path = QUERY_PATH,
    limit: int = 15,
    k: int = 5,
    out_path: Path | None = REPORT_PATH,
    offset: int = 0,
    pipeline=None,
    judge: Callable[[str, str, list[dict]], dict] | None = None,
) -> tuple[dict, list[dict]]:
    """Run RAG over `limit` queries (from `offset`), judge faithfulness, save."""
    queries = load_queries(query_path, limit, offset=offset)

    created_pipeline = pipeline is None
    if created_pipeline:
        from sarcolit.generation.rag import RagPipeline  # lazy: avoids torch import

        pipeline = RagPipeline(k=k)

    if judge is None:
        from anthropic import Anthropic

        from sarcolit.eval.judge import judge_faithfulness

        client = Anthropic()

        def judge(q: str, a: str, s: list[dict]) -> dict:
            return judge_faithfulness(client, q, a, s)

    results = []
    try:
        for i, item in enumerate(queries, 1):
            ans = pipeline.ask(item["query"])
            sources = [
                {"pmid": h.pmid, "title": h.title, "abstract": h.abstract} for h in ans.sources
            ]
            verdict = judge(item["query"], ans.answer, sources)
            results.append(
                {
                    "query": item["query"],
                    "answer": ans.answer,
                    "faithful": verdict["faithful"],
                    "score": verdict["score"],
                    "unsupported_claims": verdict.get("unsupported_claims", []),
                    "reasoning": verdict.get("reasoning", ""),
                }
            )
            print(
                f"[{i}/{len(queries)}] faithful={verdict['faithful']} score={verdict['score']:.2f}"
            )
    finally:
        if created_pipeline:
            pipeline.close()

    n = len(results)
    summary = {
        "n": n,
        "faithful_rate": sum(1 for r in results if r["faithful"]) / n if n else 0.0,
        "mean_faithfulness_score": sum(r["score"] for r in results) / n if n else 0.0,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run generation faithfulness eval.")
    ap.add_argument("--queries", type=Path, default=QUERY_PATH)
    ap.add_argument("--limit", type=int, default=15, help="number of queries to judge")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N queries")
    ap.add_argument("--k", type=int, default=5, help="top-k passed to the RAG pipeline")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args()

    summary, _ = run_generation_eval(
        args.queries, limit=args.limit, k=args.k, out_path=args.out, offset=args.offset
    )
    print(
        f"\nfaithful: {summary['faithful_rate']:.2f} "
        f"| mean score: {summary['mean_faithfulness_score']:.2f} "
        f"| n={summary['n']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
