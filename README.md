# SarcoLit

A retrieval-augmented LLM assistant for the sarcopenia and elderly muscle-health literature. Built end-to-end (data → retrieval → fine-tuning → agent → deployment → monitoring) as a public engineering log.

> Built while finishing my PhD on HD-sEMG and AI for elderly muscle health (UTC / BMBI, defending Nov 2026). The aim is to combine domain knowledge with current AI-engineering practice rather than assemble another generic RAG demo.

---

## Status

**Current tag:** `v0.0-scaffold` — repo skeleton only. No working pipeline yet.

## Milestones

Each milestone is a tagged GitHub release with a dedicated release note in [`docs/releases/`](docs/releases/) and an eval delta vs. the previous tag (where applicable). No milestone is skipped or merged into another, even if rough — the chronology is the point.

| Tag | Theme | Status |
|---|---|---|
| `v0.0-scaffold` | Repo skeleton, CI, license | ⬜ in progress |
| `v0.1-corpus` | PubMed ingestion, cleaned sarcopenia/EMG corpus | ⬜ |
| `v0.2-rag-baseline` | Naive embedding + retrieval + generation, first end-to-end answer | ⬜ |
| `v0.3-eval-harness` | Hand-curated eval set + automated retrieval/generation metrics | ⬜ |
| `v0.4-rag-improved` | Hybrid search, reranker, query rewriting; measured lift over `v0.2` | ⬜ |
| `v0.5-finetune` | LoRA fine-tune of a small open model; eval comparison; model card | ⬜ |
| `v0.6-agent` | Tool-using agent (PubMed search, citation tool) | ⬜ |
| `v0.7-serve` | FastAPI + Docker + Gradio demo on HF Spaces | ⬜ |
| `v0.8-monitor` | Logging, latency/cost dashboard, drift checks | ⬜ |
| `v1.0-biosignal-tool` | (optional) EMG classifier the agent can call | ⬜ |

## Eval results

Filled in as milestones land. Each row reports headline retrieval and generation metrics against the curated eval set introduced in `v0.3`.

| Tag | Recall@5 | MRR | Faithfulness | Answer relevance | Notes |
|---|---|---|---|---|---|
| _pending v0.3_ | — | — | — | — | — |

## Project layout

```
sarcolit/
├── src/sarcolit/         library code (ingest / retrieval / generation / eval / agent / serve)
├── evals/                versioned eval sets + per-release eval reports
├── docs/                 architecture notes, model/dataset cards, weekly log, release notes
├── notebooks/            exploration only, never the source of truth
├── data/                 gitignored; HF datasets / DVC pointers live here
├── .github/workflows/    CI (lint, tests, eval-on-PR)
└── pyproject.toml        uv-managed deps
```

## Tech stack (grows with the project)

- Python 3.12, [uv](https://github.com/astral-sh/uv) for environment + dependency management
- `ruff` (lint + format), `pytest` (tests + eval markers)
- _later:_ `sentence-transformers`, `qdrant`/`chroma`, `vllm`/`ollama`, `peft`/`trl` for LoRA, `fastapi`, `gradio`, GitHub Actions, HF Spaces for the demo

## Running locally

```bash
uv sync                # install pinned deps
uv run pytest -m "not eval"   # run the cheap test subset
```

More entry points will be documented as milestones add them.

## Engineering log

A dated record of work sessions and what was learned at each step lives in [`docs/weekly_log.md`](docs/weekly_log.md). Release-level summaries live in [`docs/releases/`](docs/releases/).

## License

[MIT](LICENSE).