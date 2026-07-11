# SarcoLit

A retrieval-augmented LLM assistant for the sarcopenia and elderly muscle-health literature. Built end-to-end (data → retrieval → fine-tuning → agent → deployment → monitoring) as a public engineering log.

> Built while finishing my PhD on HD-sEMG and AI for elderly muscle health (UTC / BMBI, defending Nov 2026). The aim is to combine domain knowledge with current AI-engineering practice rather than assemble another generic RAG demo.

---

## Status

**Current tag:** `v0.4-rag-improved` — an eval-driven milestone: three RAG improvements (reranker, hybrid search, prompt tuning) were built and measured against the `v0.3` baseline; **none beat it, so none shipped.** The finding is the value — retrieval is already strong, and the real bottleneck is the 3B model's attribution capability, which prompting can't fix → sets up `v0.5` fine-tuning. Also delivers the hand-labelled **gold eval set**. See [Eval results](#eval-results). Still fully **local and open-source** (only the eval judge uses an API model).

## Milestones

Each milestone is a tagged GitHub release with a dedicated release note in [`docs/releases/`](docs/releases/) and an eval delta vs. the previous tag (where applicable). No milestone is skipped or merged into another, even if rough — the chronology is the point.

| Tag | Theme | Status |
|---|---|---|
| `v0.0-scaffold` | Repo skeleton, CI, license | ✅ done |
| `v0.1-corpus` | PubMed ingestion, cleaned sarcopenia/EMG corpus (7,004 abstracts) | ✅ done |
| `v0.2-rag-baseline` | Naive embedding + retrieval + generation, first end-to-end answer | ✅ done |
| `v0.3-eval` | Eval harness + measured baseline (recall@k/MRR/nDCG, faithfulness) | ✅ done |
| `v0.4-rag-improved` | Measured reranker/hybrid/prompt-tuning vs baseline (none adopted); gold eval set; bottleneck located | ✅ done |
| `v0.5-finetune` | LoRA fine-tune of a small open model; eval comparison; model card | ⬜ |
| `v0.6-agent` | Tool-using agent (PubMed search, citation tool) | ⬜ |
| `v0.7-serve` | FastAPI + Docker + Gradio demo on HF Spaces | ⬜ |
| `v0.8-monitor` | Logging, latency/cost dashboard, drift checks | ⬜ |
| `v1.0-biosignal-tool` | (optional) EMG classifier the agent can call | ⬜ |

## Eval results

Headline retrieval + generation metrics per tag, against the eval sets introduced in `v0.3`. Each later tag reports a delta against this baseline.

| Tag | Recall@5 | MRR | nDCG@5 | Faithfulness | Notes |
|---|---|---|---|---|---|
| `v0.3-eval` (baseline) | 0.91 | 0.78 | 0.81 | 0.80 | synthetic known-item set: retrieval n=150, faithfulness n=30 (Opus judge). |
| `v0.4-rag-improved` | 0.91 | 0.78 | 0.81 | 0.80 | **config unchanged** — reranker, hybrid, and 3 prompt variants all measured, none beat baseline. Added **gold set** (realistic, n=14): MRR **1.00**, nDCG@5 **0.92** (bge). Bottleneck = model attribution → `v0.5`. |

Failure mode at baseline: **citation misattribution** (answer's core finding correct, but a specific claim pinned to the wrong source). `v0.4` proved prompting can't fix it (three variants, all worse) → the fix is fine-tuning (`v0.5`). Full write-ups: [`docs/releases/v0.3-eval.md`](docs/releases/v0.3-eval.md), [`docs/releases/v0.4-rag-improved.md`](docs/releases/v0.4-rag-improved.md).

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
- **Retrieval:** `BAAI/bge-base-en-v1.5` embeddings via `transformers`, [`qdrant`](https://qdrant.tech/) local vector store (BM25 hybrid + `bge-reranker-base` cross-encoder built and measured at `v0.4`, but bge-only ships — see release note)
- **Generation:** `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 (`bitsandbytes`) on GPU
- **Eval:** in-repo metrics (recall@k/MRR/nDCG, faithfulness); Claude (`anthropic`) as LLM-judge — *eval tooling only; the RAG generator stays local*
- _later:_ `peft`/`trl` for LoRA (`v0.5`), `fastapi`, `gradio`, HF Spaces for the demo (`v0.7`)

## Try it

The `v0.2-rag-baseline` pipeline runs end-to-end on a single machine, fully local. No data ships in the repo — you rebuild the corpus and index from the pipeline (the search query lives in code, so the result is reproducible).

**Requirements**
- An NVIDIA GPU with ~6 GB VRAM (the model is 4-bit quantized to fit; CPU works but is slow).
- An NCBI email — and optionally a [free API key](https://www.ncbi.nlm.nih.gov/account/) for faster fetching — to download abstracts from PubMed.
- First run downloads ~6.5 GB of models (bge ~0.4 GB, Qwen2.5-3B ~6 GB), cached afterwards.

**Setup**

```bash
# 1. Install the full ML stack (CUDA torch, transformers, bitsandbytes, qdrant)
uv sync --extra ml

# 2. Add your NCBI credentials
cp .env.example .env          # then edit: set NCBI_EMAIL (and NCBI_API_KEY if you have one)

# 3. Fetch the corpus from PubMed  (~7,000 abstracts → data/raw/pubmed/)
uv run --env-file .env python -m sarcolit.ingest.pubmed

# 4. Parse + clean + dedup into data/sarcolit-corpus-v0.1/corpus.jsonl
uv run python -m sarcolit.ingest.parse

# 5. Embed the corpus + build the vector index  (data/qdrant/)
uv run python -m sarcolit.retrieval.index
```

**Ask questions** — interactive mode loads the models once, then loops:

```bash
uv run python -m sarcolit.generation.rag
```
```
SarcoLit RAG - type a question, or blank line / Ctrl-C to quit.

> Does vitamin D supplementation improve muscle strength in older adults?
   ... grounded answer + the PMIDs it used ...
```

Other entry points:
```bash
# One-shot question (no REPL)
uv run python -m sarcolit.generation.rag "What handgrip cutoffs define sarcopenia?"

# Retrieval only — see the matching abstracts without generating an answer
uv run python -m sarcolit.retrieval.search "surface EMG in muscle aging"
```

> This is a `v0.2` **baseline**: it works and cites sources, but is not yet evaluated or tuned — answers can still mis-attribute (see `docs/releases/v0.2-rag-baseline.md`). Quantitative quality lands with the `v0.3` eval harness.

## Development

```bash
uv sync --extra dev           # light deps only (no GPU stack)
uv run pytest -m "not eval"   # cheap test subset (CI runs this)
uv run ruff check . && uv run ruff format --check .
```

## Engineering log

A dated record of work sessions and what was learned at each step lives in [`docs/weekly_log.md`](docs/weekly_log.md). Release-level summaries live in [`docs/releases/`](docs/releases/).

## License

[MIT](LICENSE).