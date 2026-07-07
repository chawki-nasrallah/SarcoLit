# Evals

The eval harness (`v0.3`) — the highest-leverage milestone of the project. Most
portfolio RAG demos skip rigorous evaluation; this one doesn't. It turns the
qualitative v0.2 observations into numbers and establishes the baseline that
v0.4/v0.5 are measured against.

## Layout

- **Eval *data* lives here** (`evals/`, committed — small: queries + relevant
  PMIDs, no abstract text):
  - `synthetic_queries.jsonl` — synthetic **known-item** retrieval set. Claude
    (Opus) wrote one realistic question per sampled abstract; that abstract's
    PMID is the known-relevant target. `{query, relevant_pmid, subtopic}`.
    Frozen once generated (generation isn't deterministic).
  - `gold_queries.jsonl` — hand-curated **expert** set: realistic queries with
    domain-judged relevant PMIDs. `{query, relevant_pmids, subtopic, notes}`.
- **Eval *code* lives in `src/sarcolit/eval/`**:
  - `synthetic.py` — generates the synthetic set (Claude Opus).
  - `metrics.py` — retrieval metrics: recall@k, MRR, nDCG (no API/GPU).
  - `judge.py` — generation faithfulness / answer relevance (Claude-as-judge).
  - `run.py` — runs the full eval, writes a report.

## Ground truth (hybrid)

- **Synthetic** gives scale automatically, but questions are phrased from the
  source abstract (somewhat easier).
- **Gold** is realistic and expert-judged, but small.
Report metrics on each; the gold set is the more trustworthy signal.

## Local vs. offline

Only the eval *tooling* (question generation, faithfulness judging) uses the
Claude API. The RAG system under test stays fully local (bge + qdrant + Qwen).

## Workflow

- Cheap subset (deterministic, no API/GPU — e.g. the metric functions) runs in
  CI via `pytest -m "not eval"`.
- Full eval runs locally; results land in the release note for each tag and the
  README eval table.
