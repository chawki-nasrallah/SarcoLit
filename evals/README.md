# Evals

The eval harness lands in `v0.3` and is the highest-leverage milestone of the whole project. Most portfolio RAG demos skip rigorous evaluation; this one will not.

## Planned contents (by `v0.3`)

- `qa_set_v1.jsonl` — hand-curated sarcopenia QA pairs with gold answers and gold citation IDs. Curated against the actual PubMed corpus from `v0.1`, not synthetic.
- `retrieval_metrics.py` — recall@k, MRR, nDCG against the gold citation IDs.
- `generation_metrics.py` — faithfulness (grounded in retrieved chunks?), answer relevance, citation precision. LLM-as-judge with calibration runs.
- `report_template.md` — eval reports per release tag get committed to `docs/releases/<tag>.md`.

## Workflow

- Cheap subset (deterministic, no API/GPU) runs in CI on every PR via `pytest -m "not eval"`.
- Full eval runs locally / on Colab; results land in the release note for each tag.