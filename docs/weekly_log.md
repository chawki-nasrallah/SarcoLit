# Engineering log

A dated record of work sessions. The format is deliberately simple so it doesn't become a chore: when, how long, what got done, what was learned, what's next. Recruiters read this top-to-bottom to see how the project actually developed.

---

## 2026-W18 — week of 2026-04-27

### 2026-05-02 (~1 h)

- **Done:** Approved the project plan (RAG + agent over the sarcopenia literature, ~7–8 wk to `v0.8`). Scaffolded the repo: `pyproject.toml` (uv), CI (ruff + pytest), `src/sarcolit/` package layout, MIT license, `.gitignore`, `.env.example`, docs skeleton, milestone-tagged release notes folder. Tagged `v0.0-scaffold` locally.
- **Learned:** Settled on the eight-stage A-to-Z slice (ingest → retrieval → generation → eval → fine-tune → agent → serve → monitor) before writing any code. Forces every milestone to map to one stage, and prevents the scaffolding from drifting into a generic Python template.
- **Next:** `v0.1-corpus` — PubMed ingestion. Read Chip Huyen Ch. 5 alongside the work, not before.