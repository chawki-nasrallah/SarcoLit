# Engineering log

A dated record of work sessions. The format is deliberately simple so it doesn't become a chore: when, how long, what got done, what was learned, what's next. Recruiters read this top-to-bottom to see how the project actually developed.

---

## 2026-W18 — week of 2026-04-27

### 2026-05-02 (~1 h)

- **Done:** Approved the project plan (RAG + agent over the sarcopenia literature, ~7–8 wk to `v0.8`). Scaffolded the repo: `pyproject.toml` (uv), CI (ruff + pytest), `src/sarcolit/` package layout, MIT license, `.gitignore`, `.env.example`, docs skeleton, milestone-tagged release notes folder. Tagged `v0.0-scaffold` locally.
- **Learned:** Settled on the eight-stage A-to-Z slice (ingest → retrieval → generation → eval → fine-tune → agent → serve → monitor) before writing any code. Forces every milestone to map to one stage, and prevents the scaffolding from drifting into a generic Python template.
- **Next:** `v0.1-corpus` — PubMed ingestion. Read Chip Huyen Ch. 5 alongside the work, not before.

### 2026-05-02 (continued, ~3 h)

- **Done:** Started `v0.1-corpus`. Steps 1 and 2 of 8 closed.
  - **Step 1 — Search strategy locked.** Iterated the PubMed query in the web UI before any code: starting from a broad MeSH query (~20k results), tightened with `[Majr]` to require sarcopenia/atrophy/EMG be *central* topics; then with subheadings (`/diagnosis`, `/classification`, `/physiopathology`, `/pathology`) to bias toward assessment-and-mechanism papers over comorbidity/treatment papers; widened the EMG branch from `[Majr]` to `[MeSH]` once a diagnostic showed only 3% EMG share; and added `NOT ("Mice"[MeSH] OR "Rats"[MeSH] OR "Disease Models, Animal"[MeSH])` to enforce true human-only after discovering `humans[MeSH]` is a soft filter. Final: 6,800 abstracts, 19% EMG share. Rationale captured in `docs/releases/v0.1-corpus.md`.
  - **Step 2 — Deps + scope decision.** Pinned `httpx`, `lxml`, `datasets`, `tenacity`, `tqdm` for v0.1 ingestion; committed `uv.lock`. Logged the abstracts-only scope decision in the architecture decisions log + release note: full-text (PMC OA Subset) deferred to `v0.4` so the eval harness at `v0.3` can measure the lift rather than ship "more text" unmeasured.
  - **Public push.** Repo now live at https://github.com/chawki-nasrallah/SarcoLit; `v0.0-scaffold` tag pushed; CI active on GitHub Actions. First CI run failed on `ruff format --check` (missing trailing newlines in scaffold files); fixed and CI green.
- **Learned:**
  - **MeSH precision toolkit.** Three orthogonal precision levers: `[Majr]` (centrality), subheadings (aspect-of-topic), and explicit `NOT` clauses for filters that look like inclusive constraints but aren't (the `humans[MeSH]` gotcha). Each one is the right tool for a different *kind* of noise.
  - **Iterative query design as a measurement loop, not a guess.** Every refinement step paired a count check (is the corpus the right size?) with a first-page eyeball (are the top results actually relevant?). Counting alone hides bias; eyeballing alone hides scale problems. Applied 5 times before locking.
  - **Stage scope decisions for measurability.** Abstract-only at v0.1 isn't a compromise — it's what makes the later v0.4 full-text upgrade *measurable* against an eval harness baseline. "More data is better" is unverifiable; "+X points on faithfulness, +Y on recall@5 from full text" is the artefact.
  - **POSIX trailing newlines exist and CI cares.** Tiny gotcha; ruff format catches it; never going to forget after watching CI go red on it.
- **Next:** Step 3 of `v0.1-corpus` — write the PubMed E-utilities fetcher in `src/sarcolit/ingest/`. First substantive code; co-author mode TBD (A: Claude scaffolds + I fill in the API logic; B: Claude writes + I explain back; C: I write from scratch + Claude reviews).

---

## 2026-W26 — week of 2026-06-22

### 2026-06-23 (~3 h)

- **Done:** Closed Steps 3 and 4 of `v0.1-corpus` — the corpus now exists on disk. Co-author mode B (Claude writes, I explain each piece back before we move on).
  - **Step 3 — PubMed fetcher** (`src/sarcolit/ingest/pubmed.py`). Two-step E-utilities flow: `esearch` with `usehistory=y` parks the result set on NCBI's history server (returns `WebEnv` + `query_key` + count); `efetch` then pages through it in batches of 200, writing raw XML straight to `data/raw/pubmed/`. Resumable (existing batches skipped), throttled (10 req/s with API key, 3 without), retries only transient errors (429/5xx/transport, never a 400 bad query) via `tenacity`. Writes a `manifest.json` recording the exact query + hash + timestamp for reproducibility. Ran it: **7,004 records, 36 batches, 129 MB raw XML.**
  - **Step 4 — Parse/clean/dedup** (`src/sarcolit/ingest/parse.py`). Walks the 36 XML files with stdlib `ElementTree`, extracts the 8 retained fields, flattens structured abstracts to plain text, dedups by PMID, drops abstract-less records, sorts by PMID for determinism, writes `corpus.jsonl` + a regenerated `dataset_card.md`. **Final corpus: 7,004 records, 16 MB, 0 dropped.** 11 tests passing incl. a determinism test (re-run → byte-identical output).
  - **Docs:** Filled in the release-note Delivered checklist + Corpus stats; recorded the chunking-deferral and the two dependency divergences (below).
- **Learned:**
  - **Small named pieces + one assembler.** Both modules follow the same shape: tiny single-purpose helpers (`_throttle`, `_is_transient`, one extractor per field) snapped together by one readable top-level function (`fetch`, `build_corpus`). Made the "explain it back" step actually tractable — each piece is small enough to reason about alone.
  - **Pinned ≠ used.** Step 2 pinned `lxml` and `datasets` on spec. In practice stdlib `ElementTree` parsed the XML fine and JSONL is `datasets`-compatible without the library — so both went unused. Lesson: pin deps when you reach for them, not in anticipation; flagged both for possible removal.
  - **Provenance is cheap insurance.** The `manifest.json` (query hash, count, timestamp, whether an API key was used) costs ~10 lines but makes the whole fetch reproducible and auditable. Same idea as the determinism test on the parse side.
  - **Windows console encoding bites.** A `→` in a `print()` crashed the run under cp1252 *after* the corpus had already been written. Swapped for ASCII `->`. Reminder that stdout encoding ≠ file encoding (files are explicitly UTF-8).
- **Next:** `v0.2-rag-baseline` — first end-to-end answer over this corpus: embed the 7,004 abstracts → vector store → retrieval → generation. Chunking decision lives here (read Chip Huyen Ch. 5 alongside). Before that: commit + tag `v0.1-corpus`.

### 2026-06-23 (continued, ~4 h) — `v0.2-rag-baseline` closed

- **Done:** Built the full local, open-source RAG pipeline end-to-end. Corrected an early slip where I'd assumed Claude as the generator — the project is deliberately a self-hosted open-source stack (`architecture.md`: "LoRA-tuned 3B").
  - **GPU stack stood up + proven first.** Added an `ml` optional-dependency extra (torch cu121, transformers, bitsandbytes, qdrant-client); kept core/CI lightweight. Checked hardware honestly — RTX 3060 Laptop, **6 GB VRAM** (not 8). Verified CUDA torch sees the GPU, then loaded **Qwen2.5-3B-Instruct in 4-bit NF4** (peak **2.12 GB** VRAM) and generated before building anything on top.
  - **Step 2 — index** (`retrieval/embedding.py`, `retrieval/index.py`). bge-base-en-v1.5 (CLS pooling + L2 norm) → 7,004 vectors → qdrant local mode. **`sentence-transformers` segfaulted** (exit 139 on load) on this Anaconda + pip-torch env; dropped it and ran bge through `transformers` directly — fixed the crash and removed a heavy dep.
  - **Step 3 — search** (`retrieval/search.py`). `Retriever` → typed `SearchHit`s. Verified sub-topic discrimination (nutrition vs CT imaging vs EMG-methods queries return disjoint, on-topic results).
  - **Step 4 — generation** (`generation/prompt.py`, `generation/generate.py`). Grounding system prompt (answer only from context, cite PMIDs, decline if unsupported) + 4-bit Qwen.
  - **Step 5 — wire** (`generation/rag.py`). `RagPipeline.ask()` single entry point. Full retrieve→generate runs in ~16 s for a ~300-token answer. **20 tests passing** (all GPU-free via fakes + in-memory qdrant).
- **Learned:**
  - **Prove the risky foundation before building on it.** Standing up + smoke-testing the GPU/4-bit path first (CUDA visible → model loads → generates) meant every later step rested on something known to work. Same instinct as the v0.1 manifest.
  - **Honest demos > flattering demos.** First retrieval "proof" was weak — every doc is about sarcopenia, so on-topic retrieval proves little. The real test is *within-corpus sub-topic discrimination*. And absolute cosine is a poor quality threshold (scores compress to ~0.6–0.86); ranking/gap is the signal.
  - **The baseline visibly fails sometimes — that's the point.** "Handgrip cutoffs" query: k=5 pulled a knee-extension paper and Qwen cited its values as handgrip — a faithfulness error. Logged to `eval_query_candidates.md`. You can't claim quality from anecdotes → motivates the `v0.3` eval harness. `k=5` is an untuned guess for the same reason.
  - **Drop the dep that fights you.** `sentence-transformers` segfault → going one layer down to the proven `transformers` path was faster than debugging a native crash, and left the stack leaner.
- **Next:** `v0.3-eval` — labelled query set + retrieval/generation metrics (recall@k, MRR, nDCG, faithfulness); tune `k`; establish the baseline numbers `v0.4`/`v0.5` are measured against. Commit + tag `v0.2-rag-baseline` first.