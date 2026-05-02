# `data/`

This directory is **gitignored** — no raw corpora or model artifacts are committed.

From `v0.1-corpus` onward, the ingestion script writes its output here in a versioned, HuggingFace-`datasets`-compatible layout, and the directory contents are tracked by [DVC](https://dvc.org/) (or referenced by an HF datasets pointer — decision in `v0.1`).

This README and `.gitkeep` are the only files in this folder that should ever be tracked by git.