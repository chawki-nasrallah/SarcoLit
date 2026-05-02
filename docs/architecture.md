# Architecture

This document grows with the milestones. At `v0.0-scaffold` it is intentionally a placeholder.

## Component map (target by `v0.7`)

```
┌────────────┐    ┌─────────────────┐    ┌──────────────┐
│  PubMed    │ →  │  ingest         │ →  │  vector DB   │
│  (E-utils) │    │  (chunk + meta) │    │  (qdrant)    │
└────────────┘    └─────────────────┘    └──────┬───────┘
                                                │
                       ┌────────────────────────┴─────────────┐
                       │                                      │
                ┌──────▼──────┐                       ┌───────▼─────┐
                │  retrieval  │                       │  agent      │
                │  hybrid +   │ ◄──── tool calls ──── │  (tools,    │
                │  rerank     │                       │   routing)  │
                └──────┬──────┘                       └───────┬─────┘
                       │                                      │
                       └─────────► generation ◄───────────────┘
                                   (LoRA-tuned 3B)
                                          │
                                          ▼
                                    FastAPI / Gradio
                                          │
                                          ▼
                                       observe
```

## Decisions log

Architecture decisions worth preserving (one-liner each, dated). Add an entry whenever a non-obvious choice is made.

- `2026-05-02` — Project path locked as RAG + Agent (SarcoLit), biosignal extension deferred to optional `v1.0`. Rationale: foundation-model fluency is the hiring screen for 2026; biosignal expertise is a differentiator added on top, not the headline.