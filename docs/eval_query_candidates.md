# Eval query candidates (raw material for v0.3)

A running scratchpad of test questions noticed while building v0.2. **Not** the
eval harness — just cheap-to-capture raw material so v0.3 doesn't start from a
blank page. For each, jot the sub-topic and any abstracts that *looked*
relevant (ideally PMIDs) so ground-truth labelling later is faster.

Format: `query` — sub-topic — notes / candidate-relevant PMIDs.

## Captured so far (from v0.2 Step 2 retrieval checks)

- **"Can surface EMG detect age-related muscle loss in older adults?"** — EMG/diagnosis — top hits were EMG-in-sarcopenia reviews (2025) and sEMG-in-aging papers; looked strong.
- **"protein and vitamin D supplementation to prevent muscle loss"** — nutrition/intervention — returned protein + vitamin D + exercise intervention papers.
- **"deep learning segmentation of skeletal muscle on CT scans"** — imaging/AI — returned automated CT muscle-segmentation papers (highest scores seen, ~0.86).
- **"motor unit number estimation from EMG signal decomposition"** — EMG methods — returned MUNIX / motor-unit-firing / MUP-classification papers.
- **"What handgrip strength cutoff values are used to diagnose sarcopenia in older adults?"** — diagnosis/cutoffs — ⚠️ **known faithfulness failure (great negative test case).** With k=5, retrieval surfaced PMID 32470897 (a *knee-extension* strength paper) alongside the handgrip papers, because "cut-off + strength + sarcopenia" dominates the embedding over the muscle/measurement distinction. The baseline Qwen answer then cited PMID 32470897's knee-extension cutoffs (38.1 kg women / 56.1 kg men) **as if they were handgrip values** — a claim attributed to a source that doesn't support it. Relevant/correct sources retrieved: PMID 40886127, 35587757 (true handgrip cutoff papers). v0.3 faithfulness metric should flag this; also a test case for tuning k (a smaller k might have excluded 32470897).

## To add as we go

- Questions that *should* have an answer in the corpus (positive cases).
- Questions that should **not** (out-of-scope) — to test that the system declines or returns low-confidence rather than hallucinating.
- Any retrieval miss observed (relevant paper we know exists but didn't surface) — these are the most valuable for tuning.

## v0.3 will turn these into

- A labelled query set (query → known-relevant PMIDs).
- Metrics: recall@k, MRR, nDCG for retrieval; faithfulness/groundedness for generation.
