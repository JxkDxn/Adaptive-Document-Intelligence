# Adaptive Document Intelligence

A cost-aware, confidence-driven document intelligence pipeline designed for large-scale heterogeneous document collections.

## Problem

Large document collections contain heterogeneous formats and visually or semantically similar document types. A robust classification system needs to balance accuracy, inference cost, latency, and human review.

## Approach

The system uses progressive classification:

1. Use inexpensive metadata and deterministic signals first.
2. Extract text from supported document formats.
3. Run a lightweight local classifier.
4. Accept high-confidence predictions.
5. Escalate uncertain cases to stronger adjudication.
6. Route unresolved cases to human review.
7. Track accuracy, coverage, review volume, and cost.

## Planned Architecture

```text
Manifest
    ↓
Metadata / Duplicate Signals
    ↓
Content Extraction
    ↓
Local Classifier
    ↓
Confidence Gate
    ├── High confidence → Classification
    └── Uncertain → Stronger Adjudication
                         ↓
                    Human Review

## Benchmark Results

The prototype was evaluated on a reproducible synthetic benchmark containing
900 documents across 15 document classes, with train/validation/test splits.
The reported classification metrics below are measured on the held-out test
set of 135 documents.

| Metric | Result |
|---|---:|
| Test documents | 135 |
| Accuracy | 100.00% |
| Macro-F1 | 100.00% |
| Critical-class minimum recall | 100.00% |
| Auto-classified | 134 |
| Auto coverage | 99.26% |
| Second-stage adjudication | 1 |
| Human review | 0 |

> **Benchmark caveat:** This is a synthetic sanity benchmark, not production
> evidence. The dataset contains strong textual class cues, so these results
> should not be interpreted as expected real-world accuracy.

### Cost Measurement

The prototype operates on pre-extracted text from the manifest. The measured
prototype cost therefore covers the classification pipeline and the notional
S3 read charge only. OCR, document extraction, and external LLM costs are
excluded.

| Cost metric | Result |
|---|---:|
| Runtime | measured per execution |
| Prototype cost | measured per execution |
| Cost / document | measured on 135-document test set |
| 5M-document projection | classification-only projection |
| Budget | ₹100,000 |