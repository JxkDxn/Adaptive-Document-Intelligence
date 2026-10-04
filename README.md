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
| Accuracy | 93.33% |
| Macro-F1 | 87.50% |
| Critical-class minimum recall | 100.00% |
| Human review | 9 |
| Second-stage adjudication | 126 |
| Prototype cost | ₹0.0313 |
| Cost / document | ₹0.000035 |
| Projected cost for 5M documents | ₹173.74 |
| Budget | ₹100,000 |
| Budget status | PASS |

### Reproducibility note

The benchmark stores representative extracted document text in the manifest,
allowing the classification and routing pipeline to be executed without
requiring 900 physical source files.

The second-stage adjudication used in this benchmark is a deterministic local
adjudication fallback. No external paid LLM API was invoked. The architecture
is designed so that this stage can be replaced by an external LLM adjudicator
for production workloads when stronger semantic reasoning is required.

The reported cost therefore represents the actual measured prototype
execution cost and does not include hypothetical external LLM charges.