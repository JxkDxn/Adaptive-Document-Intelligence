"""Classification metric calculations."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Return accuracy, F1 variants, per-class scores, and confusion matrix."""
    n = len(y_true)
    if n == 0:
        return {
            "n_documents": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "per_class": {},
            "confusion_matrix": [],
        }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = {}
    for label in labels:
        row = report.get(label, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        per_class[label] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_documents": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(labels),
    }
