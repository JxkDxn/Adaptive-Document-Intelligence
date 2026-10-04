"""Run the first local classification baseline on the synthetic benchmark.

Uses the manifest `content` column as already-extracted text. This does not
measure OCR or parser quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.classification.local_classifier import SEED, LocalDocumentClassifier
from src.evaluation.metrics import evaluate_predictions

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
OUTPUT_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "baseline_confusion_matrix.csv"
VALID_SPLITS = {"train", "validation", "test"}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split"].isin(VALID_SPLITS)].copy()
    frame["content"] = frame["content"].fillna("").astype(str)
    frame["class_label"] = frame["class_label"].astype(str)
    frame["is_exact_duplicate"] = frame["is_exact_duplicate"].map(_as_bool)
    frame["is_near_duplicate"] = frame["is_near_duplicate"].map(_as_bool)
    frame["has_misleading_filename"] = frame["has_misleading_filename"].map(_as_bool)
    return frame


def subset_metrics(test: pd.DataFrame, y_pred: list[str], labels: list[str], mask: pd.Series) -> dict:
    y_true = test.loc[mask, "class_label"].tolist()
    pred = [p for p, keep in zip(y_pred, mask.tolist()) if keep]
    return evaluate_predictions(y_true, pred, labels)


def write_confusion_csv(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true_label"
    frame.to_csv(path)


def print_summary(payload: dict) -> None:
    test = payload["test"]
    print("Local classification baseline")
    print("Assumption: predict from manifest content only (text stand-ins, not OCR/parser).")
    print(f"Seed: {payload['seed']}  |  LinearSVC C: {payload['selected_C']}")
    print(
        f"Documents: train={payload['n_train']}  "
        f"validation={payload['n_validation']}  test={payload['n_test']}"
    )
    print()
    print("Test set")
    print(f"  n             {test['n_documents']}")
    print(f"  accuracy      {test['accuracy']:.4f}")
    print(f"  macro F1      {test['macro_f1']:.4f}")
    print(f"  weighted F1   {test['weighted_f1']:.4f}")
    print()
    print("Per-class (test)")
    print(f"  {'class':42s} {'P':>7s} {'R':>7s} {'F1':>7s} {'n':>5s}")
    for label, row in test["per_class"].items():
        print(
            f"  {label:42s} {row['precision']:7.3f} {row['recall']:7.3f} "
            f"{row['f1']:7.3f} {row['support']:5d}"
        )
    print()
    print("Test subsets")
    for name, block in payload["subsets"].items():
        n = block["n_documents"]
        if n == 0:
            print(f"  {name:22s} n=0")
            continue
        print(
            f"  {name:22s} n={n:3d}  acc={block['accuracy']:.4f}  "
            f"macro_F1={block['macro_f1']:.4f}"
        )
    print()
    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {CONFUSION_PATH}")


def main() -> None:
    frame = load_manifest(MANIFEST_PATH)
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)

    labels = sorted(frame["class_label"].unique().tolist())
    classifier = LocalDocumentClassifier(random_state=SEED)
    classifier.fit(
        texts=train["content"].tolist(),
        labels=train["class_label"].tolist(),
        validation_texts=validation["content"].tolist(),
        validation_labels=validation["class_label"].tolist(),
    )
    y_pred = classifier.predict(test["content"].tolist())
    y_true = test["class_label"].tolist()

    test_metrics = evaluate_predictions(y_true, y_pred, labels)
    subsets = {
        "hard_cases": subset_metrics(test, y_pred, labels, test["difficulty"] == "hard"),
        "misleading_filenames": subset_metrics(
            test, y_pred, labels, test["has_misleading_filename"]
        ),
        "near_duplicates": subset_metrics(test, y_pred, labels, test["is_near_duplicate"]),
        "exact_duplicates": subset_metrics(test, y_pred, labels, test["is_exact_duplicate"]),
    }

    payload = {
        "seed": SEED,
        "assumption": (
            "Uses the manifest content column as extracted text because the current "
            "benchmark stores text-bearing stand-in documents. This baseline does not "
            "represent OCR or parser performance."
        ),
        "features": "TF-IDF word n-grams (1-2) and character n-grams (3-5); LinearSVC.",
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "selected_C": classifier.C_,
        "classes": labels,
        "test": test_metrics,
        "subsets": subsets,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_confusion_csv(CONFUSION_PATH, labels, test_metrics["confusion_matrix"])
    print_summary(payload)


if __name__ == "__main__":
    main()
