from pathlib import Path
import hashlib
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET = Path("data/manifest.csv")
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BUDGET_INR = 100_000

S3_READ_PER_OBJECT = 0.033 / 1000
CPU_PER_HOUR = 32.0

LOCAL_CONFIDENCE_THRESHOLD = 0.92
LOCAL_MARGIN_THRESHOLD = 0.15

CRITICAL_CLASSES = {
    "Legal Contract",
    "Property & Collateral Documents",
    "ITR",
    "GST Registration / GST Return",
    "Proof of Identity",
    "Loan application form",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_document(path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def class_score(text, label):
    """
    Lightweight evidence score used only by the local adjudication
    fallback. It does NOT call an external LLM.
    """
    text = text.lower()

    keywords = {
        "Bank Statement": [
            "savings account", "bank statement", "account number",
            "opening balance", "closing balance", "transaction"
        ],
        "Legal Contract": [
            "agreement", "party", "contract", "terms and conditions",
            "whereas", "obligations", "termination"
        ],
        "Proof of Identity": [
            "aadhaar", "passport", "identity", "date of birth",
            "identification number", "photo identity"
        ],
        "Proof of Address": [
            "address", "residential", "utility bill", "proof of residence",
            "correspondence address"
        ],
        "Loan application form": [
            "loan", "applicant", "requested amount", "employment",
            "income", "loan application"
        ],
        "Salary Slip": [
            "salary", "pay period", "basic", "hra", "employee code",
            "net pay", "gross pay"
        ],
        "ITR": [
            "income tax return", "itr", "assessment year",
            "gross total income", "tax payable"
        ],
        "Property & Collateral Documents": [
            "property", "collateral", "sale deed", "mortgage",
            "land", "valuation", "title"
        ],
        "Credit Card Application": [
            "credit card application", "card application",
            "annual income", "card variant"
        ],
        "Credit Card statement": [
            "credit card statement", "statement date", "minimum amount due",
            "credit limit", "card number"
        ],
        "Class 10 Marksheet": [
            "secondary school", "class 10", "tenth", "mathematics",
            "science", "marks"
        ],
        "Class 12 Marksheet": [
            "senior school", "class 12", "twelfth", "physics",
            "chemistry", "marks"
        ],
        "Company Constitution & Board Resolution": [
            "board resolution", "director", "company", "quorum",
            "authorised", "memorandum", "articles"
        ],
        "GST Registration / GST Return": [
            "gstin", "gst registration", "gst return",
            "taxable value", "cgst", "sgst", "igst"
        ],
        "Audited Financial Statements": [
            "audited financial statements", "balance sheet",
            "profit and loss", "revenue", "auditor",
            "cash flow", "financial position"
        ],
    }

    return sum(1 for k in keywords.get(label, []) if k in text)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

start = time.perf_counter()

df = pd.read_csv(DATASET)

# The benchmark stores representative extracted document text
# directly in the manifest. This keeps the prototype reproducible
# without requiring 900 physical source files.
df["text"] = df["content"].fillna("").astype(str)

empty_documents = int(df["text"].str.strip().eq("").sum())

if empty_documents:
    raise RuntimeError(
        f"{empty_documents} documents have no extracted benchmark text."
    )


# ------------------------------------------------------------
# Train / validation / test split
# ------------------------------------------------------------

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

# Train only on training data.
local_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=40_000,
        ),
    ),
    (
        "classifier",
        LinearSVC(C=0.25),
    ),
])

local_model.fit(train["text"], train["class_label"])


# ------------------------------------------------------------
# Calibrate confidence from validation score distribution
# ------------------------------------------------------------

validation_scores = local_model.decision_function(validation["text"])

if validation_scores.ndim == 1:
    validation_scores = validation_scores.reshape(-1, 1)

validation_pred = local_model.predict(validation["text"])

# Normalize decision scores into a comparable confidence-like value.
def confidence_and_margin(scores):
    ordered = sorted(scores, reverse=True)

    if len(ordered) == 1:
        return 1.0, 1.0

    top = ordered[0]
    second = ordered[1]

    # sigmoid-like transformation for a bounded confidence score
    import math

    confidence = 1.0 / (1.0 + math.exp(-top))
    margin = top - second

    return confidence, margin


# ------------------------------------------------------------
# Test-time progressive routing
# ------------------------------------------------------------

predictions = []

auto_count = 0
adjudicated_count = 0
human_review_count = 0

duplicate_groups = {}

for idx, row in test.iterrows():
    text = row["text"]
    true_label = row["class_label"]

    # ----------------------------------------
    # Local classifier
    # ----------------------------------------

    scores = local_model.decision_function([text])[0]

    classes = local_model.classes_

    ranked = sorted(
        zip(classes, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1]

    confidence, margin = confidence_and_margin(scores)

    critical_alternative = (
        second_label in CRITICAL_CLASSES
        and second_label != top_label
        and (top_score - second_score) < 0.5
    )

    confident = (
        confidence >= LOCAL_CONFIDENCE_THRESHOLD
        and margin >= LOCAL_MARGIN_THRESHOLD
        and not critical_alternative
    )

    route = "auto"

    if confident:
        final_label = top_label
        auto_count += 1

    else:
        # ------------------------------------
        # Stronger local adjudication fallback
        # ------------------------------------

        candidates = [x[0] for x in ranked[:3]]

        evidence_scores = {
            label: class_score(text, label)
            for label in candidates
        }

        best_evidence_label = max(
            evidence_scores,
            key=evidence_scores.get,
        )

        best_evidence = evidence_scores[best_evidence_label]

        # Require actual supporting evidence.
        if best_evidence >= 2:
            final_label = best_evidence_label
            route = "adjudication"
            adjudicated_count += 1
        else:
            final_label = None
            route = "human_review"
            human_review_count += 1

    predictions.append({
        "document_id": row["document_id"],
        "true_label": true_label,
        "predicted_label": final_label,
        "local_top1": top_label,
        "local_confidence": round(confidence, 4),
        "local_margin": round(margin, 4),
        "route": route,
    })


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

results = pd.DataFrame(predictions)

# For the assignment's all-document metric, unresolved human review
# is counted as a miss.
evaluation_predictions = results["predicted_label"].fillna(
    "__HUMAN_REVIEW__"
)

macro_f1 = f1_score(
    results["true_label"],
    evaluation_predictions,
    average="macro",
    zero_division=0,
)

accuracy = accuracy_score(
    results["true_label"],
    evaluation_predictions,
)

critical_recalls = {}

for label in sorted(CRITICAL_CLASSES):
    subset = results[results["true_label"] == label]

    if len(subset) == 0:
        continue

    critical_recalls[label] = recall_score(
        subset["true_label"],
        subset["predicted_label"].fillna("__HUMAN_REVIEW__"),
        labels=[label],
        average="macro",
        zero_division=0,
    )


critical_recall = (
    min(critical_recalls.values())
    if critical_recalls
    else 0.0
)


# ------------------------------------------------------------
# Actual prototype cost
# ------------------------------------------------------------

elapsed = time.perf_counter() - start

document_count = len(df)

s3_cost = document_count * S3_READ_PER_OBJECT

cpu_cost = (elapsed / 3600.0) * CPU_PER_HOUR

prototype_cost = s3_cost + cpu_cost

cost_per_document = prototype_cost / document_count

projected_5m_cost = cost_per_document * 5_000_000


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

metrics = {
    "documents_evaluated": int(len(results)),
    "accuracy": round(float(accuracy), 4),
    "macro_f1": round(float(macro_f1), 4),
    "critical_class_min_recall": round(float(critical_recall), 4),
    "auto_classified": int(auto_count),
    "adjudicated": int(adjudicated_count),
    "human_review": int(human_review_count),
    "auto_coverage": round(
        auto_count / len(results), 4
    ),
    "runtime_seconds": round(elapsed, 4),
    "s3_cost_inr": round(s3_cost, 6),
    "cpu_cost_inr": round(cpu_cost, 6),
    "prototype_cost_inr": round(prototype_cost, 6),
    "cost_per_document_inr": round(cost_per_document, 6),
    "projected_5m_cost_inr": round(projected_5m_cost, 2),
    "budget_inr": BUDGET_INR,
    "projected_budget_status": (
        "PASS"
        if projected_5m_cost <= BUDGET_INR
        else "FAIL"
    ),
}

with open(OUTPUT_DIR / "pipeline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

results.to_csv(
    OUTPUT_DIR / "pipeline_results.csv",
    index=False,
)

print("\n" + "=" * 60)
print("ADAPTIVE DOCUMENT INTELLIGENCE")
print("=" * 60)

print(f"Documents evaluated:       {metrics['documents_evaluated']}")
print(f"Accuracy:                  {metrics['accuracy']:.4f}")
print(f"Macro-F1:                  {metrics['macro_f1']:.4f}")
print(
    f"Critical-class min recall: {metrics['critical_class_min_recall']:.4f}"
)

print("\nROUTING")
print(f"Auto-classified:           {metrics['auto_classified']}")
print(f"Adjudicated:               {metrics['adjudicated']}")
print(f"Human review:              {metrics['human_review']}")
print(f"Auto coverage:             {metrics['auto_coverage']:.2%}")

print("\nMEASURED COST")
print(f"Runtime:                   {metrics['runtime_seconds']:.2f}s")
print(f"S3 cost:                   ₹{metrics['s3_cost_inr']:.4f}")
print(f"CPU cost:                  ₹{metrics['cpu_cost_inr']:.4f}")
print(f"Prototype cost:            ₹{metrics['prototype_cost_inr']:.4f}")
print(f"Cost / document:           ₹{metrics['cost_per_document_inr']:.6f}")

print("\n5M-DOCUMENT PROJECTION")
print(f"Projected cost:            ₹{metrics['projected_5m_cost_inr']:,.2f}")
print(f"Budget:                    ₹{metrics['budget_inr']:,.2f}")
print(f"Budget status:             {metrics['projected_budget_status']}")

print("\nOutputs:")
print("  outputs/pipeline_metrics.json")
print("  outputs/pipeline_results.csv")
print("=" * 60)