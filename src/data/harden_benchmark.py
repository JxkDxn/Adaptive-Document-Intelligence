from pathlib import Path
import pandas as pd
import re
import random

ROOT = Path("data")
MANIFEST = ROOT / "manifest.csv"
RAW = ROOT / "raw"

df = pd.read_csv(MANIFEST)

LABELS = sorted(df["class_label"].unique())

# Explicit class-name leakage that should not appear verbatim in document bodies.
LEAKS = {
    "Bank Statement": [
        "BANK STATEMENT", "BANK ACCOUNT STATEMENT", "SAVINGS ACCOUNT STATEMENT"
    ],
    "Legal Contract": [
        "LEGAL CONTRACT", "AGREEMENT", "CONTRACT"
    ],
    "Proof of Identity": [
        "PROOF OF IDENTITY", "IDENTITY DOCUMENT", "IDENTITY PROOF"
    ],
    "Proof of Address": [
        "PROOF OF ADDRESS", "ADDRESS PROOF"
    ],
    "Loan application form": [
        "LOAN APPLICATION", "LOAN APPLICATION FORM"
    ],
    "Salary Slip": [
        "SALARY SLIP", "PAYSLIP", "PAY SLIP"
    ],
    "ITR": [
        "INCOME TAX RETURN", "ITR FORM", "ITR"
    ],
    "Property & Collateral Documents": [
        "PROPERTY & COLLATERAL", "PROPERTY DOCUMENT", "COLLATERAL DOCUMENT"
    ],
    "Credit Card Application": [
        "CREDIT CARD APPLICATION", "CARD APPLICATION"
    ],
    "Credit Card statement": [
        "CREDIT CARD STATEMENT", "CARD STATEMENT"
    ],
    "Class 10 Marksheet": [
        "CLASS 10", "SECONDARY SCHOOL EXAMINATION", "MATRICULATION"
    ],
    "Class 12 Marksheet": [
        "CLASS 12", "SENIOR SCHOOL CERTIFICATE", "HIGHER SECONDARY"
    ],
    "Company Constitution & Board Resolution": [
        "COMPANY CONSTITUTION", "BOARD RESOLUTION"
    ],
    "GST Registration / GST Return": [
        "GST REGISTRATION", "GST RETURN", "GST REGISTRATION / GST RETURN"
    ],
    "Audited Financial Statements": [
        "AUDITED FINANCIAL STATEMENTS", "AUDITED FINANCIALS"
    ],
}

# Generic cross-domain distractors.
DISTRACTORS = [
    "The document was received through the normal processing channel.",
    "Customer information and reference details are recorded for verification.",
    "The record contains dates, identifiers, and supporting information.",
    "This document may be reviewed as part of an internal verification workflow.",
    "The information below is subject to standard validation and record checks.",
    "Reference numbers and supporting details should be retained for audit purposes.",
    "The submitted information should be checked against the corresponding source record.",
]

def clean(text, label):
    if not isinstance(text, str):
        return ""

    text = text.replace("(SYNTHETIC)", "")
    text = text.replace("SYNTHETIC", "")

    # Remove the target class name and known explicit variants.
    for phrase in LEAKS.get(label, []):
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)

    # Remove obvious metadata leakage.
    text = re.sub(r"document\s*type\s*[:=-].*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"class\s*label\s*[:=-].*", "", text, flags=re.IGNORECASE)

    # Collapse whitespace left by removals.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

rng = random.Random(42)

changed = 0

for _, row in df.iterrows():
    path = RAW / row["filename"]
    if not path.exists():
        continue

    original = path.read_text(errors="ignore")
    text = clean(original, row["class_label"])

    # Add realistic generic language and occasional cross-domain distractors.
    additions = rng.sample(DISTRACTORS, 2)

    # Hard cases receive stronger cross-domain contamination.
    if bool(row.get("is_hard_case", False)):
        other_labels = [x for x in LABELS if x != row["class_label"]]
        other = rng.choice(other_labels)

        distractor_terms = {
            "Bank Statement": "account number, transaction date, debit and credit entries",
            "Legal Contract": "party details, effective date, obligations and signatures",
            "Proof of Identity": "name, date of birth, identification number",
            "Proof of Address": "residential address, correspondence details, verification",
            "Loan application form": "applicant details, requested amount, repayment information",
            "Salary Slip": "employee details, pay period, earnings and deductions",
            "ITR": "income details, tax computation, financial year",
            "Property & Collateral Documents": "property details, ownership, valuation and supporting records",
            "Credit Card Application": "applicant details, card request, eligibility information",
            "Credit Card statement": "account activity, transactions, statement period and balance",
            "Class 10 Marksheet": "student details, subjects, examination results",
            "Class 12 Marksheet": "student details, subjects, examination results",
            "Company Constitution & Board Resolution": "company details, directors, authorization and resolution",
            "GST Registration / GST Return": "taxpayer details, registration information, reporting period",
            "Audited Financial Statements": "financial position, revenue, expenses and reporting period",
        }

        additions.append(
            f"Related records may contain {distractor_terms.get(other, 'supporting information')}."
        )

    text = text + "\n\n" + "\n".join(additions)

    if text != original:
        path.write_text(text)
        changed += 1

print(f"Hardened {changed} documents.")
print("Removed explicit class-name leakage and added shared/cross-domain language.")
