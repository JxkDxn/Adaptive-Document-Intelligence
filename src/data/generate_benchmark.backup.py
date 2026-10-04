"""Generate a 900-document synthetic classification benchmark.

Files are locally generated text-bearing stand-ins. The manifest records the
intended format and generation characteristics; no OCR, parsers, or models
are used here.
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SEED = 42
N_PER_CLASS = 60
N_DOCS = 900
N_UNIQUE_PER_CLASS = 44
N_EXACT_DUP_PER_CLASS = 8
N_NEAR_DUP_PER_CLASS = 8

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
DATA_README_PATH = DATA_DIR / "README.md"

CLASSES = [
    "Bank Statement",
    "Legal Contract",
    "Proof of Identity",
    "Proof of Address",
    "Loan application form",
    "Salary Slip",
    "ITR",
    "Property & Collateral Documents",
    "Credit Card Application",
    "Credit Card statement",
    "Class 10 Marksheet",
    "Class 12 Marksheet",
    "Company Constitution & Board Resolution",
    "GST Registration / GST Return",
    "Audited Financial Statements",
]

CONFUSABLE = {
    "Proof of Identity": "Proof of Address",
    "Proof of Address": "Proof of Identity",
    "Bank Statement": "Credit Card statement",
    "Credit Card statement": "Bank Statement",
    "Class 10 Marksheet": "Class 12 Marksheet",
    "Class 12 Marksheet": "Class 10 Marksheet",
    "ITR": "GST Registration / GST Return",
    "GST Registration / GST Return": "ITR",
    "Legal Contract": "Property & Collateral Documents",
    "Property & Collateral Documents": "Legal Contract",
}

FORMAT_COUNTS = {
    "pdf": 405,
    "jpg": 162,
    "tiff": 90,
    "docx": 72,
    "png": 54,
    "eml": 27,
    "msg": 18,
    "xlsx": 36,
    "txt": 18,
    "pptx": 14,
    "rtf": 4,
}

BANKS = [
    "HDFC Bank",
    "ICICI Bank",
    "State Bank of India",
    "Axis Bank",
    "Kotak Mahindra Bank",
]
CITIES = [
    "Bengaluru",
    "Mumbai",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Delhi",
    "Kolkata",
    "Ahmedabad",
]
FIRST = ["Ananya", "Rahul", "Fatima", "Vikram", "Sneha", "Arjun", "Meera", "Kabir", "Isha", "Dev"]
LAST = ["Sharma", "Iyer", "Khan", "Reddy", "Nair", "Patel", "Das", "Joshi", "Menon", "Gupta"]
COMPANIES = [
    "Meridian Textiles Pvt Ltd",
    "Harborlight Logistics Ltd",
    "Nimbus Payments Pvt Ltd",
    "Cedar Grove Properties Ltd",
    "Helios Manufacturing Pvt Ltd",
]
BOARDS = [
    "CBSE",
    "CISCE",
    "State Board",
    "NIOS",
]
SCHOOLS = [
    "Greenfield Public School",
    "Lakeside Senior Secondary School",
    "St. Martin's High School",
    "Vidya Niketan",
]


def slug(label: str) -> str:
    return (
        label.lower()
        .replace("&", "and")
        .replace("/", " ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "_")
    )


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def pan(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return f"{letters}{digits}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"


def aadhaar(rng: random.Random) -> str:
    return " ".join("".join(str(rng.randint(0, 9)) for _ in range(4)) for _ in range(3))


def gstin(rng: random.Random, pan_no: str) -> str:
    state = rng.choice(["27", "29", "36", "33", "07", "24"])
    return f"{state}{pan_no}1Z{rng.choice('123456789ABCDEF')}"


def ifsc(rng: random.Random) -> str:
    return f"{rng.choice(['HDFC', 'ICIC', 'SBIN', 'UTIB', 'KKBK'])}0{rng.randint(100000, 999999)}"


def acct(rng: random.Random) -> str:
    return str(rng.randint(10**11, 10**12 - 1))


def amount(rng: random.Random, lo: float = 1200.0, hi: float = 285000.0) -> str:
    return f"{rng.uniform(lo, hi):,.2f}"


def date(rng: random.Random, year: int | None = None) -> str:
    y = year or rng.choice([2022, 2023, 2024, 2025])
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{d:02d}-{m:02d}-{y}"


def address(rng: random.Random) -> str:
    return (
        f"{rng.randint(12, 240)} {rng.choice(['MG Road', 'Lake View', 'Residency', 'Industrial Area'])}, "
        f"{rng.choice(CITIES)} {rng.randint(400001, 560102)}"
    )


def ocr_corrupt(text: str, rng: random.Random, rate: float = 0.07) -> str:
    mapping = {"a": "4", "e": "3", "o": "0", "l": "1", "s": "5", "t": "7", "B": "8", "g": "9"}
    out = []
    for ch in text:
        if rng.random() < rate and ch in mapping:
            out.append(mapping[ch])
        elif rng.random() < rate / 4:
            out.append(ch + " ")
        else:
            out.append(ch)
    return "".join(out)


def sparsify(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:3]) + ("\n" if lines else "")


def wrap_format(fmt: str, title: str, body: str) -> str:
    if fmt == "pdf":
        return f"%PDF-1.4\n% synthetic text-bearing stand-in\n1 0 obj\n<< /Title ({title}) >>\nstream\n{body}\nendstream\nendobj\n%%EOF\n"
    if fmt in {"jpg", "png", "tiff"}:
        return f"SYNTHETIC-{fmt.upper()}-IMAGE\n# raster stand-in; OCR text follows\nTITLE: {title}\n{body}\n"
    if fmt == "docx":
        return (
            "<?xml version='1.0' encoding='UTF-8'?>\n"
            "<w:document>\n"
            f"<w:title>{title}</w:title>\n"
            f"<w:body>{body}</w:body>\n"
            "</w:document>\n"
        )
    if fmt == "xlsx":
        rows = [f"title,{title}"]
        for i, line in enumerate(body.splitlines()):
            if line.strip():
                rows.append(f"r{i},{line.replace(',', ';')}")
        return "\n".join(rows) + "\n"
    if fmt in {"eml", "msg"}:
        return (
            f"From: records-noreply@synthetic.example\n"
            f"To: intake@lender.example\n"
            f"Subject: {title}\n"
            f"MIME-Version: 1.0\n"
            f"Content-Type: text/plain; charset=UTF-8\n\n"
            f"{body}\n"
        )
    if fmt == "pptx":
        slides = [f"# Slide 1: {title}"]
        chunks = [ln for ln in body.splitlines() if ln.strip()]
        for i in range(0, min(len(chunks), 12), 4):
            slides.append(f"# Slide {len(slides)+1}\n" + "\n".join(chunks[i : i + 4]))
        return "\n\n".join(slides) + "\n"
    if fmt == "rtf":
        return "{\\rtf1\\ansi\n" + body.replace("\n", "\\par\n") + "\n}\n"
    return f"{title}\n\n{body}\n"


def apply_quality(text: str, quality: str, rng: random.Random) -> str:
    if quality == "sparse":
        return sparsify(text)
    if quality == "noisy":
        return ocr_corrupt(text, rng, rate=0.09)
    if quality == "ocr":
        return ocr_corrupt(text, rng, rate=0.05)
    return text


def near_mutate(text: str, rng: random.Random) -> str:
    mutated = ocr_corrupt(text, rng, rate=0.03)
    return mutated + "\n[rescanned copy; low-resolution archive]\n"


def body_bank_statement(rng: random.Random, ident: dict, hard: bool) -> str:
    bank = ident["bank"]
    name = ident["name"]
    extra = ""
    if hard:
        extra = (
            f"\nCredit facility reference: CARD-{rng.randint(1000, 9999)}\n"
            f"Available credit mentioned for overdraft: INR {amount(rng, 20000, 80000)}\n"
            "Minimum amount due: not applicable (savings account).\n"
        )
    txns = "\n".join(
        f"{date(rng)}  {'CR' if i % 2 == 0 else 'DR'}  INR {amount(rng, 80, 24000)}  "
        f"{rng.choice(['UPI', 'NEFT', 'POS', 'ATM', 'IMPS'])}"
        for i in range(8)
    )
    return (
        f"{bank} — Savings Account Statement\n"
        f"Account holder: {name}\n"
        f"Account no: {ident['account']}\n"
        f"IFSC: {ident['ifsc']}\n"
        f"Statement period: {date(rng, 2024)} to {date(rng, 2025)}\n"
        f"Opening balance: INR {amount(rng, 8000, 90000)}\n"
        f"{txns}\n"
        f"Closing balance: INR {amount(rng, 5000, 110000)}\n"
        f"Address on file: {ident['address']}\n"
        f"{extra}"
        "This is a bank account statement, not a card statement.\n"
    )


def body_cc_statement(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = ""
    if hard:
        extra = (
            f"\nLinked bank account (masked): {ident['account'][-4:]}\n"
            f"IFSC of auto-debit account: {ident['ifsc']}\n"
        )
    txns = "\n".join(
        f"{date(rng)}  INR {amount(rng, 120, 18000)}  {rng.choice(['POS', 'ECOM', 'FUEL', 'EMI'])}"
        for _ in range(8)
    )
    return (
        f"{ident['bank']} — Credit Card Statement\n"
        f"Cardholder: {ident['name']}\n"
        f"Card number: XXXX-XXXX-XXXX-{rng.randint(1000, 9999)}\n"
        f"Statement date: {date(rng, 2025)}\n"
        f"Credit limit: INR {amount(rng, 50000, 400000)}\n"
        f"Available credit: INR {amount(rng, 8000, 120000)}\n"
        f"Minimum amount due: INR {amount(rng, 800, 9000)}\n"
        f"Total amount due: INR {amount(rng, 2000, 45000)}\n"
        f"{txns}\n"
        f"{extra}"
        "Payment due date applies. This is not a savings account statement.\n"
    )


def body_identity(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = f"\nAddress: {ident['address']}\n" if hard else "\nPhoto and signature present.\n"
    return (
        "PROOF OF IDENTITY — GOVERNMENT PHOTO ID (SYNTHETIC)\n"
        f"Name: {ident['name']}\n"
        f"Date of birth: {date(rng, rng.randint(1975, 2002))}\n"
        f"Aadhaar (synthetic): {ident['aadhaar']}\n"
        f"PAN (synthetic): {ident['pan']}\n"
        f"Gender: {rng.choice(['F', 'M', 'X'])}\n"
        f"{extra}"
        "This document is issued as identity evidence. Do not treat as address proof alone.\n"
    )


def body_address(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = ""
    if hard:
        extra = (
            f"\nCustomer ID / Aadhaar last 4 (synthetic): {ident['aadhaar'][-4:]}\n"
            "KYC photo attached on utility letterhead.\n"
        )
    return (
        f"{rng.choice(['BESCOM', 'MSEDCL', 'BWSSB', 'Indane Gas'])} — PROOF OF ADDRESS\n"
        f"Consumer name: {ident['name']}\n"
        f"Service address: {ident['address']}\n"
        f"Bill date: {date(rng, 2025)}\n"
        f"Consumer no: {rng.randint(1000000, 9999999)}\n"
        f"Amount: INR {amount(rng, 400, 4200)}\n"
        f"{extra}"
        "This is an address proof document (utility / residence), not a photo ID card.\n"
    )


def body_contract(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = ""
    if hard:
        extra = (
            f"\nSchedule mention: premises at {ident['address']}\n"
            "Stamp duty and registration clause included by reference.\n"
        )
    return (
        "MASTER SERVICES AGREEMENT (SYNTHETIC)\n"
        f"This Agreement is made on {date(rng, 2024)} between {ident['company']} ('Provider') "
        f"and {ident['name']} ('Client').\n"
        "WHEREAS the parties wish to set out terms for professional services;\n"
        "1. Scope of work as in Exhibit A.\n"
        "2. Fees payable within 30 days of invoice.\n"
        "3. Confidentiality and limitation of liability.\n"
        "4. Governing law: India. Arbitration in {city}.\n".replace("{city}", ident["city"])
        + extra
        + "This is a commercial contract, not a title or mortgage deed.\n"
    )


def body_property(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = ""
    if hard:
        extra = (
            "\nThe Mortgagor covenants as under a deed of agreement between the parties.\n"
            "IN WITNESS WHEREOF the parties have signed this instrument.\n"
        )
    return (
        "PROPERTY & COLLATERAL DOCUMENT — MEMORANDUM OF DEPOSIT / MORTGAGE (SYNTHETIC)\n"
        f"Dated: {date(rng, 2023)}\n"
        f"Mortgagor: {ident['name']}\n"
        f"Mortgagee: {ident['bank']}\n"
        f"Property schedule: {ident['address']}\n"
        f"Survey no: {rng.randint(10, 90)}/{rng.randint(1, 12)}\n"
        f"Market value: INR {amount(rng, 2_000_000, 18_000_000)}\n"
        f"Charge created to secure loan {ident['loan_id']}.\n"
        f"{extra}"
        "Encumbrance and original title deposit recorded. This is not a services contract.\n"
    )


def body_loan(rng: random.Random, ident: dict, hard: bool) -> str:
    return (
        "LOAN APPLICATION FORM (SYNTHETIC)\n"
        f"Applicant: {ident['name']}\n"
        f"PAN: {ident['pan']}\n"
        f"Employment: {ident['company']}\n"
        f"Requested amount: INR {amount(rng, 100000, 2500000)}\n"
        f"Purpose: {rng.choice(['Home', 'LAP', 'Personal', 'Education'])}\n"
        f"Declared monthly income: INR {amount(rng, 35000, 220000)}\n"
        f"Existing bank: {ident['bank']}  A/c {ident['account']}\n"
        f"Address: {ident['address']}\n"
        "Declaration: information is true and complete. Processing fee as applicable.\n"
        + ("Attached: bank statement and ID copies listed in annex.\n" if hard else "")
    )


def body_salary(rng: random.Random, ident: dict, hard: bool) -> str:
    basic = rng.randint(25000, 120000)
    return (
        f"{ident['company']} — SALARY SLIP\n"
        f"Employee: {ident['name']}\n"
        f"Employee code: EMP{rng.randint(1000, 9999)}\n"
        f"Pay period: {rng.choice(['Apr 2025', 'May 2025', 'Jun 2025', 'Jul 2025'])}\n"
        f"PAN: {ident['pan']}\n"
        f"Basic: INR {basic:,.2f}\n"
        f"HRA: INR {basic * 0.4:,.2f}\n"
        f"Special allowance: INR {basic * 0.2:,.2f}\n"
        f"PF / Tax deductions: INR {basic * 0.16:,.2f}\n"
        f"Net pay: INR {basic * 1.44:,.2f}\n"
        f"Credit to {ident['bank']} A/c {ident['account']}\n"
        + ("YTD TDS may be used for ITR working. This is a payslip, not a tax return.\n" if hard else "")
    )


def body_itr(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = ""
    if hard:
        extra = (
            f"\nGSTIN (if registered): {ident['gstin']}\n"
            "Turnover disclosure for business ITR schedule.\n"
        )
    return (
        "INCOME TAX RETURN ACKNOWLEDGEMENT (SYNTHETIC ITR)\n"
        f"Assessee: {ident['name']}\n"
        f"PAN: {ident['pan']}\n"
        f"Assessment year: {rng.choice(['2023-24', '2024-25', '2025-26'])}\n"
        f"ITR form: {rng.choice(['ITR-1', 'ITR-2', 'ITR-3', 'ITR-4'])}\n"
        f"e-Filing acknowledgement: {rng.randint(1000000000, 1999999999)}\n"
        f"Gross total income: INR {amount(rng, 400000, 2800000)}\n"
        f"Tax payable: INR {amount(rng, 0, 180000)}\n"
        f"{extra}"
        "This is an income-tax return artefact, not a GST return.\n"
    )


def body_gst(rng: random.Random, ident: dict, hard: bool) -> str:
    extra = ""
    if hard:
        extra = f"\nPAN of taxpayer: {ident['pan']}\nTurnover also relevant for income tax.\n"
    return (
        "GST REGISTRATION / GST RETURN (SYNTHETIC)\n"
        f"Legal name: {ident['company']}\n"
        f"Authorized signatory: {ident['name']}\n"
        f"GSTIN: {ident['gstin']}\n"
        f"Return period: {rng.choice(['GSTR-1', 'GSTR-3B'])} {rng.choice(['Apr-2025', 'May-2025', 'Q1-2025'])}\n"
        f"Taxable value: INR {amount(rng, 50000, 4000000)}\n"
        f"IGST/CGST/SGST payable: INR {amount(rng, 2000, 180000)}\n"
        f"Principal place: {ident['address']}\n"
        f"{extra}"
        "Filed on GST portal. This is not an income-tax ITR acknowledgement.\n"
    )


def body_cc_application(rng: random.Random, ident: dict, hard: bool) -> str:
    return (
        f"{ident['bank']} — CREDIT CARD APPLICATION\n"
        f"Applicant: {ident['name']}\n"
        f"PAN: {ident['pan']}\n"
        f"DOB: {date(rng, rng.randint(1978, 2000))}\n"
        f"Residence: {ident['address']}\n"
        f"Employer: {ident['company']}\n"
        f"Requested product: {rng.choice(['Regalia', 'Millennia', 'Amazon Pay', 'SimplySAVE'])}\n"
        f"Declared income: INR {amount(rng, 300000, 1800000)} per annum\n"
        "Consent for bureau check and marketing communication.\n"
        + ("Existing savings a/c for relationship pricing listed.\n" if hard else "")
    )


def body_marksheet(rng: random.Random, ident: dict, standard: str, hard: bool) -> str:
    other = "12" if standard == "10" else "10"
    extra = ""
    if hard:
        extra = (
            f"\nSchool also offers Class {other}. Candidate name appears on both rolls.\n"
            "Subjects overlap: English, Mathematics, Science-related papers.\n"
        )
    subjects_10 = [("English", 78), ("Mathematics", 84), ("Science", 81), ("Social Science", 76), ("Hindi", 72)]
    subjects_12 = [("English", 80), ("Mathematics", 88), ("Physics", 79), ("Chemistry", 74), ("Computer Science", 91)]
    subjects = subjects_10 if standard == "10" else subjects_12
    lines = "\n".join(f"{n}: {m}/100" for n, m in subjects)
    heading = (
        "SECONDARY SCHOOL EXAMINATION (CLASS 10) MARKSHEET"
        if standard == "10"
        else "SENIOR SCHOOL CERTIFICATE (CLASS 12) MARKSHEET"
    )
    return (
        f"{heading} — {ident['board']} (SYNTHETIC)\n"
        f"Student: {ident['name']}\n"
        f"School: {ident['school']}, {ident['city']}\n"
        f"Roll no: {rng.randint(1000000, 9999999)}\n"
        f"Year of passing: {rng.choice([2016, 2017, 2018, 2019, 2020, 2021]) if standard == '10' else rng.choice([2018, 2019, 2020, 2021, 2022, 2023])}\n"
        f"{lines}\n"
        f"Result: PASS  Division: {rng.choice(['I', 'II'])}\n"
        f"{extra}"
        f"This certifies Class {standard} performance only.\n"
    )


def body_board_resolution(rng: random.Random, ident: dict, hard: bool) -> str:
    return (
        "COMPANY CONSTITUTION & BOARD RESOLUTION (SYNTHETIC)\n"
        f"Company: {ident['company']}\n"
        f"CIN (synthetic): U{rng.randint(10000, 99999)}{ident['city'][:3].upper()}{rng.randint(2012, 2022)}PTC{rng.randint(100000, 999999)}\n"
        f"Extract of minutes dated {date(rng, 2024)}\n"
        f"RESOLVED THAT {ident['name']}, Director, is authorised to borrow from {ident['bank']} "
        f"upto INR {amount(rng, 1_000_000, 50_000_000)} and execute charge documents.\n"
        "Quorum present. Resolution passed unanimously.\n"
        "Certified true copy of AoA/MoA clauses on borrowing powers attached.\n"
        + ("Security may include property mortgage as collateral.\n" if hard else "")
    )


def body_financials(rng: random.Random, ident: dict, hard: bool) -> str:
    return (
        "AUDITED FINANCIAL STATEMENTS (SYNTHETIC)\n"
        f"{ident['company']}\n"
        f"Year ended 31 March {rng.choice([2023, 2024, 2025])}\n"
        "Independent auditor's report: unmodified opinion.\n"
        f"Revenue from operations: INR {amount(rng, 8_000_000, 240_000_000)}\n"
        f"Profit before tax: INR {amount(rng, 200_000, 18_000_000)}\n"
        f"Total assets: INR {amount(rng, 5_000_000, 180_000_000)}\n"
        "Balance sheet, P&L, cash flow, and notes form an integral part of these statements.\n"
        f"Director: {ident['name']}\n"
        + (f"GSTIN {ident['gstin']} disclosed in notes to accounts.\n" if hard else "")
    )


BODY_FN = {
    "Bank Statement": lambda rng, ident, hard: body_bank_statement(rng, ident, hard),
    "Legal Contract": lambda rng, ident, hard: body_contract(rng, ident, hard),
    "Proof of Identity": lambda rng, ident, hard: body_identity(rng, ident, hard),
    "Proof of Address": lambda rng, ident, hard: body_address(rng, ident, hard),
    "Loan application form": lambda rng, ident, hard: body_loan(rng, ident, hard),
    "Salary Slip": lambda rng, ident, hard: body_salary(rng, ident, hard),
    "ITR": lambda rng, ident, hard: body_itr(rng, ident, hard),
    "Property & Collateral Documents": lambda rng, ident, hard: body_property(rng, ident, hard),
    "Credit Card Application": lambda rng, ident, hard: body_cc_application(rng, ident, hard),
    "Credit Card statement": lambda rng, ident, hard: body_cc_statement(rng, ident, hard),
    "Class 10 Marksheet": lambda rng, ident, hard: body_marksheet(rng, ident, "10", hard),
    "Class 12 Marksheet": lambda rng, ident, hard: body_marksheet(rng, ident, "12", hard),
    "Company Constitution & Board Resolution": lambda rng, ident, hard: body_board_resolution(rng, ident, hard),
    "GST Registration / GST Return": lambda rng, ident, hard: body_gst(rng, ident, hard),
    "Audited Financial Statements": lambda rng, ident, hard: body_financials(rng, ident, hard),
}


def make_ident(rng: random.Random) -> dict:
    p = pan(rng)
    return {
        "name": person(rng),
        "bank": rng.choice(BANKS),
        "city": rng.choice(CITIES),
        "address": address(rng),
        "company": rng.choice(COMPANIES),
        "pan": p,
        "gstin": gstin(rng, p),
        "aadhaar": aadhaar(rng),
        "ifsc": ifsc(rng),
        "account": acct(rng),
        "loan_id": f"LN{rng.randint(100000, 999999)}",
        "board": rng.choice(BOARDS),
        "school": rng.choice(SCHOOLS),
    }


def misleading_filename(rng: random.Random, true_label: str, fmt: str, doc_id: str) -> str:
    other = CONFUSABLE.get(true_label, rng.choice([c for c in CLASSES if c != true_label]))
    return f"{slug(other)}_{doc_id[-4:]}.{fmt}"


def honest_filename(label: str, fmt: str, doc_id: str) -> str:
    return f"{slug(label)}_{doc_id}.{fmt}"


@dataclass
class Record:
    document_id: str
    class_label: str
    format: str
    content_core: str
    quality: str
    difficulty: str
    duplicate_group: str
    is_exact_duplicate: bool
    is_near_duplicate: bool
    has_misleading_filename: bool
    split: str = ""
    filename: str = ""
    content: str = ""
    size_bytes: int = 0
    checksum: str = ""


def take_format(counter: Counter, rng: random.Random, pair: bool) -> str:
    need = 2 if pair else 1
    candidates = [fmt for fmt, n in counter.items() if n >= need]
    if not candidates:
        raise RuntimeError(f"cannot allocate format pair={pair} from {dict(counter)}")
    weights = [counter[fmt] for fmt in candidates]
    fmt = rng.choices(candidates, weights=weights, k=1)[0]
    counter[fmt] -= need
    if counter[fmt] == 0:
        del counter[fmt]
    return fmt


def assign_splits(group_sizes: dict[str, int], rng: random.Random) -> dict[str, str]:
    pairs = [g for g, n in group_sizes.items() if n == 2]
    singles = [g for g, n in group_sizes.items() if n == 1]
    rng.shuffle(pairs)
    rng.shuffle(singles)
    n_pairs_val = rng.randint(2, 4)
    n_pairs_test = rng.randint(2, 4)
    n_sing_val = 9 - 2 * n_pairs_val
    n_sing_test = 9 - 2 * n_pairs_test
    assignment: dict[str, str] = {}
    i_p = 0
    i_s = 0
    for _ in range(n_pairs_val):
        assignment[pairs[i_p]] = "validation"
        i_p += 1
    for _ in range(n_sing_val):
        assignment[singles[i_s]] = "validation"
        i_s += 1
    for _ in range(n_pairs_test):
        assignment[pairs[i_p]] = "test"
        i_p += 1
    for _ in range(n_sing_test):
        assignment[singles[i_s]] = "test"
        i_s += 1
    for g in pairs[i_p:] + singles[i_s:]:
        assignment[g] = "train"
    return assignment


def build_class_records(
    label: str,
    class_index: int,
    rng: random.Random,
    format_counter: Counter,
) -> list[Record]:
    qualities_cycle = (
        ["clean"] * 24
        + ["ocr"] * 8
        + ["noisy"] * 6
        + ["sparse"] * 6
    )
    rng.shuffle(qualities_cycle)
    hard_idx = set(rng.sample(range(N_UNIQUE_PER_CLASS), 12))
    misleading_idx = set(rng.sample(range(N_UNIQUE_PER_CLASS), 8))

    pair_formats = [take_format(format_counter, rng, pair=True) for _ in range(16)]
    single_formats = [take_format(format_counter, rng, pair=False) for _ in range(28)]
    base_formats = pair_formats + single_formats

    bases: list[Record] = []
    for i in range(N_UNIQUE_PER_CLASS):
        ident = make_ident(rng)
        hard = i in hard_idx
        quality = qualities_cycle[i]
        core = BODY_FN[label](rng, ident, hard)
        core = apply_quality(core, quality, rng)
        if hard:
            difficulty = "hard"
        elif quality in {"noisy", "sparse", "ocr"}:
            difficulty = "medium"
        else:
            difficulty = "easy"
        doc_num = class_index * N_PER_CLASS + i + 1
        document_id = f"DOC-{doc_num:04d}"
        group = f"GRP-{class_index:02d}-{i:02d}"
        bases.append(
            Record(
                document_id=document_id,
                class_label=label,
                format=base_formats[i],
                content_core=core,
                quality=quality,
                difficulty=difficulty,
                duplicate_group=group,
                is_exact_duplicate=False,
                is_near_duplicate=False,
                has_misleading_filename=i in misleading_idx,
            )
        )

    records = list(bases)
    next_id = class_index * N_PER_CLASS + N_UNIQUE_PER_CLASS + 1

    for j in range(N_EXACT_DUP_PER_CLASS):
        src = bases[j]
        document_id = f"DOC-{next_id:04d}"
        next_id += 1
        records.append(
            Record(
                document_id=document_id,
                class_label=label,
                format=src.format,
                content_core=src.content_core,
                quality=src.quality,
                difficulty=src.difficulty,
                duplicate_group=src.duplicate_group,
                is_exact_duplicate=True,
                is_near_duplicate=False,
                has_misleading_filename=False,
            )
        )

    for j in range(N_NEAR_DUP_PER_CLASS):
        src = bases[N_EXACT_DUP_PER_CLASS + j]
        document_id = f"DOC-{next_id:04d}"
        next_id += 1
        near_core = near_mutate(src.content_core, rng)
        records.append(
            Record(
                document_id=document_id,
                class_label=label,
                format=src.format,
                content_core=near_core,
                quality="ocr",
                difficulty="medium" if src.difficulty != "hard" else "hard",
                duplicate_group=src.duplicate_group,
                is_exact_duplicate=False,
                is_near_duplicate=True,
                has_misleading_filename=False,
            )
        )

    group_sizes: dict[str, int] = defaultdict(int)
    for rec in records:
        group_sizes[rec.duplicate_group] += 1
    splits = assign_splits(dict(group_sizes), rng)
    for rec in records:
        rec.split = splits[rec.duplicate_group]
        rec.filename = (
            misleading_filename(rng, rec.class_label, rec.format, rec.document_id)
            if rec.has_misleading_filename
            else honest_filename(rec.class_label, rec.format, rec.document_id)
        )
        rec.content = wrap_format(rec.format, rec.class_label, rec.content_core)
    return records


def write_data_readme() -> None:
    text = """# Synthetic document classification benchmark

This folder holds a **locally generated** 900-document benchmark for evaluating
document-type classification. It is not a dump of real customer files.

## How it was generated

- Generator: `python -m src.data.generate_benchmark`
- Fixed seed: `42` (deterministic filenames, splits, and text)
- 15 classes × 60 documents = 900 rows in `manifest.csv`
- File bodies live under `raw/` as **text-bearing stand-ins** named with the
  intended extension (`pdf`, `jpg`, `tiff`, `docx`, `png`, `eml`, `msg`,
  `xlsx`, `txt`, `pptx`, plus a few `rtf` as “other”)
- Intended format mix approximately matches the assignment (PDF ~45%, JPG ~18%,
  TIFF ~10%, DOCX ~8%, PNG ~6%, EML/MSG ~5%, XLSX ~4%, TXT ~2%, PPTX ~1.5%,
  other ~0.5%)
- Each class has 44 unique bases, 8 exact byte-identical duplicates, and 8
  near-duplicates (light OCR-like mutation)
- Train / validation / test are assigned **by duplicate group** so exact
  duplicates (and their paired near-duplicates) never cross splits
- Hard cases overlap vocabulary with a known confusable class:
  identity vs address, bank vs card statement, class 10 vs 12 marksheet,
  ITR vs GST, legal contract vs property/collateral
- Some filenames are deliberately labelled as the confusable class

## Manifest fields

`document_id`, `filename`, `format`, `class_label`, `content`, `size_bytes`,
`checksum`, `duplicate_group`, `difficulty`, `split`, plus generation flags
(`text_quality`, `is_exact_duplicate`, `is_near_duplicate`,
`has_misleading_filename`).

`content` is the full stand-in file body. `checksum` is SHA-256 of those bytes.

## Limitations of synthetic data

- Files are **not** valid PDFs, images, Office packages, or mailboxes. They
  only carry representative text so a later extractor can be wired up without
  shipping copyrighted or private documents.
- Visual layout, stamps, handwriting, seals, and real OCR error distributions
  are not reproduced; “OCR” here is character substitution.
- Identifiers (PAN, GSTIN, Aadhaar, account numbers) are fake and must not be
  treated as real PII.
- Class priors are uniform (60 each), which is cleaner than production traffic.
- Language is English-only boilerplate; real packets are often mixed-language
  scans with attachments.

Use this set to test routing, leakage, and difficulty tags—not as a substitute
for a labelled production sample.
"""
    DATA_README_PATH.write_text(text, encoding="utf-8")


def generate() -> list[Record]:
    rng = random.Random(SEED)
    if DATA_DIR.exists():
        for child in DATA_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    format_counter: Counter = Counter(FORMAT_COUNTS)
    records: list[Record] = []
    for class_index, label in enumerate(CLASSES):
        class_rng = random.Random(rng.randint(0, 10**9))
        records.extend(build_class_records(label, class_index, class_rng, format_counter))
    if sum(format_counter.values()) != 0:
        raise RuntimeError(f"unassigned formats: {dict(format_counter)}")

    records.sort(key=lambda r: r.document_id)
    fieldnames = [
        "document_id",
        "filename",
        "format",
        "class_label",
        "content",
        "size_bytes",
        "checksum",
        "duplicate_group",
        "difficulty",
        "split",
        "text_quality",
        "is_exact_duplicate",
        "is_near_duplicate",
        "has_misleading_filename",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            path = RAW_DIR / slug(rec.class_label) / rec.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            data = rec.content.encode("utf-8")
            path.write_bytes(data)
            rec.size_bytes = len(data)
            rec.checksum = hashlib.sha256(data).hexdigest()
            writer.writerow(
                {
                    "document_id": rec.document_id,
                    "filename": rec.filename,
                    "format": rec.format,
                    "class_label": rec.class_label,
                    "content": rec.content,
                    "size_bytes": rec.size_bytes,
                    "checksum": rec.checksum,
                    "duplicate_group": rec.duplicate_group,
                    "difficulty": rec.difficulty,
                    "split": rec.split,
                    "text_quality": rec.quality,
                    "is_exact_duplicate": rec.is_exact_duplicate,
                    "is_near_duplicate": rec.is_near_duplicate,
                    "has_misleading_filename": rec.has_misleading_filename,
                }
            )
    write_data_readme()
    return records


def verify(records: list[Record]) -> None:
    files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
    n_files = len(files)
    labels = Counter(r.class_label for r in records)
    splits = Counter(r.split for r in records)
    difficulty = Counter(r.difficulty for r in records)
    n_exact = sum(r.is_exact_duplicate for r in records)
    n_near = sum(r.is_near_duplicate for r in records)
    n_hard = sum(r.difficulty == "hard" for r in records)
    n_mis = sum(r.has_misleading_filename for r in records)

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        n_manifest = sum(1 for _ in csv.DictReader(fh))

    # Exact duplicates must not leak across splits.
    groups: dict[str, set[str]] = defaultdict(set)
    exact_groups = {r.duplicate_group for r in records if r.is_exact_duplicate}
    for r in records:
        if r.duplicate_group in exact_groups:
            groups[r.duplicate_group].add(r.split)
    leaked = {g: s for g, s in groups.items() if len(s) > 1}

    print("=== Benchmark verification ===")
    print(f"1. documents on disk:     {n_files} (expected 900)")
    print(f"2. classes:               {len(labels)} (expected 15)")
    print("3. documents per class:")
    for label in CLASSES:
        print(f"    {label:42s} {labels[label]:3d}")
    print(f"4. manifest rows:         {n_manifest} (expected 900)")
    print("5. splits:")
    for name in ("train", "validation", "test"):
        print(f"    {name:12s} {splits[name]:3d}")
    print("6. difficulty / duplicates:")
    print(f"    easy={difficulty['easy']}  medium={difficulty['medium']}  hard={n_hard}")
    print(f"    exact duplicates={n_exact}  near-duplicates={n_near}")
    print(f"    misleading filenames={n_mis}")
    print(f"    exact-dup groups spanning splits: {len(leaked)}")

    assert n_files == 900, n_files
    assert len(records) == 900
    assert n_manifest == 900
    assert len(labels) == 15
    assert all(labels[c] == 60 for c in CLASSES)
    assert leaked == {}
    print("All checks passed.")


def main() -> None:
    records = generate()
    verify(records)


if __name__ == "__main__":
    main()
