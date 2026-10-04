# Synthetic document classification benchmark

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
