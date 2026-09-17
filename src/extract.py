"""
PDF -> frozen plain text. Run once. Never run again after labelling starts.

WHY "FROZEN"
------------
Every gold span in the eval set is (doc_id, char_start, char_end) relative to
these exact text files. Re-extracting with a different parser, or a different
version of the same parser, shifts whitespace by a character or two and
silently invalidates every label in the goldset. Nothing errors. Your recall
numbers just quietly become wrong.

So: commit data/documents/*.txt to git, and treat them as data, not as
build artefacts.

TABLES
------
NSE circulars are substantially tabular. pdfplumber's extract_table() gets
the cell structure; plain extract_text() flattens it into unlabelled rows.
This script uses extract_table() and re-emits each row with its header
prefixed, so a chunk containing "Normal Market open time | 11:15" still
carries the column names and the table caption.

Without that, a retrieved row reads as "Normal Market open time 11:15" with
no indication of what table or date it belongs to -- effectively unretrievable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber

RAW = Path("data/raw_pdfs")
OUT = Path("data/documents")

# Lines matching these are page furniture, not content.
NOISE = [
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.I),
    re.compile(r"^\s*Continuation Sheet\s*$", re.I),
    re.compile(r"^\s*$"),
]


def clean_lines(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        if any(p.match(line) for p in NOISE):
            continue
        out.append(line.rstrip())
    return out


def table_to_lines(table: list[list[str | None]], caption: str | None) -> list[str]:
    """Flatten a table so every row carries its header.

    A row stripped of its header is nearly unretrievable, because the query
    terms live in the header, not the row.
    """
    if not table or len(table) < 2:
        return []
    header = [(c or "").strip() for c in table[0]]
    lines = []
    if caption:
        lines.append(f"[Table: {caption}]")
    lines.append(" | ".join(header))
    for row in table[1:]:
        cells = [(c or "").strip() for c in row]
        # Prefix each cell with its column name so the pairing survives chunking
        paired = [f"{h}: {c}" for h, c in zip(header, cells) if c]
        if paired:
            lines.append("  " + "; ".join(paired))
    return lines


def extract(pdf_path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            text = page.extract_text() or ""

            parts.extend(clean_lines(text))

            for t in tables:
                parts.extend(table_to_lines(t, caption=None))
                parts.append("")

    # Collapse runs of blank lines; keep paragraph breaks for the chunker
    joined = "\n".join(parts)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip() + "\n"


def quality_report(doc_id: str, text: str) -> list[str]:
    """Flag extractions that need a manual look."""
    warnings = []
    if len(text) < 500:
        warnings.append("very short -- scanned PDF? needs OCR?")
    if text.count("\ufffd") > 0:
        warnings.append("contains replacement characters -- encoding problem")
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(1, len(text)) < 0.4:
        warnings.append("low alphabetic ratio -- extraction may be garbled")
    if not re.search(r"[A-Z]{2,}[/-]", text[:2000]):
        warnings.append("no circular identifier found in the first 2000 chars")
    return warnings


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {RAW}")

    flagged = []
    for p in pdfs:
        text = extract(p)
        dest = OUT / f"{p.stem}.txt"
        dest.write_text(text, encoding="utf-8")

        warns = quality_report(p.stem, text)
        status = "  ".join(warns) if warns else "ok"
        print(f"{p.stem:32s} {len(text):>7,d} chars   {status}")
        if warns:
            flagged.append(p.stem)

    print(f"\n{len(pdfs)} extracted -> {OUT}")
    if flagged:
        print(f"\nOPEN THESE AND READ THEM before going further:")
        for f in flagged:
            print(f"  {OUT / (f + '.txt')}")


if __name__ == "__main__":
    main()
