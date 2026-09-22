"""
Validate the corpus before anything downstream uses it.

Every document's identity here comes from its filename, and filenames are typed
by people. One circular in this corpus was saved under the serial number it
*cited* rather than its own -- SEBI-10557.txt held circular 18038 -- and nothing
in the pipeline noticed. The citation graph eventually exposed it: two
circulars citing each other with the same date is impossible in real
regulatory data. This script makes that failure loud and early instead.

Three checks:

  Identity. The serial in the filename must appear in the document's own
  header -- the text before its subject line. A serial that only appears in
  the subject or body is a citation, not an identity.

  Exact duplicates. Two files with the same normalised content are always a
  bug.

  Near duplicates. Files sharing most of their five-word shingles. Usually a
  re-issued or corrected circular saved twice; occasionally a legitimately
  similar pair, like successive versions of a list. Reported, not failed.

Exits non-zero on any identity or exact-duplicate failure, so a CI step or a
Makefile target can refuse to continue.

Usage
  PYTHONPATH=src python src/validate_corpus.py
"""

from __future__ import annotations

import hashlib
import re
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data/documents"
NEAR_DUP = 0.90

SUBJECT = re.compile(r"^\s*Sub(?:ject)?\s*:", re.M)


def header(text: str) -> str:
    """Text before the subject line -- where a circular states its own ID."""
    m = SUBJECT.search(text)
    return text[:m.start()] if m else text[:600]


def shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def main() -> int:
    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted(DOCS.glob("*.txt"))}
    failures = 0

    # --- identity ------------------------------------------------------------
    for d, t in texts.items():
        serial = d.rsplit("-", 1)[-1]
        if not re.search(rf"(?<![0-9]){re.escape(serial)}(?![0-9])", header(t)):
            failures += 1
            first = " ".join(header(t).split())[:140]
            print(f"IDENTITY  {d}: serial {serial} is not in this document's own header")
            print(f"          header reads: {first!r}")

    # --- exact duplicates ----------------------------------------------------
    seen: dict[str, str] = {}
    for d, t in texts.items():
        h = hashlib.sha256(" ".join(t.split()).encode()).hexdigest()
        if h in seen:
            failures += 1
            print(f"DUPLICATE {d} has identical content to {seen[h]}")
        else:
            seen[h] = d

    # --- near duplicates -----------------------------------------------------
    sh = {d: shingles(t) for d, t in texts.items()}
    near = []
    for a, b in combinations(texts, 2):
        if not sh[a] or not sh[b]:
            continue
        j = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
        if j >= NEAR_DUP:
            near.append((j, a, b))
    for j, a, b in sorted(near, reverse=True):
        print(f"SIMILAR   {a} ~ {b}  ({j:.0%} shared) -- check they are different circulars")

    print(f"\n{len(texts)} documents checked -- "
          f"{'PASS' if not failures else f'{failures} FAILURE(S)'}"
          f"{f', {len(near)} similar pair(s) to eyeball' if near else ''}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
