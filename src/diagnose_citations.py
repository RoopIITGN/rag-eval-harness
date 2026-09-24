"""
Diagnose citation verification failures.

The evaluation records how many citations failed but not which ones, so a
refusal rate can't be read as either "the generator fabricates quotes" or "the
comparison is too strict". Those need opposite fixes, so the distinction has to
be made from the data.

This re-runs only the queries that failed, and prints each rejected quote beside
the chunk it cited, with a guess at the cause: a character the normaliser
doesn't fold (curly quotes, dashes, currency symbols), a quote spanning two
chunks, or text that simply isn't there.

Usage
  PYTHONPATH=src python src/diagnose_citations.py
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

import generate
import retrieve

ROOT = Path(__file__).resolve().parents[1]
GENS = ROOT / "reports/generations.jsonl"


def fold(s: str) -> str:
    """Aggressive normalisation, used only to explain a failure -- not by the gate."""
    s = unicodedata.normalize("NFKC", s)
    for a, b in [("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-"), ("\u00a0", " ")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    recs = [json.loads(l) for l in GENS.read_text(encoding="utf-8").splitlines() if l.strip()]
    failing = [r for r in recs if r["bad_citations"]]
    print(f"{len(failing)} queries had a citation rejected; re-running them\n")

    causes = {"unicode only": 0, "spans two chunks": 0, "near miss": 0, "not present": 0}
    for n, rec in enumerate(failing, start=1):
        r = generate.answer(rec["query"])
        if not r.bad_citations:
            print(f"[{n}] q{rec['query_id']}: passed on re-run (generator is non-deterministic)\n")
            continue

        chunks = retrieve.retrieve(rec["query"], generate.CONFIG, generate.MODE,
                                   top=generate.TOP_K)[:generate.TOP_K]
        by_id = {f"chunk_{i}": c for i, c in enumerate(chunks, start=1)}
        all_text = " ".join(c["content"] for c in chunks)

        for bad in r.bad_citations:
            quote = bad.get("source_quote", "")
            cited = by_id.get(bad.get("chunk_id", ""))
            if cited is None:
                cause = "not present"
            elif fold(quote) in fold(cited["content"]):
                cause = "unicode only"
            elif fold(quote) in fold(all_text):
                cause = "spans two chunks"
            else:
                m = difflib.SequenceMatcher(None, fold(quote), fold(cited["content"]))
                best = max((b.size for b in m.get_matching_blocks()), default=0)
                cause = "near miss" if best > 0.7 * len(fold(quote)) else "not present"
            causes[cause] += 1
            print(f"[{n}] q{rec['query_id']} {bad.get('chunk_id')} — {cause}")
            print(f"    quoted : {quote[:110]!r}")
            if cited and cause != "unicode only":
                snippet = cited["content"][:110].replace("\n", " ")
                print(f"    chunk  : {snippet!r}")
            print()

    print("=" * 60)
    for k, v in causes.items():
        print(f"  {k:18s} {v}")
    print()
    print("unicode only     -> the gate is too strict; fold these characters in _norm")
    print("spans two chunks -> the model stitched two chunks; tighten the prompt")
    print("near miss        -> paraphrased inside a quote; the gate is working")
    print("not present      -> fabricated; the gate is working and this is the finding")


if __name__ == "__main__":
    main()
