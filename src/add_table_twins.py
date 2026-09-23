"""
Add the missing twin span for answers that live in a table.

extract.py writes every table twice: once as the page's raw flattened text
("Cash and FDR Addition 7:30 PM 9:00 PM") and once as labelled rows ("Type:
Cash and FDR Addition; Current cut off time: 7:30 PM; New cut off time:
9:00 PM"). Both contain the answer. A gold label naming only one scores a
retriever that returns the other as a miss.

That isn't random noise. The two forms sit ~150 tokens apart, so they usually
share a 512-token chunk but often fall in different 256-token chunks -- the
false misses land disproportionately on the smallest configuration and bias
the chunking comparison.

For each single-span record, this finds any other line in the same document
carrying the same values, and adds it as a second span with require: "any".

Scope, deliberately narrow:
  - windows of up to four lines; beyond that nothing more is found, and the
    risk of a coincidental match rises.
  - single-span records only. Multi-span records (multi-hop, widened) already
    have their own require semantics; a flat span list can't express "either
    form of span k", so they're reported and left alone.
  - rejected records are skipped.
  - matches need at least MIN_TOKENS value tokens, so a bare "8:00 PM" can't
    pair with every other 8:00 PM in the document.

Records are otherwise untouched: the review decision is not altered, and each
change is marked with twin_span_added: true so it's auditable.

Usage
  PYTHONPATH=src python src/add_table_twins.py            # dry run, report only
  PYTHONPATH=src python src/add_table_twins.py --apply    # write changes
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/goldset.jsonl"
DOCS = ROOT / "data/documents"
MIN_TOKENS = 4

TOKEN = re.compile(r"[A-Za-z0-9]+(?::[0-9]+)?")      # keeps 7:30 as one token


def is_labelled(text: str) -> bool:
    """A row written by extract.py's table_to_lines: 'Header: value; Header: value'."""
    return ";" in text and re.search(r"(^|; ?)[^;]*: ", text.strip()) is not None


def canon(text: str) -> list[str]:
    """Value tokens only -- column labels stripped from labelled rows, so the two
    forms of one table row reduce to the same token list."""
    if is_labelled(text):
        cells = [c.split(": ", 1)[1] if ": " in c else c for c in text.strip().split(";")]
        text = " ".join(cells)
    return [t.lower() for t in TOKEN.findall(text)]


def lines_with_offsets(text: str):
    pos = 0
    for line in text.split("\n"):
        yield pos, pos + len(line), line
        pos += len(line) + 1


def find_twin(doc: str, span: dict) -> dict | None:
    """Look for a one- or two-line window carrying the same values.

    Up to four lines, because a labelled row wraps once per column header that
    contains a line break. The ETF lists' "Minimum Quantity Required\n(in
    multiple thereon)" costs one; the dividend strike tables have three --
    "Instrument\nType", "OLD STRIKE\nPRICE", "REVISED STRIKE\nPRICE" -- so
    their labelled rows occupy four lines.

    The limit only ever applied to the candidate window. A gold span itself can
    span any number of lines, which is why labelled-span -> body-row twins were
    found all along and the reverse direction was not.
    """
    target = canon(doc[span["start"]:span["end"]])
    if len(target) < MIN_TOKENS:
        return None
    lines = list(lines_with_offsets(doc))
    for width in (1, 2, 3, 4):
        for k in range(len(lines) - width + 1):
            s, e = lines[k][0], lines[k + width - 1][1]
            if not (e <= span["start"] or s >= span["end"]):   # overlaps the span itself
                continue
            window = doc[s:e]
            if canon(window) == target:
                lead = len(window) - len(window.lstrip())       # drop table indent
                return {"doc_id": span["doc_id"], "start": s + lead, "end": e}
    return None


def main() -> None:
    apply = "--apply" in sys.argv
    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs: dict[str, str] = {}
    added, skipped_multi = [], []

    for r in rows:
        if r.get("review", {}).get("decision") == "reject":
            continue
        if r.get("twin_span_added"):
            continue                                          # idempotent
        if len(r["gold_spans"]) != 1:
            skipped_multi.append(r["query_id"])
            continue
        span = r["gold_spans"][0]
        doc = docs.setdefault(span["doc_id"],
                              (DOCS / f"{span['doc_id']}.txt").read_text(encoding="utf-8"))
        twin = find_twin(doc, span)
        if twin:
            added.append((r, span, twin, doc))

    for r, span, twin, doc in added:
        print(f"query {r['query_id']}: {r['query'][:70]}")
        print(f"   has  : {doc[span['start']:span['end']]!r}")
        print(f"   twin : {doc[twin['start']:twin['end']]!r}\n")

    print(f"{len(added)} records gain a twin span")
    if skipped_multi:
        print(f"{len(skipped_multi)} multi-span records left alone "
              f"(check by hand if they point at table rows): {skipped_multi}")

    if not apply:
        print("\ndry run -- nothing written. Re-run with --apply to save.")
        return

    for r, span, twin, _ in added:
        r["gold_spans"].append(twin)
        r["require"] = "any"
        r["twin_span_added"] = True

    tmp = GOLD.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(GOLD)
    print(f"\nwritten -> {GOLD}")


if __name__ == "__main__":
    main()
