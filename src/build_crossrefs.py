"""
Find which circulars in the corpus relate to each other, and which is newest.

Single-document eval-set generation can't see that a later circular amended or
replaced an earlier one. A query generated from the earlier circular then
carries a stale answer as ground truth -- and a retriever that correctly
returns the newer circular gets scored as a miss. This finds those
relationships so the review loop can flag affected queries.

Two detectors, because each misses what the other catches:

  Citations. Document A cites B if B's serial number appears in A's text. This
  catches SEBI extensions and amendments, which name the circular they modify.
  Serials shorter than four digits are skipped -- '41' or '122' match
  regulation numbers, amounts and dates too often -- and listed for manual
  checking.

  Subject lines. Documents sharing a subject line form a family. This catches
  chains where the linking circular isn't in the corpus: in the collateral
  timings chain, 72357 and 75321 each cite a circular we don't have, so no
  citation connects them, but all three share "Change in Collateral Timings".
  Recurring NSE notices are the main case; SEBI subject lines are usually
  unique to one circular.

Each document's date is read from its header, so the review screen can say
which member of a family is current rather than leaving it to be worked out.

A citation means "mentions", not "amends" -- many are benign ("refer to
circular X for connectivity details"). Whether a link is a supersession is a
judgment made during review; the context is printed to make it quick.

Output: reports/cross_references.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data/documents"
OUT = ROOT / "reports/cross_references.json"
MIN_SERIAL_LEN = 4

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = re.compile(rf"({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})")
SUBJECT = re.compile(r"^\s*Sub(?:ject)?\s*:\s*(.+)$", re.M)


def doc_date(text: str) -> str | None:
    """First date in the header -- the circular's own issue date.

    Searched only in the opening lines: dates further down are usually the
    dates of circulars being cited, not this one.
    """
    m = DATE.search(text[:800])
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date().isoformat()


def doc_subject(text: str) -> str | None:
    m = SUBJECT.search(text[:1500])
    return re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".").lower() if m else None


def main() -> None:
    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted(DOCS.glob("*.txt"))}
    serial = {d: d.rsplit("-", 1)[-1] for d in texts}
    date = {d: doc_date(t) for d, t in texts.items()}

    # --- citations -----------------------------------------------------------
    citations = []
    for src, text in texts.items():
        for dst, s in serial.items():
            if dst == src or len(s) < MIN_SERIAL_LEN:
                continue
            m = re.search(rf"(?<![0-9]){re.escape(s)}(?![0-9])", text)
            if m:
                ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 110):m.end() + 110])
                citations.append({"from": src, "to": dst,
                                  "from_date": date[src], "to_date": date[dst],
                                  "context": ctx})

    # --- subject families ----------------------------------------------------
    groups = defaultdict(list)
    for d, t in texts.items():
        subj = doc_subject(t)
        if subj:
            groups[subj].append(d)
    families = []
    for subj, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda d: date[d] or "")        # oldest first
        families.append({"subject": subj,
                         "members": [{"doc_id": d, "date": date[d]} for d in members],
                         "newest": members[-1]})

    short = sorted(d for d, s in serial.items() if len(s) < MIN_SERIAL_LEN)
    undated = sorted(d for d, v in date.items() if v is None)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"citations": citations, "families": families,
                               "unchecked_short_serials": short, "undated": undated},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    # --- report --------------------------------------------------------------
    print(f"CITATIONS -- {len(citations)}\n")
    for c in sorted(citations, key=lambda c: (c["from_date"] or "", c["from"])):
        print(f"  {c['from']} ({c['from_date']})  cites  {c['to']} ({c['to_date']})")
        print(f"      ...{c['context']}...\n")

    print(f"SUBJECT FAMILIES -- {len(families)}\n")
    for f in families:
        print(f"  \"{f['subject']}\"")
        for m in f["members"]:
            tag = "   <- newest" if m["doc_id"] == f["newest"] else ""
            print(f"      {m['date']}  {m['doc_id']}{tag}")
        print()

    if short:
        print(f"short serials, citations not checked (verify by hand): {', '.join(short)}")
    if undated:
        print(f"no date found in header (family order may be wrong): {', '.join(undated)}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
