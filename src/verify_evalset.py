"""
Human review loop for the generated eval set.

Nothing in goldset.jsonl counts until a person has confirmed it. This walks
through every unreviewed record, shows the gold span inside its surrounding
text, and records a decision. Progress is saved after every decision, so the
review can run across several sittings -- quit any time and re-run to resume.

Keys
  a  accept      the highlighted span answers the query
  e  edit        span is right, query is vague -- rewrite the query
  w  widen       another document answers it too -- add a second gold span
  p  re-span     query is good, span is wrong -- point it at the right passage
  n  new         reject this one and write your own query + span in its place
  r  reject      boilerplate, wrong span, ambiguous, or trivial
  g  go to       jump to any query id, then come back to where you were
  s  skip        leave for later
  q  quit        save and exit

Rejected records stay in the file with verified: false and a reason, so the
rejection rate is reportable instead of silently discarded. Replacing a query
('n') never overwrites: the original is rejected as 'replaced' and a new record
is appended with query_source: human_authored, each pointing at the other.

Re-reviewing a record that was already decided never loses the earlier
decision: it moves into review_history, and any prior changes carry forward.
Records you don't touch are written back exactly as they were.

Usage
  PYTHONPATH=src python src/verify_evalset.py                  # unreviewed, in order
  PYTHONPATH=src python src/verify_evalset.py --ids 17,18,1001 # just these, reviewed or not
  PYTHONPATH=src python src/verify_evalset.py --from 1001      # unreviewed, from this id on
  PYTHONPATH=src python src/verify_evalset.py --list           # id, status, query
  PYTHONPATH=src python src/verify_evalset.py --stats          # progress only
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/goldset.jsonl"
DOCS = ROOT / "data/documents"
CONTEXT = 300          # characters of surrounding text shown on each side
XREFS = ROOT / "reports/cross_references.json"

_tty = sys.stdout.isatty()
BOLD, YEL, DIM, CYAN, RED, RST = (
    ("\033[1m", "\033[93m", "\033[2m", "\033[96m", "\033[91m", "\033[0m")
    if _tty else ("", "", "", "", "", ""))

REJECT_REASONS = {
    "1": "boilerplate",                  # contact blocks, signatures, standard text
    "2": "ambiguous_across_documents",   # several circulars answer it equally
    "3": "wrong_span",                   # quote doesn't actually answer the query
    "4": "trivial_or_unanswerable",
    "5": "other",
}

_docs: dict[str, str] = {}
_xref: dict | None = None


def xref() -> dict:
    """Output of build_crossrefs.py, if it has been run."""
    global _xref
    if _xref is None:
        _xref = json.loads(XREFS.read_text(encoding="utf-8")) if XREFS.exists() else {}
    return _xref


def show_xrefs(r: dict) -> None:
    """Warn when the gold document belongs to a family of related circulars.

    A query generated from a circular that a later one amended or replaced
    carries a stale answer -- and a retriever that correctly returns the newer
    circular would be scored as a miss. Unless the query names the circular's
    own number, in which case the older answer is exactly right.
    """
    gold = {s["doc_id"] for s in r["gold_spans"]}
    data = xref()
    cites = [c for c in data.get("citations", data.get("edges", []))
             if c["from"] in gold or c["to"] in gold]
    fams = [f for f in data.get("families", [])
            if gold & {m["doc_id"] for m in f["members"]}]
    if not (cites or fams):
        return

    print(f"{BOLD}{RED}! related circulars -- check whether the answer is still current{RST}")
    for f in fams:
        members = {m["doc_id"] for m in f["members"]}
        print(f"  same subject: \"{f['subject']}\"")
        for m in f["members"]:
            tags = [t for t, on in (("gold", m["doc_id"] in gold),
                                    ("newest", m["doc_id"] == f["newest"])) if on]
            print(f"    {m['date']}  {m['doc_id']}" + (f"   <- {', '.join(tags)}" if tags else ""))
        if (gold & members) - {f["newest"]}:
            print(f"  {BOLD}{RED}gold is in an older circular than {f['newest']}{RST}"
                  f"{DIM} -- fine only if the query names that older circular's number{RST}")
    for c in cites:
        arrow = (f"cites {c['to']} ({c.get('to_date')})" if c["from"] in gold
                 else f"cited by {c['from']} ({c.get('from_date')})")
        print(f"  {arrow}")
        print(f"    {DIM}...{c['context']}...{RST}")
    print()


def doc(doc_id: str) -> str:
    if doc_id not in _docs:
        _docs[doc_id] = (DOCS / f"{doc_id}.txt").read_text(encoding="utf-8")
    return _docs[doc_id]


def load() -> list[dict]:
    return [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(rows: list[dict]) -> None:
    """Atomic write: a crash mid-save can't leave a half-written goldset."""
    tmp = GOLD.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(GOLD)


def locate(text: str, passage: str) -> tuple[int, int] | None:
    idx = text.find(passage)
    if idx != -1:
        return idx, idx + len(passage)
    cleaned = passage.strip()
    if not cleaned:
        return None
    pattern = re.sub(r"\\\s+", r"\\s+", re.escape(cleaned))
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def show_span(span: dict, label: str = "") -> None:
    text = doc(span["doc_id"])
    s, e = span["start"], span["end"]
    before, hit, after = text[max(0, s - CONTEXT):s], text[s:e], text[e:e + CONTEXT]
    print(f"{DIM}{label}{span['doc_id']}  chars {s}-{e}  ({e - s} chars){RST}")
    print(f"{DIM}...{before}{RST}{BOLD}{YEL}{hit}{RST}{DIM}{after}...{RST}\n")


def ask_span(default_doc: str | None = None) -> dict | None:
    """Prompt for a document and a verbatim passage; return a span or None."""
    hint = f" [{default_doc}]" if default_doc else ""
    doc_id = input(f"  doc_id{hint} > ").strip() or default_doc
    if not doc_id or not (DOCS / f"{doc_id}.txt").exists():
        print(f"  no document {doc_id}")
        return None
    passage = input("  paste the answering passage, verbatim > ").strip()
    span = locate(doc(doc_id), passage)
    if span is None:
        print("  passage not found verbatim in that document")
        return None
    return {"doc_id": doc_id, "start": span[0], "end": span[1]}


def next_id(rows: list[dict]) -> int:
    return max(r["query_id"] for r in rows) + 1


def stamp(r: dict, decision: str, reason: str | None = None) -> None:
    if "review" in r:                                  # re-review: keep the old one
        r.setdefault("review_history", []).append(r["review"])
    r["verified"] = decision == "accept"
    r["review"] = {
        "decision": decision,
        "reason": reason,
        "changes": r.pop("_changes", []),
        "at": datetime.now().isoformat(timespec="seconds"),
        "by": os.environ.get("REVIEWER", "roop"),
    }


def stats(rows: list[dict]) -> None:
    reviewed = [r for r in rows if "review" in r]
    accepted = [r for r in reviewed if r["review"]["decision"] == "accept"]
    rejected = [r for r in reviewed if r["review"]["decision"] == "reject"]
    changes = Counter(c for r in accepted for c in r["review"]["changes"])
    reasons = Counter(r["review"]["reason"] for r in rejected)
    types = Counter(r["query_type"] for r in accepted)

    print(f"\n{BOLD}review progress{RST}")
    print(f"  reviewed  {len(reviewed)}/{len(rows)}   remaining {len(rows) - len(reviewed)}")
    print(f"  accepted  {len(accepted)}   by type {dict(types)}")
    if changes:
        print(f"            of which {dict(changes)}")
    print(f"  rejected  {len(rejected)}   {dict(reasons) if reasons else ''}")
    src = Counter(r.get("query_source", "?") for r in accepted)
    print(f"  accepted by source  {dict(src)}")


def review(r: dict, progress: str, rows: list[dict]):
    """Show one record and loop until a terminal decision.

    Returns a key ('a', 'r', 's', 'q') or ('g', query_id) to jump elsewhere.
    """
    prev = r.get("review")
    if prev:
        # Carry earlier changes forward, so re-accepting an edited query still
        # records that it was edited.
        r.setdefault("_changes", list(prev.get("changes", [])))
    seeded = list(r.get("_changes", []))

    def leaving(key):
        if prev and r.get("_changes", []) != seeded:
            print(f"{BOLD}{RED}  you changed a record that was already decided, then left without"
                  f" deciding again -- the change is saved, the earlier decision still stands."
                  f" Use --ids {r['query_id']} to finish it.{RST}")
        return key

    while True:
        print("\n" + "=" * 78)
        print(f"{DIM}{progress}  query {r['query_id']}  ·  {r['query_type']}"
              f"  ·  require: {r.get('require', 'all')}{RST}")
        if prev:
            why = f" ({prev['reason']})" if prev.get("reason") else ""
            print(f"{BOLD}{RED}already reviewed: {prev['decision']}{why} on {prev['at']}"
                  f" -- a new decision replaces it; the old one is kept in history{RST}")
        if r.get("replaced_by"):
            print(f"{DIM}  (replaced by query {r['replaced_by']} -- accepting this again"
                  f" leaves both in the set){RST}")
        print(f"\n{BOLD}{CYAN}{r['query']}{RST}\n")
        for n, span in enumerate(r["gold_spans"], start=1):
            show_span(span, label=f"span {n}  " if len(r["gold_spans"]) > 1 else "")
        show_xrefs(r)

        choice = input(f"{BOLD}[a]ccept [e]dit [w]iden re-s[p]an [n]ew [r]eject "
                       f"[g]o to [s]kip [q]uit > {RST}").strip().lower()

        if choice == "a":
            return choice
        if choice in ("s", "q"):
            return leaving(choice)

        if choice == "g":
            target = input("  go to query id > ").strip()
            if target.isdigit():
                return leaving(("g", int(target)))
            print("  (not a number)")
            continue

        if choice == "r":
            print("  " + "   ".join(f"{k}={v}" for k, v in REJECT_REASONS.items()))
            reason = REJECT_REASONS.get(input("  reason > ").strip())
            if reason:
                stamp(r, "reject", reason)
                return "r"
            print("  (not a valid reason -- nothing saved)")

        elif choice == "e":
            new = input("  new query > ").strip()
            if new:
                r.setdefault("original_query", r["query"])
                r["query"] = new
                r.setdefault("_changes", []).append("edited_query")

        elif choice == "w":
            span = ask_span()
            if span:
                r["gold_spans"].append(span)
                r["require"] = "any"      # either document's passage answers it
                r.setdefault("_changes", []).append("added_span")

        elif choice == "p":
            span = ask_span(default_doc=r["gold_spans"][0]["doc_id"])
            if span:
                r.setdefault("original_spans", r["gold_spans"])
                r.setdefault("original_answer_passage", r.get("answer_passage"))
                r["gold_spans"] = [span]
                r["answer_passage"] = doc(span["doc_id"])[span["start"]:span["end"]]
                r["require"] = "all"
                r.pop("twin_span_added", None)   # new span may need its own twin
                r.setdefault("_changes", []).append("respanned")

        elif choice == "n":
            print("  Write the query the way an analyst would type it -- not by")
            print("  rephrasing the passage, which bakes lexical overlap into the label.")
            query = input("  new query > ").strip()
            if not query:
                continue
            qtype = {"1": "natural", "2": "keyword", "3": "exact_id"}.get(
                input("  type  1=natural 2=keyword 3=exact_id > ").strip())
            if not qtype:
                print("  (not a valid type -- nothing saved)")
                continue
            span = ask_span(default_doc=r["gold_spans"][0]["doc_id"])
            if not span:
                continue
            new = {
                "query_id": next_id(rows),
                "query": query,
                "query_type": qtype,
                "query_source": "human_authored",
                "replaces": r["query_id"],
                "gold_spans": [span],
                "require": "all",
                "answer_passage": doc(span["doc_id"])[span["start"]:span["end"]],
                "reference_answer": None,
                "_changes": ["human_authored"],
            }
            stamp(new, "accept")
            rows.append(new)
            r["replaced_by"] = new["query_id"]
            stamp(r, "reject", "replaced")
            print(f"  added query {new['query_id']}, rejected {r['query_id']} as replaced")
            return "r"


def list_rows(rows: list[dict]) -> None:
    for r in rows:
        rv = r.get("review")
        status = (f"{rv['decision']:6s}" + (f" {rv['reason']}" if rv.get("reason") else "")
                  if rv else "pending")
        print(f"{r['query_id']:>5}  {status:34s} {r['query_type']:13s} {r['query'][:70]}")


def parse_ids(arg: str) -> list[int]:
    return [int(x) for x in arg.replace(" ", "").split(",") if x]


def main() -> None:
    rows = load()
    args = sys.argv[1:]
    by_id = {r["query_id"]: i for i, r in enumerate(rows)}

    if "--stats" in args:
        stats(rows)
        return
    if "--list" in args:
        list_rows(rows)
        return

    if "--ids" in args:
        wanted = parse_ids(args[args.index("--ids") + 1])
        missing = [q for q in wanted if q not in by_id]
        if missing:
            print(f"no such query id: {missing}")
        queue = deque(by_id[q] for q in wanted if q in by_id)
    else:
        start = int(args[args.index("--from") + 1]) if "--from" in args else None
        pending = [i for i, r in enumerate(rows) if "review" not in r
                   and (start is None or r["query_id"] >= start)]
        pending.sort(key=lambda i: rows[i]["query_id"])
        queue = deque(pending)

    if not queue:
        print("nothing to review")
        stats(rows)
        return

    stats(rows)
    input(f"\n{len(queue)} queued. Enter to start, Ctrl-C to abort. ")

    done, decided = 0, set()
    while queue:
        i = queue.popleft()
        if i in decided:                  # already handled via 'go to' this session
            continue
        choice = review(rows[i], f"[{done} done, {len(queue)} queued]", rows)

        if isinstance(choice, tuple):     # ('g', query_id)
            target = by_id.get(choice[1])
            if target is None:
                print(f"  no query {choice[1]}")
                queue.appendleft(i)
                continue
            queue.appendleft(i)           # come back to this one afterwards
            queue.appendleft(target)
            decided.discard(target)       # allow revisiting a record decided earlier this session
            continue

        if choice == "q":
            break
        if choice == "a":
            stamp(rows[i], "accept")
        if choice in ("a", "r"):
            decided.add(i)
            save(rows)                    # after every decision -- resumable
            done += 1
            by_id = {r["query_id"]: k for k, r in enumerate(rows)}   # 'n' may have appended

    # A skipped or abandoned record must not keep a half-applied change marker.
    for r in rows:
        if "review" in r and "_changes" in r:
            r.pop("_changes")
    save(rows)
    print(f"\n{done} decided this session.")
    stats(rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\ninterrupted -- decisions up to the last one are saved")
