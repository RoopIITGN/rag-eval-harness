"""
One-off repair: re-spanned records kept their old answer_passage.

Before this was fixed in verify_evalset.py, re-pointing a query with 'p'
updated gold_spans but left answer_passage quoting the original text -- for a
supersession query, that's the stale answer. Scoring uses gold_spans, so no
result was affected, but the file misdescribed itself. This rewrites
answer_passage from the current first span and keeps the old one as
original_answer_passage.

Safe to re-run: records already correct are left alone.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/goldset.jsonl"
rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]

fixed = []
for r in rows:
    changes = list((r.get("review") or {}).get("changes", []))
    changes += [c for h in r.get("review_history", []) for c in h.get("changes", [])]
    if "respanned" not in changes:
        continue
    s = r["gold_spans"][0]
    text = (ROOT / "data/documents" / f"{s['doc_id']}.txt").read_text(encoding="utf-8")[s["start"]:s["end"]]
    if r.get("answer_passage") != text:
        r.setdefault("original_answer_passage", r.get("answer_passage"))
        r["answer_passage"] = text
        fixed.append(r["query_id"])

tmp = GOLD.with_suffix(".jsonl.tmp")
tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
tmp.replace(GOLD)
print(f"fixed answer_passage on: {fixed or 'nothing -- already correct'}")
