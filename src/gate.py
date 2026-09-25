"""
CI gate. Fails the build when the committed results get worse, or when the
evaluation set stops being internally consistent.

It computes metrics from the committed artefacts -- reports/runs.jsonl and
reports/generations.jsonl -- rather than re-running the pipeline. That means no
Azure credentials in CI, a few seconds instead of forty minutes, and numbers
that only move when someone commits new results. It also means a breach is
never noise: there is no run-to-run variation inside CI to absorb.

Two kinds of check.

  Integrity. Every gold span must still resolve against the frozen text in
  data/documents, query ids must be unique, and verified records must carry at
  least one span. This is the check that matters most in practice: re-extracting
  a PDF shifts every offset in that document and silently invalidates its
  labels. Nothing errors -- the numbers just quietly become wrong -- so it is
  asserted here instead.

  Thresholds. Metrics against config/thresholds.yaml. `block` fails the build;
  `warn` prints and passes.

Usage
  PYTHONPATH=src python src/gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

import metrics

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/goldset.jsonl"
DOCS = ROOT / "data/documents"
RUNS = ROOT / "reports/runs.jsonl"
GENS = ROOT / "reports/generations.jsonl"
THRESHOLDS = ROOT / "config/thresholds.yaml"

CONFIG, MODE = "512-recursive", "hybrid"


# ------------------------------------------------------------------ integrity

def check_integrity() -> list[str]:
    problems: list[str] = []
    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]

    ids = [r["query_id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate query ids: {dupes}")

    cache: dict[str, str] = {}
    for r in rows:
        if not r.get("verified"):
            continue
        if not r.get("gold_spans"):
            problems.append(f"q{r['query_id']}: verified but has no gold span")
            continue
        for s in r["gold_spans"]:
            path = DOCS / f"{s['doc_id']}.txt"
            if not path.exists():
                problems.append(f"q{r['query_id']}: no document {s['doc_id']}")
                continue
            text = cache.setdefault(s["doc_id"], path.read_text(encoding="utf-8"))
            if s["end"] > len(text) or s["start"] >= s["end"]:
                problems.append(
                    f"q{r['query_id']}: span {s['start']}-{s['end']} outside "
                    f"{s['doc_id']} (length {len(text)}) -- was the text re-extracted?")
    return problems


# -------------------------------------------------------------------- metrics

def measured() -> dict[str, float]:
    gold = metrics.load_gold(str(GOLD))
    runs = metrics.load_runs(str(RUNS))
    out: dict[str, float] = {}

    h5, n5 = metrics.recall_at_k(runs, gold, CONFIG, MODE, 5)
    h20, _ = metrics.recall_at_k(runs, gold, CONFIG, MODE, 20)
    out["retrieval.recall_at_5"] = h5 / n5
    out["retrieval.recall_at_20"] = h20 / n5

    recs = [json.loads(l) for l in GENS.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(recs)
    answered = [r for r in recs if r["answered"]]
    graded = [r for r in answered
              if r.get("verdict") and r["verdict"].get("grounded") is not None]
    out["generation.groundedness"] = (
        sum(1 for r in graded if r["verdict"]["grounded"]) / len(graded)) if graded else 0.0
    out["generation.refusal_rate"] = (n - len(answered)) / n
    out["generation.answered_without_source"] = (
        sum(1 for r in recs if r.get("retrieval_hit") is False and r["answered"]) / n)
    out["generation.gate3_rejection_rate"] = (
        sum(1 for r in recs if r.get("bad_citations")) / n)
    return out


# ----------------------------------------------------------------------- main

def main() -> int:
    print("=== goldset integrity")
    problems = check_integrity()
    for p in problems:
        print(f"  FAIL  {p}")
    print("  ok" if not problems else f"  {len(problems)} problem(s)")

    print("\n=== thresholds")
    spec = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))
    values = measured()
    blocked, warned = [], []

    print(f"  {'metric':40s} {'measured':>10} {'limit':>10}  result")
    for section, entries in spec.items():
        if not isinstance(entries, dict):
            continue
        for name, rule in entries.items():
            key = f"{section}.{name}"
            if key not in values:
                continue
            v = values[key]
            if "min" in rule:
                ok, limit = v >= rule["min"], f">= {rule['min']:.2f}"
            else:
                ok, limit = v <= rule["max"], f"<= {rule['max']:.2f}"
            if ok:
                verdict = "pass"
            elif rule.get("on_breach") == "block":
                verdict = "BLOCK"
                blocked.append(key)
            else:
                verdict = "warn"
                warned.append(key)
            print(f"  {key:40s} {v:>9.1%} {limit:>10}  {verdict}")

    print()
    if problems or blocked:
        if problems:
            print(f"FAILED: {len(problems)} integrity problem(s)")
        if blocked:
            print(f"FAILED: {', '.join(blocked)} below threshold")
        return 1
    if warned:
        print(f"passed, with warnings: {', '.join(warned)}")
    else:
        print("passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
