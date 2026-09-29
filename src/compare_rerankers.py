"""
Compare rerankers offline, against the retrieval runs already committed.

No index, no Azure, no cost. `reports/runs.jsonl` holds the top-20 chunk ids per
query and `data/chunks_*.jsonl` holds the text, so any cross-encoder can be
scored over exactly the same candidates the original run produced. That is the
whole point of having saved the ranked lists: a reranking question is a
rescoring question, not a retrieval question.

Two limits worth stating before reading any result.

  The ceiling is recall@20 -- 97.2% against hybrid's 93.9%. Reranking can only
  reorder what was retrieved, so the entire prize is about seven queries however
  good the model is.

  The generator flips on 20% of queries between identical runs, which is about
  six queries. A reranker that gains three is inside that. This is why the
  script reports a paired test rather than two percentages.

Usage
  PYTHONPATH=src python src/compare_rerankers.py
  PYTHONPATH=src python src/compare_rerankers.py BAAI/bge-reranker-base
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG, MODE, K = "512-recursive", "hybrid", 5

MODELS = sys.argv[1:] or [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # the one already measured
    "BAAI/bge-reranker-v2-m3",                # 8192-token window, contrastively trained
]


def load():
    gold = metrics.load_gold(str(ROOT / "data/goldset.jsonl"))
    runs = metrics.load_runs(str(ROOT / "reports/runs.jsonl"))
    text = {r["id"]: r["content"] for r in
            map(json.loads, (ROOT / f"data/chunks_{CONFIG}.jsonl")
                .read_text(encoding="utf-8").splitlines())}
    return gold, runs[(CONFIG, MODE)], text


def rescore(model_name: str, gold, base, text) -> dict[int, list[dict]]:
    """Reorder each query's existing candidates with this model."""
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(model_name, device="cpu", max_length=512
                     if "MiniLM" in model_name else 1024)
    out, t0 = {}, time.time()
    for n, (qid, results) in enumerate(base.items(), start=1):
        g = gold.get(qid)
        if not g:
            continue
        pairs = [(g["query"], text[r["id"]]) for r in results if r["id"] in text]
        usable = [r for r in results if r["id"] in text]
        if not usable:
            continue
        scores = m.predict(pairs, show_progress_bar=False)
        order = sorted(zip(usable, scores), key=lambda p: p[1], reverse=True)
        out[qid] = [{**r, "rank": i} for i, (r, _) in enumerate(order, start=1)]
        if n % 20 == 0:
            rate = n / (time.time() - t0)
            print(f"    {n}/{len(base)}  eta {(len(base)-n)/rate/60:.1f}m", end="\r")
    print(" " * 50, end="\r")
    return out


def hits(order, gold) -> dict[int, bool]:
    return {qid: (lambda r: r is not None and r <= K)(
                metrics.satisfied_at_rank(res, gold[qid]["gold_spans"],
                                          gold[qid].get("require", "all")))
            for qid, res in order.items() if qid in gold}


def paired(a: dict[int, bool], b: dict[int, bool]) -> tuple[int, int, float]:
    """a_only, b_only, p -- only queries where the two disagree carry information."""
    from scipy.stats import binomtest
    shared = set(a) & set(b)
    a_only = sum(1 for q in shared if a[q] and not b[q])
    b_only = sum(1 for q in shared if b[q] and not a[q])
    n = a_only + b_only
    return a_only, b_only, (binomtest(a_only, n, 0.5).pvalue if n else 1.0)


def main() -> None:
    gold, base, text = load()
    baseline = hits(base, gold)
    n = len(baseline)
    print(f"{n} queries, candidates from {CONFIG} + {MODE} (pool {len(next(iter(base.values())))})\n")
    print(f"  {'configuration':42s} {'recall@5':>9} {'vs no rerank':>26}")
    print(f"  {'no reranking (RRF order)':42s} {sum(baseline.values())/n:>8.1%}")

    by_type_rows = []
    for name in MODELS:
        print(f"\n  scoring with {name} ...")
        order = rescore(name, gold, base, text)
        h = hits(order, gold)
        a_only, b_only, p = paired(baseline, h)     # a = no rerank
        verdict = (f"lost {a_only}, gained {b_only}, p={p:.3f}")
        print(f"  {name:42s} {sum(h.values())/len(h):>8.1%} {verdict:>26}")

        sub: dict[str, list[bool]] = {}
        for qid, ok in h.items():
            sub.setdefault(gold[qid]["query_type"], []).append(ok)
        by_type_rows.append((name, sub))

    base_sub: dict[str, list[bool]] = {}
    for qid, ok in baseline.items():
        base_sub.setdefault(gold[qid]["query_type"], []).append(ok)
    types = sorted(base_sub)

    print(f"\n  recall@{K} by query type\n")
    print(f"  {'configuration':42s} " + " ".join(f"{t:>13}" for t in types))
    print(f"  {'no reranking':42s} " +
          " ".join(f"{sum(base_sub[t])/len(base_sub[t]):>12.0%} " for t in types))
    for name, sub in by_type_rows:
        print(f"  {name:42s} " +
              " ".join(f"{sum(sub.get(t,[0]))/max(1,len(sub.get(t,[0]))):>12.0%} " for t in types))

    print("\n  Read the paired columns, not the percentages: the ceiling here is")
    print("  recall@20, and a gain smaller than the run-to-run variance is not a gain.")


if __name__ == "__main__":
    main()
