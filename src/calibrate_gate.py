"""
Calibrate Gate 1 -- the refusal that fires BEFORE the generator is called.

When retrieval fails, the honest response is "I don't have that", not an answer
built from whatever came back. The cheapest way to detect that is the retrieval
score itself: if the best chunk scores poorly, nothing worth answering from was
found.

Picking the threshold by eye would be guessing. This reads the saved runs,
splits queries into those retrieval got right at k=5 and those it missed, and
shows how the two score distributions separate. A threshold is only worth
setting where they do.

Reads reports/runs.jsonl -- no API calls, no index queries.

Usage
  PYTHONPATH=src python src/calibrate_gate.py
"""

from __future__ import annotations

from pathlib import Path

import metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG, MODE, K = "512-recursive", "hybrid", 5


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"min": s[0], "p10": q(0.10), "p25": q(0.25), "median": q(0.50),
            "p75": q(0.75), "max": s[-1]}


def main() -> None:
    gold = metrics.load_gold(str(ROOT / "data/goldset.jsonl"))
    runs = metrics.load_runs(str(ROOT / "reports/runs.jsonl"))
    per_query = runs[(CONFIG, MODE)]

    hit_top1, miss_top1, hit_mean5, miss_mean5 = [], [], [], []
    for qid, results in per_query.items():
        g = gold.get(qid)
        if not g or not results:
            continue
        rank = metrics.satisfied_at_rank(results, g["gold_spans"], g.get("require", "all"))
        top1 = results[0]["search_score"]
        mean5 = sum(r["search_score"] for r in results[:5]) / min(5, len(results))
        (hit_top1 if (rank and rank <= K) else miss_top1).append(top1)
        (hit_mean5 if (rank and rank <= K) else miss_mean5).append(mean5)

    print(f"{CONFIG} / {MODE}: {len(hit_top1)} retrieved at k={K}, {len(miss_top1)} missed\n")
    for name, hit, miss in [("top-1 score", hit_top1, miss_top1),
                            ("mean of top 5", hit_mean5, miss_mean5)]:
        h, m = quantiles(hit), quantiles(miss)
        print(f"--- {name}")
        print("        " + "".join(f"{k:>10}" for k in h))
        print("  hit   " + "".join(f"{v:>10.4f}" for v in h.values()))
        print("  miss  " + "".join(f"{v:>10.4f}" for v in m.values()))
        # A threshold is only useful if it rejects misses without rejecting hits.
        print(f"  {'threshold':>12} {'refused hits':>14} {'refused misses':>16}")
        lo, hi = min(hit + miss), max(hit + miss)
        for i in range(1, 10):
            t = lo + (hi - lo) * i / 10
            fh = sum(1 for x in hit if x < t)
            fm = sum(1 for x in miss if x < t)
            print(f"  {t:>12.4f} {fh:>10}/{len(hit):<4} {fm:>12}/{len(miss):<4}")
        print()

    print("Read the table as a trade: a useful threshold refuses a large share of")
    print("the misses while refusing almost none of the hits. If no row does both,")
    print("the score carries no signal here and Gate 1 should stay open -- with")
    print("Gate 2 (citation and quote verification) doing the work instead.")


if __name__ == "__main__":
    main()
