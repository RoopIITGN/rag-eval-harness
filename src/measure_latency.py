"""
Latency, split into the parts that mean different things.

One end-to-end number would be unreadable here: the embedding call crosses from
Central India to Sweden Central, so a single figure mostly measures a network
hop this deployment chose for model availability, not the retrieval system.

  search      -- Azure's own processing, as reported by the service. No network.
  embed       -- the cross-region hop. An artefact of the deployment.
  retrieval   -- what a client actually waits for, embed + search + parse.

Generation is not included: it is a reasoning model and runs to seconds, an
order of magnitude above everything here.

Usage
  PYTHONPATH=src python src/measure_latency.py [n_queries]
"""
import json, sys, time
from pathlib import Path
import metrics, retrieve

ROOT = Path(__file__).resolve().parents[1]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    gold = metrics.load_gold(str(ROOT / "data/goldset.jsonl"))
    queries = [g["query"] for g in list(gold.values())[:N]]
    retrieve.embed.cache_clear()          # otherwise the hop is measured once

    embed_ms, search_ms, total_ms = [], [], []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        retrieve.embed(q)
        t1 = time.perf_counter()
        rows = retrieve.retrieve(q, "512-recursive", "hybrid", top=20)
        t2 = time.perf_counter()
        embed_ms.append((t1 - t0) * 1000)
        total_ms.append((t2 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)   # embedding is cached by now
        print(f"  {i}/{len(queries)}", end="\r")

    print(f"\n{len(queries)} queries\n")
    print(f"  {'stage':12s} {'p50':>9} {'p95':>9} {'max':>9}")
    for name, xs in [("embed", embed_ms), ("search", search_ms), ("retrieval", total_ms)]:
        print(f"  {name:12s} {pct(xs,.5):>8.0f}ms {pct(xs,.95):>8.0f}ms {max(xs):>8.0f}ms")

    out = ROOT / "reports/latency.json"
    out.write_text(json.dumps({
        "n": len(queries),
        "note": "embed crosses Central India -> Sweden Central; search is Azure-side only",
        **{k: {"p50": pct(v, .5), "p95": pct(v, .95), "max": max(v)}
           for k, v in [("embed_ms", embed_ms), ("search_ms", search_ms),
                        ("retrieval_ms", total_ms)]}}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
