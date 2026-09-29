"""
Was the reranker ever shown the whole chunk?

`ms-marco-MiniLM-L-6-v2` takes 512 tokens for the query and the document
TOGETHER. Chunks here run to 512 tokens on their own, so on the longer ones the
document is silently cut before the model scores it -- the model may never see
the text the answer is in.

That would be a mechanical explanation for reranking costing 7.5 points of
recall, and a better one than "the model is out of domain". The two have
different fixes: a longer-context reranker fixes truncation, a differently
trained one fixes domain.

Nothing is scored here and no model runs. This only tokenises the same
(query, chunk) pairs the reranker saw and counts them. No Azure, no cost.

Usage
  PYTHONPATH=src python src/check_truncation.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports/runs.jsonl"
GOLD = ROOT / "data/goldset.jsonl"
CHUNKS = ROOT / "data/chunks_512-recursive.jsonl"
CONFIG, MODE, LIMIT = "512-recursive", "hybrid", 512


def main() -> None:
    from sentence_transformers import CrossEncoder

    queries = {r["query_id"]: r["query"]
               for r in map(json.loads, GOLD.read_text(encoding="utf-8").splitlines())
               if r.get("verified")}
    text = {r["id"]: r["content"]
            for r in map(json.loads, CHUNKS.read_text(encoding="utf-8").splitlines())}

    tok = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu").tokenizer

    lengths: list[int] = []
    truncated_top5 = Counter()          # per query: how many of its top 5 were cut
    pairs_seen = 0

    for line in RUNS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["config"] != CONFIG or rec["mode"] != MODE:
            continue
        q = queries.get(rec["query_id"])
        if q is None:
            continue
        cut_here = 0
        for i, r in enumerate(rec["results"]):
            content = text.get(r["id"])
            if content is None:
                continue
            n = len(tok(q, content)["input_ids"])
            lengths.append(n)
            pairs_seen += 1
            if n > LIMIT and i < 5:
                cut_here += 1
        truncated_top5[cut_here] += 1

    if not lengths:
        raise SystemExit("no pairs matched -- check that runs.jsonl holds chunk ids")

    over = [n for n in lengths if n > LIMIT]
    s = sorted(lengths)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]

    print(f"{pairs_seen} (query, chunk) pairs, limit {LIMIT} tokens\n")
    print(f"  {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6}")
    print(f"  {q(.5):>6} {q(.9):>6} {q(.99):>6} {max(s):>6}\n")
    print(f"  truncated pairs: {len(over)} of {pairs_seen} ({100*len(over)/pairs_seen:.1f}%)")
    if over:
        print(f"  when truncated, tokens lost: median {q(.5) and sorted(over)[len(over)//2]-LIMIT}, "
              f"worst {max(over)-LIMIT}")
    print()
    affected = sum(v for k, v in truncated_top5.items() if k)
    print(f"  queries with at least one truncated chunk in the top 5: "
          f"{affected} of {sum(truncated_top5.values())}")
    print()

    pct = 100 * len(over) / pairs_seen
    if pct < 2:
        print("Reading: truncation is not the explanation. Almost every chunk fit,")
        print("so the reranker saw the text and still misordered it -- which points")
        print("at the training domain, not the context window.")
    elif pct < 15:
        print("Reading: truncation is real but partial. It may account for some of")
        print("the loss, not all of it. A longer-context reranker would separate the")
        print("two causes -- if it recovers most of the gap, it was the window.")
    else:
        print("Reading: truncation is a mechanical explanation for the reranking")
        print("loss. On these pairs the model was scoring a document it could not")
        print("fully see. A reranker with a longer window is the first thing to try.")


if __name__ == "__main__":
    main()
