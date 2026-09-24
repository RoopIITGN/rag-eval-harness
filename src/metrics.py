"""
Retrieval metrics computed offline from saved ranked lists.

Design principle: retrieve top-50 ONCE per (query, config), save the full
ranked list, then compute every metric from that file. recall@5 and recall@50
are the same retrieval with different cutoffs -- never re-query to compute a
new metric.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict


# --------------------------------------------------------------- hit logic

def satisfied_at_rank(retrieved: list[dict], gold_spans: list[dict],
                      require: str = "all", threshold: float = 0.80) -> int | None:
    """Rank at which the query becomes answerable. None if never.

    Coverage accumulates across chunks: an answer split by a chunk boundary
    is satisfied once the retrieved set jointly covers `threshold` of the
    gold span. Strict single-chunk matching unfairly penalises small chunks
    for something the generator wouldn't care about.
    """
    covered: dict[int, set[int]] = {i: set() for i in range(len(gold_spans))}

    for rank, chunk in enumerate(retrieved, start=1):
        for i, span in enumerate(gold_spans):
            if chunk["doc_id"] != span["doc_id"]:
                continue
            lo = max(chunk["char_start"], span["start"])
            hi = min(chunk["char_end"], span["end"])
            if hi > lo:
                covered[i].update(range(lo, hi))

        done = [len(covered[i]) / max(1, s["end"] - s["start"]) >= threshold
                for i, s in enumerate(gold_spans)]

        if (all(done) if require == "all" else any(done)):
            return rank
    return None


def satisfied_at_budget(retrieved: list[dict], gold_spans: list[dict],
                        require: str = "all", threshold: float = 0.80) -> int | None:
    """Cumulative TOKEN count at which the query becomes answerable.

    This is the fair cross-chunk-size comparison. Top-5 of 512-token chunks
    and top-5 of 256-token chunks hand the generator different amounts of
    text, so fixed-k quietly favours larger chunks.
    """
    covered: dict[int, set[int]] = {i: set() for i in range(len(gold_spans))}
    cumulative = 0

    for chunk in retrieved:
        cumulative += chunk["token_count"]
        for i, span in enumerate(gold_spans):
            if chunk["doc_id"] != span["doc_id"]:
                continue
            lo = max(chunk["char_start"], span["start"])
            hi = min(chunk["char_end"], span["end"])
            if hi > lo:
                covered[i].update(range(lo, hi))

        done = [len(covered[i]) / max(1, s["end"] - s["start"]) >= threshold
                for i, s in enumerate(gold_spans)]

        if (all(done) if require == "all" else any(done)):
            return cumulative
    return None


# ----------------------------------------------------------------- loading

def load_runs(path: str) -> dict[tuple[str, str], dict[int, list[dict]]]:
    runs: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            runs[(rec["config"], rec["mode"])][rec["query_id"]] = rec["results"]
    return runs


def load_gold(path: str) -> dict[int, dict]:
    with open(path) as f:
        return {r["query_id"]: r for r in map(json.loads, f) if r.get("verified")}


# ----------------------------------------------------------------- metrics

def recall_at_k(runs, gold, config, mode, k) -> tuple[int, int]:
    hits = 0
    per_query = runs[(config, mode)]
    for qid, results in per_query.items():
        g = gold.get(qid)
        if g is None:
            continue
        r = satisfied_at_rank(results, g["gold_spans"], g.get("require", "all"))
        hits += int(r is not None and r <= k)
    return hits, sum(1 for q in per_query if q in gold)


def recall_at_budget(runs, gold, config, mode, budget) -> tuple[int, int]:
    hits = 0
    per_query = runs[(config, mode)]
    for qid, results in per_query.items():
        g = gold.get(qid)
        if g is None:
            continue
        b = satisfied_at_budget(results, g["gold_spans"], g.get("require", "all"))
        hits += int(b is not None and b <= budget)
    return hits, sum(1 for q in per_query if q in gold)


def recall_by_subgroup(runs, gold, config, mode, k) -> dict[str, tuple[int, int]]:
    """Recall broken out by query_type.

    Blended recall hides the thing that matters: exact_id queries are where
    dense retrieval fails and BM25 earns its place. Reporting one number
    averages that signal away.
    """
    buckets: dict[str, list[bool]] = defaultdict(list)
    for qid, results in runs[(config, mode)].items():
        g = gold.get(qid)
        if g is None:
            continue
        r = satisfied_at_rank(results, g["gold_spans"], g.get("require", "all"))
        buckets[g["query_type"]].append(r is not None and r <= k)
    return {t: (sum(v), len(v)) for t, v in buckets.items()}


# --------------------------------------------------------- significance

def mcnemar_runs(runs, gold, key_a: tuple[str, str], key_b: tuple[str, str], k) -> dict:
    """Paired comparison between any two runs, each identified by (config, mode).

    Only DISCORDANT queries carry information -- queries both runs get right
    (or both wrong) say nothing about which is better, and including them just
    adds variance.

    Keyed by (config, mode) rather than by config alone, because the headline
    comparison of this harness is four retrieval modes within one chunking
    configuration, not one mode across configurations.
    """
    from scipy.stats import binomtest

    hit = lambda r: r is not None and r <= k
    a_only = b_only = both = neither = 0
    for qid, g in gold.items():
        ra, rb = runs[key_a].get(qid), runs[key_b].get(qid)
        if ra is None or rb is None:
            continue
        ha = hit(satisfied_at_rank(ra, g["gold_spans"], g.get("require", "all")))
        hb = hit(satisfied_at_rank(rb, g["gold_spans"], g.get("require", "all")))
        if ha and hb:   both += 1
        elif ha:        a_only += 1
        elif hb:        b_only += 1
        else:           neither += 1

    n_disc = a_only + b_only
    p = binomtest(a_only, n_disc, 0.5).pvalue if n_disc else 1.0
    return {"a_only": a_only, "b_only": b_only, "both": both,
            "neither": neither, "discordant": n_disc, "p_value": p}


def mcnemar(runs, gold, config_a, config_b, mode, k) -> dict:
    """Two configurations at the same retrieval mode."""
    return mcnemar_runs(runs, gold, (config_a, mode), (config_b, mode), k)


def bootstrap_ci(runs, gold, config, mode, k, n_boot=5000, seed=0):
    """Percentile bootstrap CI on recall@k.

    Honest error bars without sacrificing queries to a holdout -- which at
    this sample size would leave both halves too small to be useful.
    """
    rng = random.Random(seed)
    outcomes = []
    for qid, results in runs[(config, mode)].items():
        g = gold.get(qid)
        if g is None:
            continue
        r = satisfied_at_rank(results, g["gold_spans"], g.get("require", "all"))
        outcomes.append(int(r is not None and r <= k))

    n = len(outcomes)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(sum(rng.choices(outcomes, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
