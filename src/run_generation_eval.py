"""
Evaluate the generation layer: answer every verified query, judge groundedness,
and measure how much of the result is noise.

Four things are measured, and the last is what makes the first three usable.

  Groundedness -- does the answer follow from the chunks it was given? Judged by
  gpt-5, a different family from the gpt-5-mini generator, so the judge is not
  scoring its own output. Groundedness is attribution, not correctness: a
  confident answer built from the wrong retrieved chunk is perfectly grounded
  and still wrong, which is why retrieval is scored separately.

  Refusal correctness -- retrieval already told us, per query, whether the gold
  span was in the top 5. So refusals can be checked without a judge: refusing
  when retrieval missed is right, refusing when it succeeded is a lost answer,
  and answering when it missed is the dangerous case.

  Gate rejections -- how often a citation failed verification. If this is zero,
  the gate is decoration; if it is high, the generator is fabricating quotes and
  the gate is the only thing catching it.

  Variance -- neither model accepts temperature 0, so both are non-deterministic.
  A sample of queries is answered twice and judged twice. Any difference smaller
  than these numbers is not a difference.

Writes reports/generation.md and reports/generations.jsonl, the second holding
every answer and verdict so the report can be rebuilt without new API calls.

Usage
  PYTHONPATH=src python src/run_generation_eval.py                # full run
  PYTHONPATH=src python src/run_generation_eval.py --limit 20     # cheap trial
  PYTHONPATH=src python src/run_generation_eval.py --report       # rebuild report
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
import random
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

import generate
import metrics

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GOLD = ROOT / "data/goldset.jsonl"
RUNS = ROOT / "reports/runs.jsonl"
OUT = ROOT / "reports/generations.jsonl"
REPORT = ROOT / "reports/generation.md"

VARIANCE_SAMPLE = 30
WORKERS = 6        # each query is 2-4 sequential reasoning calls; the wait dominates
MAX_TOKENS = 4000

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "verdict",
        "description": "Judge whether an answer is supported by the supplied context.",
        "parameters": {
            "type": "object",
            "properties": {
                "grounded": {
                    "type": "boolean",
                    "description": "True only if EVERY claim in the answer is supported "
                                   "by the context. Judge support, not correctness.",
                },
                "unsupported_claim": {
                    "type": "string",
                    "description": "The first claim not supported by the context, or an "
                                   "empty string if grounded.",
                },
            },
            "required": ["grounded", "unsupported_claim"],
        },
    },
}

JUDGE_PROMPT = """Decide whether the ANSWER is supported by the CONTEXT.

Judge support, not truth. An answer may be factually right and still \
unsupported if the context does not contain it -- mark that ungrounded. An \
answer may be wrong and still grounded if the context says so.

CONTEXT
{context}

QUESTION: {question}

ANSWER
{answer}
"""


def client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def judge(c: AzureOpenAI, question: str, answer: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(f"[{ch['chunk_id']}] {ch['content']}" for ch in chunks)
    for attempt in range(4):
        try:
            r = c.chat.completions.create(
                model=os.environ["AZURE_OPENAI_JUDGE_DEPLOYMENT"],
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    context=context, question=question, answer=answer)}],
                tools=[JUDGE_TOOL],
                tool_choice={"type": "function", "function": {"name": "verdict"}},
                max_completion_tokens=MAX_TOKENS,
            )
            calls = r.choices[0].message.tool_calls
            if not calls:
                return {"grounded": None, "unsupported_claim": "judge returned no verdict"}
            return json.loads(calls[0].function.arguments)
        except Exception as e:
            if attempt == 3:
                return {"grounded": None, "unsupported_claim": f"judge error: {type(e).__name__}"}
            time.sleep(2 ** attempt * 5)


def retrieved_text(result: generate.Result) -> list[dict]:
    """Rebuild the context the generator saw, for the judge."""
    import retrieve
    chunks = retrieve.retrieve(result.query, generate.CONFIG, generate.MODE,
                               top=generate.TOP_K)[:generate.TOP_K]
    return [{"chunk_id": f"chunk_{i}", "content": ch["content"]}
            for i, ch in enumerate(chunks, start=1)]


def run(gold: dict, limit: int | None) -> None:
    c = client()
    runs = metrics.load_runs(str(RUNS))
    per_query = runs[(generate.CONFIG, generate.MODE)]

    items = list(gold.items())
    if limit:
        items = items[:limit]
    sample = {qid for qid, _ in random.Random(0).sample(items, min(VARIANCE_SAMPLE, len(items)))}

    def one(item):
        try:
            return _one(item)
        except Exception as e:                       # one query must not end the run
            qid, g = item
            return {"query_id": qid, "query": g["query"], "query_type": g["query_type"],
                    "retrieval_hit": None, "answered": False,
                    "refusal_reason": f"pipeline error: {type(e).__name__}",
                    "answer": "", "citations": 0, "bad_citations": 0, "verdict": None}

    def _one(item):
        qid, g = item
        res = per_query.get(qid, [])
        rank = metrics.satisfied_at_rank(res, g["gold_spans"], g.get("require", "all"))
        retrieval_hit = bool(rank and rank <= generate.TOP_K)

        r = generate.answer(g["query"], client=c)
        ctx = retrieved_text(r)
        verdict = judge(c, g["query"], r.answer, ctx) if r.answered else None

        rec = {"query_id": qid, "query": g["query"], "query_type": g["query_type"],
               "retrieval_hit": retrieval_hit, "answered": r.answered,
               "refusal_reason": r.refusal_reason, "answer": r.answer,
               "citations": len(r.citations), "bad_citations": len(r.bad_citations),
               "verdict": verdict}
        if qid in sample:
            r2 = generate.answer(g["query"], client=c)
            rec["repeat_answered"] = r2.answered
            rec["repeat_answer"] = r2.answer
            if r.answered:
                rec["repeat_verdict"] = judge(c, g["query"], r.answer, ctx)
        return rec

    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, OUT.open("w", encoding="utf-8") as f:
        for rec in pool.map(one, items):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(items)}  {rate:.2f}/s  "
                  f"eta {(len(items)-done)/rate/60:.1f}m", end="\r")
    print(f"\ngenerations written -> {OUT}")


def pct(a: int, b: int) -> str:
    return f"{100*a/b:.1f}%" if b else "-"


def build_report(recs: list[dict]) -> str:
    o: list[str] = []
    w = o.append
    n = len(recs)
    answered = [r for r in recs if r["answered"]]
    refused = [r for r in recs if not r["answered"]]

    w("# Generation\n")
    w(f"{n} verified queries. Generator `gpt-5-mini`, judge `gpt-5` — different "
      f"models, same family, which is noted as a limitation. Retrieval is "
      f"`{generate.CONFIG}` + `{generate.MODE}`, top {generate.TOP_K}.\n")

    # ---- answered vs refused -------------------------------------------
    w("## Answered and refused\n")
    w(f"| | count | share |")
    w("|---|---|---|")
    w(f"| Answered | {len(answered)} | {pct(len(answered), n)} |")
    w(f"| Refused | {len(refused)} | {pct(len(refused), n)} |")
    w("")
    reasons = Counter(r["refusal_reason"] for r in refused)
    if reasons:
        w("Refusal reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()) + ".\n")

    # ---- groundedness ---------------------------------------------------
    graded = [r for r in answered if r["verdict"] and r["verdict"].get("grounded") is not None]
    grounded = sum(1 for r in graded if r["verdict"]["grounded"])
    w("## Groundedness\n")
    w(f"**{pct(grounded, len(graded))}** of answered queries ({grounded}/{len(graded)}) "
      f"are judged fully supported by the retrieved chunks.\n")
    w("Groundedness is attribution, not correctness. An answer drawn confidently "
      "from the wrong retrieved chunk scores as grounded — which is why retrieval "
      "is measured separately, and why the supersession slice matters.\n")

    by_type: dict[str, list[bool]] = {}
    for r in graded:
        by_type.setdefault(r["query_type"], []).append(r["verdict"]["grounded"])
    w("| Query type | answered | grounded |")
    w("|---|---|---|")
    for t, v in sorted(by_type.items()):
        w(f"| {t} | {len(v)} | {pct(sum(v), len(v))} |")
    w("")

    # ---- refusal correctness -------------------------------------------
    w("## Refusal correctness\n")
    w("Retrieval already tells us, per query, whether the gold span was in the "
      "top 5. So refusals can be scored without a judge.\n")
    quad = Counter((r["retrieval_hit"], r["answered"]) for r in recs)
    w("| Retrieval | Pipeline | count | |")
    w("|---|---|---|---|")
    w(f"| found the answer | answered | {quad[(True, True)]} | as intended |")
    w(f"| found the answer | refused | {quad[(True, False)]} | lost answer |")
    w(f"| missed | refused | {quad[(False, False)]} | correct refusal |")
    w(f"| missed | answered | {quad[(False, True)]} | **answered without the source** |")
    w("")
    w("The last row is the one that matters. An answer produced when retrieval "
      "missed is either drawn from the model's own knowledge or from the wrong "
      "chunk, and citation verification is the only thing standing between it "
      "and the user.\n")

    # ---- gate 3 ---------------------------------------------------------
    rejected = [r for r in recs if r["bad_citations"]]
    w("## Citation verification\n")
    w(f"{len(rejected)} of {n} queries had at least one citation fail verification "
      f"— a quote that does not appear in the chunk it cites, or a chunk that was "
      f"never retrieved.\n")
    if not rejected:
        w("None failed in this run. The gate costs nothing to keep, but on this "
          "corpus it is not currently catching anything.\n")

    # ---- variance -------------------------------------------------------
    rep = [r for r in recs if "repeat_answered" in r]
    w("## Variance\n")
    if not rep:
        w("No repeat sample in this run.\n")
    else:
        flip = sum(1 for r in rep if r["repeat_answered"] != r["answered"])
        jrep = [r for r in rep if r.get("repeat_verdict") and r["verdict"]]
        jflip = sum(1 for r in jrep
                    if r["repeat_verdict"].get("grounded") != r["verdict"].get("grounded"))
        w(f"Neither model accepts `temperature=0`, so both are non-deterministic. "
          f"A sample of {len(rep)} queries was answered twice and judged twice.\n")
        w(f"| | flips | of | rate |")
        w("|---|---|---|---|")
        w(f"| Generator: answered vs refused | {flip} | {len(rep)} | {pct(flip, len(rep))} |")
        w(f"| Judge: grounded vs not, same answer | {jflip} | {len(jrep)} | {pct(jflip, len(jrep))} |")
        w("")
        w("These are the floor. A measured difference smaller than the judge flip "
          "rate is not a difference, and any future change to the prompt or the "
          "generator has to clear it before it can be called an improvement.\n")

    # ---- failures --------------------------------------------------------
    bad = [r for r in graded if not r["verdict"]["grounded"]]
    if bad:
        w(f"## Ungrounded answers — {len(bad)}\n")
        w("| Query | Type | First unsupported claim |")
        w("|---|---|---|")
        for r in bad[:20]:
            claim = (r["verdict"].get("unsupported_claim") or "")[:90]
            w(f"| {r['query_id']}. {r['query'][:60]} | {r['query_type']} | {claim} |")
        w("")
    return "\n".join(o) + "\n"


def main() -> None:
    gold = metrics.load_gold(str(GOLD))
    if "--report" not in sys.argv:
        limit = None
        if "--limit" in sys.argv:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        print(f"{len(gold)} verified queries"
              + (f", running first {limit}" if limit else ""))
        run(gold, limit)
    elif not OUT.exists():
        sys.exit(f"{OUT} not found — run without --report first")

    recs = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    REPORT.write_text(build_report(recs), encoding="utf-8")
    print(f"report written -> {REPORT}")


if __name__ == "__main__":
    main()
