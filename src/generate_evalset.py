"""
Generate a retrieval eval set without baking in a bias against keyword search.

THE PROBLEM THIS SOLVES
-----------------------
The naive approach -- "show an LLM a chunk, ask it to write a question" --
produces questions that are semantic paraphrases of the passage that answers
them. Dense retrieval then looks excellent and BM25 looks useless, because
the generator deliberately avoided lexical overlap.

Any hybrid-vs-dense ablation run on such a set measures the question
generator, not the retrieval system.

WHAT THIS DOES INSTEAD
----------------------
1. Generates from the FULL DOCUMENT, not a single chunk, so the question
   isn't anchored to one passage's wording.
2. Runs a second pass rewriting each question the way a user would actually
   type it -- terse, keyword-shaped, no polite framing.
3. Deliberately mixes in exact-identifier lookups ("what does NSE/MSD/75324
   say about X"), which is the query class where BM25 wins and dense fails.
   Omitting these guarantees a misleading ablation.
4. Tags every query by type so recall can be reported by subgroup.
5. Locates the answer span by string search against the FROZEN document text,
   so gold labels are (doc_id, char_start, char_end) and survive re-chunking.

Output: data/goldset.jsonl -- one record per query.
Every record needs human verification before use; see verify_evalset.py.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from anthropic import Anthropic

DOCS_DIR = Path("data/documents")
OUT_PATH = Path("data/goldset.jsonl")

CIRCULAR_ID = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9()]*)(?:[/-][A-Z0-9()]+){2,}\b")

# Query types, and the share of the set each should occupy.
# The exact_id slice is what keeps the BM25 leg honest.
QUERY_MIX = {
    "natural":    0.40,   # how a person would ask, full sentence
    "keyword":    0.25,   # terse, search-box style
    "exact_id":   0.20,   # names a circular number -- BM25's home ground
    "multi_hop":  0.15,   # needs evidence from two documents
}


GENERATE_PROMPT = """You are building an evaluation set for a document \
retrieval system over Indian securities-market circulars.

Below is the FULL TEXT of one circular. Write {n} questions that a compliance \
or operations analyst might realistically ask, where the answer is contained \
in this document.

Rules:
- Base each question on the document as a whole, not on one paragraph.
- Do NOT paraphrase a single sentence into a question. Ask what someone would \
  actually want to know.
- Vary the specificity. Some questions should be about obligations, some about \
  dates or thresholds, some about who a provision applies to.
- For each question, quote the EXACT sentence or passage from the document that \
  answers it. Copy it verbatim, character for character.

Return JSON only, no preamble:
{{"questions": [{{"question": "...", "answer_passage": "..."}}]}}

DOCUMENT ({doc_id}):
{text}
"""


REWRITE_PROMPT = """Rewrite each question below as the same person would type \
it into a search box -- terse, keyword-shaped, no polite framing, no full \
sentence structure. Keep the meaning identical.

Example:
  in:  "What is the deadline for reporting a margin shortfall to the Exchange?"
  out: "margin shortfall reporting deadline"

Return JSON only: {{"rewritten": ["...", "..."]}}

QUESTIONS:
{questions}
"""


def load_documents() -> dict[str, str]:
    """Load the FROZEN extracted text. Every offset is relative to these
    exact strings -- re-extracting with a different parser invalidates
    every label in the goldset."""
    return {p.stem: p.read_text(encoding="utf-8") for p in DOCS_DIR.glob("*.txt")}


def locate(text: str, passage: str) -> tuple[int, int] | None:
    """Find the answer passage in the source document.

    Models normalise whitespace and quotes when copying, so exact matching
    fails often. Fall back to a whitespace-insensitive search.
    """
    idx = text.find(passage)
    if idx != -1:
        return idx, idx + len(passage)

    # Whitespace-insensitive retry
    pattern = re.escape(passage.strip())
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def generate_for_document(client, doc_id: str, text: str, n: int) -> list[dict]:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        temperature=0.7,          # variety matters here, unlike everywhere else
        messages=[{"role": "user",
                   "content": GENERATE_PROMPT.format(n=n, doc_id=doc_id, text=text)}],
    )
    raw = resp.content[0].text.strip().removeprefix("```json").removesuffix("```")
    return json.loads(raw)["questions"]


def rewrite_as_queries(client, questions: list[str]) -> list[str]:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0.3,
        messages=[{"role": "user",
                   "content": REWRITE_PROMPT.format(questions=json.dumps(questions))}],
    )
    raw = resp.content[0].text.strip().removeprefix("```json").removesuffix("```")
    return json.loads(raw)["rewritten"]


def make_exact_id_queries(doc_id: str, text: str, base: list[dict]) -> list[dict]:
    """Turn a question into an identifier lookup.

    This is the query class dense retrieval cannot handle -- circular numbers
    carry almost no semantic signal, so '/19839/' and '/13804/' embed to
    nearly the same point. Leaving these out of the eval set would make the
    BM25 leg look pointless and the ablation misleading.
    """
    ids = CIRCULAR_ID.findall(text[:1500])   # the circular's own ID is up top
    if not ids:
        return []
    cid = ids[0]
    out = []
    for q in base:
        stripped = re.sub(r"^(what|when|who|how)\s+(is|are|does|do)\s+", "", q["question"],
                          flags=re.I).rstrip("?")
        out.append({**q, "question": f"{cid} {stripped}", "query_type": "exact_id"})
    return out


def main(per_doc: int = 4, seed: int = 0):
    random.seed(seed)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt files in {DOCS_DIR}. Run ingest.py first.")

    records, qid, unlocated = [], 0, 0

    for doc_id, text in docs.items():
        generated = generate_for_document(client, doc_id, text, per_doc)

        # Split the generated questions across query types
        natural = generated[: max(1, int(per_doc * 0.6))]
        for_keyword = generated[max(1, int(per_doc * 0.6)):]

        typed: list[dict] = [{**q, "query_type": "natural"} for q in natural]

        if for_keyword:
            rewritten = rewrite_as_queries(client, [q["question"] for q in for_keyword])
            typed += [{**q, "question": r, "query_type": "keyword"}
                      for q, r in zip(for_keyword, rewritten)]

        typed += make_exact_id_queries(doc_id, text, natural[:1])

        for q in typed:
            span = locate(text, q["answer_passage"])
            if span is None:
                unlocated += 1
                continue                      # cannot label it; drop it
            qid += 1
            records.append({
                "query_id": qid,
                "query": q["question"],
                "query_type": q["query_type"],
                "query_source": "llm_generated",
                "verified": False,            # flipped by verify_evalset.py
                "gold_spans": [{"doc_id": doc_id, "start": span[0], "end": span[1]}],
                "require": "all",
                "answer_passage": q["answer_passage"],
                "reference_answer": None,     # filled later, for groundedness
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for r in records:
        by_type[r["query_type"]] = by_type.get(r["query_type"], 0) + 1

    print(f"wrote {len(records)} queries to {OUT_PATH}")
    print(f"  by type: {by_type}")
    print(f"  dropped {unlocated} (answer passage not locatable in source text)")
    print(f"\nNEXT: run verify_evalset.py and hand-check at least 20%.")


if __name__ == "__main__":
    main()
