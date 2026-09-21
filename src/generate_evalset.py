"""
Generate a retrieval eval set without baking in a bias against keyword search.

THE PROBLEM THIS SOLVES
-----------------------
The naive approach -- "show an LLM a chunk, ask it to write a question" --
produces questions that are semantic paraphrases of the passage that answers
them. Dense retrieval then looks excellent and BM25 looks useless, because the
generator deliberately avoided lexical overlap. Any hybrid-vs-dense ablation
run on such a set measures the question generator, not the retrieval system.

WHAT THIS DOES INSTEAD
----------------------
1. Generates from the FULL DOCUMENT, not a single chunk, so the question isn't
   anchored to one passage's wording.
2. Rewrites a share of questions into search-box phrasing -- terse,
   keyword-shaped, the way a person actually types.
3. Deliberately includes exact-identifier lookups, the query class where BM25
   wins and dense retrieval fails. Omitting these guarantees a misleading
   ablation.
4. Tags every query by type so recall is reported by subgroup, not blended.
5. Locates the answer span by string search against the FROZEN document text,
   so gold labels are (doc_id, char_start, char_end) and survive re-chunking.

Every record is written with verified: false. Run verify_evalset.py before use.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data/documents"
OUT_PATH = ROOT / "data/goldset.jsonl"

PER_DOC = 3           # 47 docs x 3 -> ~140 candidates, all hand-verifiable
MAX_TOKENS = 8000     # reasoning models spend budget before emitting output
HEADER_CHARS = 250    # spans starting inside the letterhead are not useful

CIRCULAR_ID = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9()]*)(?:[/-][A-Z0-9()]+){2,}\b")


GENERATE_PROMPT = """You are building an evaluation set for a document \
retrieval system over Indian securities-market circulars.

Below is the FULL TEXT of one circular. Write {n} questions that a compliance \
or operations analyst might realistically ask, where the answer is contained in \
this document.

Rules:
- Base each question on the document as a whole, not on one paragraph.
- Do NOT paraphrase a single sentence into a question. Ask what someone would \
actually want to know.
- Vary the specificity: some about obligations, some about dates or thresholds, \
some about who a provision applies to.
- Do NOT ask about the letterhead, the circular's own reference number, or the \
signatory. Ask about substance.
- For each question, quote the EXACT passage from the document that answers it. \
Copy it verbatim, character for character, including punctuation. This is used \
to locate the answer in the source text, so an approximate quote is useless.

Return JSON only, no preamble, no code fences:
{{"questions": [{{"question": "...", "answer_passage": "..."}}]}}

DOCUMENT ({doc_id}):
{text}
"""

REWRITE_PROMPT = """Rewrite each question below as the same person would type it \
into a search box: terse, keyword-shaped, no polite framing, no full sentence \
structure. Keep the meaning identical.

Example:
  in:  "What is the deadline for reporting a margin shortfall to the Exchange?"
  out: "margin shortfall reporting deadline"

Return JSON only, no preamble, no code fences:
{{"rewritten": ["...", "..."]}}

QUESTIONS:
{questions}
"""


def client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def ask(c: AzureOpenAI, prompt: str) -> dict:
    """One call. No temperature -- reasoning models reject anything but the
    default, and variety here comes from the documents, not from sampling."""
    resp = c.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError(f"empty response (finish_reason={resp.choices[0].finish_reason})")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    return json.loads(raw)


def locate(text: str, passage: str) -> tuple[int, int] | None:
    """Find the answer passage in the frozen document text.

    Models normalise whitespace and quote characters when copying, so exact
    matching fails often enough to need a fallback.
    """
    idx = text.find(passage)
    if idx != -1:
        return idx, idx + len(passage)

    cleaned = passage.strip()
    if not cleaned:
        return None

    pattern = re.escape(cleaned)
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)          # whitespace-insensitive
    pattern = pattern.replace("\\'", "['\u2019]").replace('\\"', '["\u201c\u201d]')
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def own_serial(doc_id: str, text: str) -> str | None:
    """The circular's serial number, confirmed present in its own header.

    Taken from the filename rather than parsed out of the text: PDF extraction
    inserts stray spaces inside identifiers ("NCL/CMPT/ 74926",
    "POD1 I/10421"), which breaks any regex that tries to parse the full ID.
    The serial is what distinguishes one circular from another, and it's what
    a person actually types.

    The header check guards against a mislabelled filename -- a missing
    exact-ID query costs one data point, a wrong one corrupts a measurement.
    """
    serial = doc_id.rsplit("-", 1)[-1]
    if re.search(rf"(?<![0-9]){re.escape(serial)}(?![0-9])", text[:3000]):
        return serial
    return None


def make_exact_id_query(doc_id: str, text: str, q: dict) -> dict | None:
    serial = own_serial(doc_id, text)
    if serial is None:
        return None
    issuer = doc_id.split("-")[0]          # SEBI or NSE
    stripped = re.sub(r"^(what|when|who|which|how)\s+(is|are|does|do|must|should)\s+",
                      "", q["question"], flags=re.I).rstrip("?")
    return {**q, "question": f"{issuer} circular {serial} {stripped}",
            "query_type": "exact_id"}


def main() -> None:
    docs = sorted(DOCS_DIR.glob("*.txt"))
    if not docs:
        sys.exit(f"No documents in {DOCS_DIR}. Run extract.py first.")

    c = client()
    records: list[dict] = []
    qid = 0
    dropped = Counter()

    for n, doc in enumerate(docs, start=1):
        text = doc.read_text(encoding="utf-8")
        print(f"[{n}/{len(docs)}] {doc.stem}", end=" ", flush=True)

        try:
            generated = ask(c, GENERATE_PROMPT.format(
                n=PER_DOC, doc_id=doc.stem, text=text))["questions"]
        except Exception as e:
            print(f"-> generation failed: {type(e).__name__}")
            dropped["generation_failed"] += PER_DOC
            continue

        # Split: keep most as natural phrasing, rewrite one into search style
        natural = generated[:-1] if len(generated) > 1 else generated
        to_rewrite = generated[-1:] if len(generated) > 1 else []

        typed = [{**q, "query_type": "natural"} for q in natural]

        if to_rewrite:
            try:
                rewritten = ask(c, REWRITE_PROMPT.format(
                    questions=json.dumps([q["question"] for q in to_rewrite])))["rewritten"]
                typed += [{**q, "question": r, "query_type": "keyword"}
                          for q, r in zip(to_rewrite, rewritten)]
            except Exception:
                typed += [{**q, "query_type": "natural"} for q in to_rewrite]

        if natural:
            eid = make_exact_id_query(doc.stem, text, natural[0])
            if eid:
                typed.append(eid)

        kept = 0
        for q in typed:
            span = locate(text, q.get("answer_passage", ""))
            if span is None:
                dropped["span_not_found"] += 1
                continue
            if span[0] < HEADER_CHARS:
                dropped["span_in_header"] += 1
                continue
            qid += 1
            kept += 1
            records.append({
                "query_id": qid,
                "query": q["question"],
                "query_type": q["query_type"],
                "query_source": "llm_generated",
                "verified": False,
                "gold_spans": [{"doc_id": doc.stem, "start": span[0], "end": span[1]}],
                "require": "all",
                "answer_passage": q["answer_passage"],
                "reference_answer": None,
            })
        print(f"-> {kept} kept")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_type = Counter(r["query_type"] for r in records)
    print(f"\nwrote {len(records)} candidate queries -> {OUT_PATH}")
    print(f"  by type: {dict(by_type)}")
    if dropped:
        print(f"  dropped: {dict(dropped)}")
    print("\nNEXT: verify_evalset.py -- nothing counts until verified: true")


if __name__ == "__main__":
    main()
