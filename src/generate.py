"""
Answer a question from retrieved circulars, with citations that are verified in
code rather than trusted.

The pipeline is retrieve -> generate -> verify, and the verification is the
point. A model asked politely for citations will produce citations; whether they
refer to text that exists is a separate question, and the only way to know is to
check. Both checks here are deterministic and free: a cited chunk id is either
in the retrieved set or it isn't, and a quoted span either appears in that chunk
or it doesn't. Neither needs a second model.

Three refusal paths, none of them a prompt instruction:

  Gate 1 -- retrieval score threshold. MEASURED AND LEFT OPEN. RRF sums
  reciprocal ranks, so the top score sits near 2/61 whether or not anything
  relevant was retrieved; the distributions for hits and misses overlap almost
  entirely. See src/calibrate_gate.py. The production system gated on a
  cross-encoder score, which is a real relevance estimate on a wide scale; this
  repo doesn't ship the reranker, so that signal isn't available.

  Gate 2 -- sufficient_context. A required boolean in the response schema, not a
  request in the prompt. False refuses regardless of what else came back.

  Gate 3 -- citation and quote verification. Every cited chunk must be one that
  was retrieved, and every quoted span must appear in that chunk. Quotes are
  compared with whitespace normalised, because PDF extraction breaks lines
  inside sentences and an answer shouldn't be rejected for reflowing them.

The schema is enforced with forced tool use rather than asked for in the prompt:
JSON mode constrains syntax, a forced tool call constrains the shape.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

import retrieve

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CONFIG, MODE, TOP_K = "512-recursive", "hybrid", 5
MAX_TOKENS = 16000

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer_with_citations",
        "description": "Answer the question using only the supplied chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "sufficient_context": {
                    "type": "boolean",
                    "description": "True only if the chunks contain the answer. "
                                   "False if they do not, even partially.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer. Empty string if sufficient_context is false.",
                },
                "citations": {
                    "type": "array",
                    "description": "One entry per claim made in the answer.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string",
                                         "description": "Exactly as labelled, e.g. chunk_3"},
                            "source_quote": {"type": "string",
                                             "description": "Verbatim span from that chunk "
                                                            "supporting the claim."},
                        },
                        "required": ["chunk_id", "source_quote"],
                    },
                },
            },
            "required": ["sufficient_context", "answer", "citations"],
        },
    },
}

PROMPT = """Answer the question using ONLY the numbered chunks below. They are \
extracts from Indian securities-market regulatory circulars.

Rules:
- Every claim in your answer must be supported by a quote from a chunk.
- Quote verbatim. Copy the characters exactly; do not paraphrase inside a quote.
- Keep each quote SHORT -- one sentence or clause, at most about 25 words. Quote \
the smallest span that supports the claim.
- A quote must come from ONE chunk. Never join text from two chunks into a \
single quote. Cite twice instead.
- If the chunks do not contain the answer, set sufficient_context to false and \
leave the answer empty. Do not answer from your own knowledge.
- Circulars amend one another. If two chunks disagree, prefer the later circular \
and say which one you relied on.
- State the current position in your FIRST sentence.
- Do not claim that one circular supersedes another unless a chunk says so. You \
may note that an earlier chunk gives a different value, but only describe a \
supersession that the text states.
- The answer is prose only. Do not put quotes, "Supporting quote:" labels or \
circular reference numbers in it -- citations belong in the citations field.

{chunks}

QUESTION: {question}
"""


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def _norm(s: str) -> str:
    """Whitespace-insensitive comparison: extraction breaks lines mid-sentence."""
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass
class Result:
    query: str
    answered: bool
    answer: str = ""
    refusal_reason: str | None = None
    citations: list[dict] = field(default_factory=list)
    bad_citations: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    raw: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def verify(payload: dict, chunks: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Gate 3. Returns (verified citations, rejected citations with a reason)."""
    good, bad = [], []
    for c in payload.get("citations", []):
        cid, quote = c.get("chunk_id", ""), c.get("source_quote", "")
        chunk = chunks.get(cid)
        if chunk is None:
            bad.append({**c, "problem": "cites a chunk that was not retrieved"})
        elif not quote.strip():
            bad.append({**c, "problem": "empty quote"})
        elif _norm(quote) not in _norm(chunk["content"]):
            bad.append({**c, "problem": "quote does not appear in the cited chunk"})
        else:
            # Offsets of the QUOTE in the source document, not of the whole
            # chunk -- a caller linking into the text needs the span it cites.
            # Located on normalised text, so a quote reflowed across a line
            # break still resolves; falls back to the chunk if it cannot.
            start, end = chunk["char_start"], chunk["char_end"]
            i = chunk["content"].find(quote)
            if i != -1:
                start, end = chunk["char_start"] + i, chunk["char_start"] + i + len(quote)
            good.append({**c, "doc_id": chunk["doc_id"],
                         "char_start": start, "char_end": end})
    return good, bad


def answer(question: str, config: str = CONFIG, mode: str = MODE,
           top_k: int = TOP_K, client: AzureOpenAI | None = None) -> Result:
    client = client or _client()
    retrieved = retrieve.retrieve(question, config, mode, top=top_k)[:top_k]
    chunks = {f"chunk_{i}": r for i, r in enumerate(retrieved, start=1)}
    meta = [{"chunk_id": k, "doc_id": r["doc_id"], "char_start": r["char_start"],
             "char_end": r["char_end"]} for k, r in chunks.items()]

    if not retrieved:
        return Result(question, False, refusal_reason="nothing retrieved", chunks=meta)

    block = "\n\n".join(f"[{k}] (circular {r['doc_id']})\n{r['content']}"
                        for k, r in chunks.items())

    resp = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_GENERATOR_DEPLOYMENT"],
        messages=[{"role": "user", "content": PROMPT.format(chunks=block, question=question)}],
        tools=[ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": "answer_with_citations"}},
        max_completion_tokens=MAX_TOKENS,
    )
    calls = resp.choices[0].message.tool_calls
    if not calls:
        return Result(question, False, refusal_reason="model returned no tool call",
                      chunks=meta)
    try:
        payload = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError:
        return Result(question, False, refusal_reason="tool arguments were not valid JSON",
                      chunks=meta)

    # Gate 2 -- the model's own declaration, as a required field.
    if not payload.get("sufficient_context"):
        return Result(question, False, refusal_reason="insufficient context",
                      chunks=meta, raw=payload)

    # Gate 3 -- verification in code.
    good, bad = verify(payload, chunks)
    if not good:
        return Result(question, False, refusal_reason="no citation survived verification",
                      bad_citations=bad, chunks=meta, raw=payload)
    if bad:
        return Result(question, False, refusal_reason="some citations failed verification",
                      citations=good, bad_citations=bad, chunks=meta, raw=payload)

    return Result(question, True, answer=payload.get("answer", ""),
                  citations=good, chunks=meta, raw=payload)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the current deadline for enrolling with PaRRVA?"
    r = answer(q)
    print(f"Q: {r.query}\n")
    if r.answered:
        print(r.answer + "\n")
        for c in r.citations:
            print(f"  [{c['chunk_id']}] {c['doc_id']}: \"{c['source_quote'][:90]}\"")
    else:
        print(f"REFUSED -- {r.refusal_reason}")
        for c in r.bad_citations:
            print(f"  rejected [{c.get('chunk_id')}]: {c['problem']}")
    print(f"\nretrieved: {', '.join(c['doc_id'] for c in r.chunks)}")
