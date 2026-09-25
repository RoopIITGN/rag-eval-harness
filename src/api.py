"""
HTTP service over the pipeline.

The response shape is the point. A refusal is a normal 200 response with
`answered: false` and a reason, not an error: refusing is a correct outcome, and
an API that signals it as a failure invites callers to retry or to fall back to
an ungrounded answer. Citations are returned as structured objects carrying the
document and character offsets, so a caller can link straight into the source
text rather than trusting a rendered footnote.

Nothing here decides anything. Retrieval configuration, the citation schema and
the refusal gates all live in generate.py, which is what the evaluation harness
measures -- so the numbers in reports/ describe this endpoint, not an
approximation of it.

Run
  PYTHONPATH=src uvicorn api:app --reload --port 8000
  curl -s localhost:8000/healthz
  curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \\
       -d '{"question":"What is the current deadline for enrolling with PaRRVA?"}'
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

import generate

app = FastAPI(
    title="rag-eval-harness",
    description="Hybrid retrieval over Indian securities-market circulars, "
                "with citations verified in code.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source_quote: str
    char_start: int
    char_end: int


class AskResponse(BaseModel):
    question: str
    answered: bool
    answer: str = ""
    refusal_reason: str | None = None
    citations: list[Citation] = []
    retrieved: list[str] = []
    elapsed_ms: int = 0


@app.get("/healthz")
def healthz() -> dict:
    """Liveness plus configuration, so a deployed instance can be identified.

    Deliberately reports which deployments are configured, not whether they
    answer: a health check that calls a reasoning model is slow, costs money on
    every probe, and fails for reasons unrelated to this process being alive.
    """
    required = ("AZURE_SEARCH_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_GENERATOR_DEPLOYMENT")
    missing = [k for k in required if not os.environ.get(k)]
    return {
        "status": "ok" if not missing else "misconfigured",
        "missing_env": missing,
        "config": generate.CONFIG,
        "mode": generate.MODE,
        "top_k": generate.TOP_K,
        "generator": os.environ.get("AZURE_OPENAI_GENERATOR_DEPLOYMENT", ""),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    r = generate.answer(req.question)
    return AskResponse(
        question=r.query,
        answered=r.answered,
        answer=r.answer,
        refusal_reason=r.refusal_reason,
        citations=[Citation(**c) for c in r.citations],
        retrieved=[c["doc_id"] for c in r.chunks],
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )
