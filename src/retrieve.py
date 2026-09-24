"""Retrieval against the three indexes, in four modes.

The four modes are the ablation: keyword-only, vector-only, hybrid with RRF,
and hybrid plus a cross-encoder rerank. Every mode returns the same shape, so
the eval code doesn't care which produced a result.

Design notes:

- Query embeddings are cached. Across 3 configs x 4 modes that's 12 retrievals
  per query, all needing the same vector. Embedding once per query instead of
  twelve times cuts the slowest part of the loop by 90%.

- Retrieval always fetches 20 and returns the full ranked list. recall@5 is the
  same retrieval with a smaller cutoff, computed offline. Never re-query to get
  a different k.

- Azure applies RRF automatically when a query carries both search_text and
  vector_queries. k is fixed at 60 in the managed implementation and is not
  configurable.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

POOL = 20            # candidates retrieved and reranked; see README on why not 50
SELECT = ["id", "doc_id", "char_start", "char_end", "token_count", "content"]

_aoai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
_clients: dict[str, SearchClient] = {}
_reranker = None


@lru_cache(maxsize=4096)
def embed(text: str) -> tuple[float, ...]:
    """Cached so one query embeds once, not once per config-mode pair.

    Retries on transient failures: a long evaluation run should survive a DNS
    blip or a dropped connection rather than losing everything computed so far.
    """
    import time
    for attempt in range(5):
        try:
            return tuple(_aoai.embeddings.create(
                input=[text], model=_deployment).data[0].embedding)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def client(config: str) -> SearchClient:
    if config not in _clients:
        _clients[config] = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=f"circulars-{config}",
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
        )
    return _clients[config]


def reranker():
    """Lazy-loaded: only the rerank mode pays the model load cost."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return _reranker


def retrieve(query: str, config: str, mode: str, top: int = POOL) -> list[dict]:
    """Return a ranked list. mode: bm25 | dense | hybrid | hybrid+rerank."""
    kwargs: dict = {"top": top, "select": SELECT}

    if mode in ("dense", "hybrid", "hybrid+rerank"):
        kwargs["vector_queries"] = [VectorizedQuery(
            vector=list(embed(query)), k_nearest_neighbors=top,
            fields="content_vector")]

    kwargs["search_text"] = query if mode in ("bm25", "hybrid", "hybrid+rerank") else None

    results = [{
        "rank": i,
        "id": r["id"],
        "doc_id": r["doc_id"],
        "char_start": r["char_start"],
        "char_end": r["char_end"],
        "token_count": r["token_count"],
        "content": r["content"],
        "search_score": r["@search.score"],
    } for i, r in enumerate(client(config).search(**kwargs), start=1)]

    if mode == "hybrid+rerank" and results:
        pairs = [(query, r["content"]) for r in results]
        scores = reranker().predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda r: r["rerank_score"], reverse=True)
        for i, r in enumerate(results, start=1):
            r["rank"] = i

    return results


if __name__ == "__main__":
    q = "margin shortfall reporting deadline"
    for mode in ["bm25", "dense", "hybrid", "hybrid+rerank"]:
        rows = retrieve(q, "512-recursive", mode, top=5)
        print(f"\n=== {mode} ===")
        for r in rows:
            score = r.get("rerank_score", r["search_score"])
            print(f"  {r['rank']}. {score:>8.4f}  {r['doc_id']:20s} "
                  f"{r['content'][:60].replace(chr(10), ' ')}")
