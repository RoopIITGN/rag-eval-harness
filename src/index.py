"""Embed chunks and upload them to their Azure AI Search index.

Batching matters twice here: the embeddings API accepts a list per call, and
the search client accepts a batch of documents per upload. At 10K TPM a
one-chunk-per-call loop would take hours and trip rate limits.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EMBED_BATCH = 32        # chunks per embeddings call
UPLOAD_BATCH = 200      # documents per search upload


def embed_batch(client: AzureOpenAI, texts: list[str], deployment: str) -> list[list[float]]:
    """Embed with retry. Honour the server's Retry-After rather than guessing."""
    from openai import RateLimitError

    for attempt in range(6):
        try:
            resp = client.embeddings.create(input=texts, model=deployment)
            return [d.embedding for d in resp.data]
        except RateLimitError as e:
            if attempt == 5:
                raise
            wait = 60          # Azure's 429 asks for 60s on this tier
            hdrs = getattr(getattr(e, "response", None), "headers", {}) or {}
            if "retry-after" in hdrs:
                wait = int(hdrs["retry-after"])
            print(f"    rate limited, waiting {wait}s (attempt {attempt + 1}/6)")
            time.sleep(wait)
        except Exception as e:
            if attempt == 5:
                raise
            wait = 2 ** attempt
            print(f"    retry in {wait}s ({type(e).__name__})")
            time.sleep(wait)


def index_config(config: str, aoai: AzureOpenAI, deployment: str) -> None:
    path = ROOT / f"data/chunks_{config}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} - run chunk.py first")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    print(f"\n{config}: {len(rows)} chunks")

    # --- embed -------------------------------------------------------
    vectors: list[list[float]] = []
    t0 = time.time()
    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i:i + EMBED_BATCH]
        vectors.extend(embed_batch(aoai, [r["content"] for r in batch], deployment))
        print(f"  embedded {len(vectors)}/{len(rows)}", end="\r")
    print(f"  embedded {len(vectors)}/{len(rows)} in {time.time() - t0:.0f}s")

    assert len(vectors) == len(rows), "embedding count mismatch"
    assert len(vectors[0]) == 1536, f"unexpected dimension {len(vectors[0])}"

    # --- upload ------------------------------------------------------
    docs = [{
        "id": r["id"],
        "doc_id": r["doc_id"],
        "content": r["content"],
        "char_start": r["char_start"],
        "char_end": r["char_end"],
        "token_count": r["token_count"],
        "content_vector": v,
    } for r, v in zip(rows, vectors)]

    search = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=f"circulars-{config}",
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )

    uploaded = 0
    for i in range(0, len(docs), UPLOAD_BATCH):
        result = search.upload_documents(documents=docs[i:i + UPLOAD_BATCH])
        failed = [r for r in result if not r.succeeded]
        if failed:
            sys.exit(f"upload failed for {len(failed)} docs, first: {failed[0].error_message}")
        uploaded += len(result)
        print(f"  uploaded {uploaded}/{len(docs)}", end="\r")
    print(f"  uploaded {uploaded}/{len(docs)}")


def main():
    aoai = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    for config in ["256-fixed", "512-fixed", "512-recursive"]:
        index_config(config, aoai, deployment)

    print("\nall three indexes populated")


if __name__ == "__main__":
    main()
