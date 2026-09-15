# rag-eval-harness

Hybrid retrieval over Indian securities-market regulatory circulars, with an
evaluation harness that scores retrieval and generation independently.

**Status: in progress.** Building in the open — see commit history.

## What this will be

- ~47 public SEBI and NSE circulars, chunked three ways
- Hybrid BM25 + dense retrieval with RRF fusion on Azure AI Search
- Cross-encoder reranking over the candidate pool
- An eval set labelled by character span, so gold labels survive re-chunking
- Retrieval and generation measured separately, with paired significance tests

## Why span-level labels

Gold labels point at a character range in the source document, not a chunk ID.
A chunk ID is only valid for one chunking configuration — re-chunk and every
label is meaningless. A character span points at the document, which doesn't
change. Label once, evaluate every configuration.
