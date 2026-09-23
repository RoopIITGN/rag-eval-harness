# rag-eval-harness

Hybrid retrieval over Indian securities-market regulatory circulars, with an
evaluation harness that scores retrieval and generation independently.

Built on 47 public SEBI and NSE circulars. Azure AI Search for hybrid BM25 +
dense retrieval with RRF fusion, a local cross-encoder reranker, and an
evaluation layer that measures each stage separately so failures can be
attributed to the right one.

> **What this is.** A from-scratch, public reproduction of a retrieval
> architecture I built in production at Garuda Yashas Capital, rebuilt here on
> public documents with an LLM-generated, human-verified evaluation set. The
> numbers below are from this repo and differ from the production system —
> different corpus, different eval set, smaller scale. The architecture,
> methodology and failure analysis are the same.

**Status: in progress.** Retrieval is built and indexed. In the evaluation set,
every supersession chain has been reviewed and the remaining generated
questions are under review. Generation and the refusal gates are next. Building
in the open — see commit history.

---

## Corpus

47 public circulars — 25 SEBI, 22 NSE — selected for structural variety rather
than recency: substantive circulars carrying obligations and thresholds,
amendment circulars that reference and modify earlier ones, and table-heavy
operational notices.

| | |
|---|---|
| Documents | 47 |
| Total | 91,587 tokens |
| Median document | 959 tokens |
| Smallest / largest | 415 / 16,425 tokens |
| Chunks at 512 tokens (64 overlap) | 224–225 |

The whole corpus fits in a long-context window, so retrieval isn't strictly
necessary for answering. It's here for cost — retrieval is paid once at index
time rather than on every query — and for clause-level citation, which a stuffed
prompt can't provide.

At ~225 chunks the index is small enough that approximate nearest-neighbour
search buys nothing measurable, so exhaustive KNN is used throughout. The
reranking pool is 20 rather than the more common 50: at this corpus size, 50
candidates is nearly a quarter of the entire index, which leaves the reranker
little to discriminate between.

The amendment circulars are there deliberately. They form chains in which a
later circular extends, modifies, replaces or completes an earlier one — see
[Supersession](#supersession). NSE circulars are substantially tabular, which is
the harder chunking case.

`data/corpus_manifest.txt` lists the document IDs. PDFs are not redistributed;
`data/documents/` is committed deliberately — those files are part of the
dataset, not a build artefact.

### Two ingestion faults, and what caught them

**A scanned PDF.** `extract.py` runs quality checks on every extraction —
character count, alphabetic ratio, presence of a circular identifier — and
flagged one document as an image rather than text. It was replaced rather than
OCR'd: OCR errors would be baked into the frozen text, and every gold-span
offset is a character position in those exact bytes. See
`reports/extraction_log.txt`.

**A mislabelled file.** One circular was saved under the serial number it
*cited* rather than its own, so `SEBI-10557.txt` held circular 18038 — a
duplicate, with the real 10557 missing from the corpus. Nothing in the pipeline
noticed, because a document's identity came from a filename a person typed. The
citation graph eventually exposed it: two circulars citing each other with the
same date is impossible in real regulatory data.

The fix is `validate_corpus.py`, which now runs after every extraction and exits
non-zero on failure. It checks that each file's serial appears in the document's
own header — before its subject line, where a circular states its own
identity — rather than merely somewhere in the text; that no two files have
identical content; and it reports near-duplicates for inspection. The four eval
queries generated from the duplicate were rejected with that reason recorded,
and the real circular was added, which turned out to form a genuine supersession
pair with 18038.

---

## Chunking configurations

Three configurations are built from the frozen text, each with character offsets
verified against the source before indexing.

| Configuration | Chunks | Mean tokens |
|---|---|---|
| 256-fixed | 428 | 242 |
| 512-fixed | 224 | 459 |
| 512-recursive | 225 | 441 |

`512-fixed` and `512-recursive` produce near-identical chunk counts at similar
mean length, so comparing them isolates **where boundaries fall** rather than how
much text each chunk holds. Fixed splitting cuts at a token count regardless of
structure; recursive splitting tries paragraph breaks first, then line breaks,
then sentences, falling back to a hard cut only when nothing else fits.

Every chunk is checked with `verify()` before use: `text[char_start:char_end]`
must equal the chunk exactly, and the chunks must cover every character of the
document. An offset bug produces no error downstream — it silently shifts every
gold-span match, so the assertion runs at build time instead.

---

## Results

Generated by `src/run_ablation.py` into [`reports/ablation.md`](reports/ablation.md).

| Configuration | recall@5 | 95% CI | recall@20 |
|---|---|---|---|
| BM25 only | — | — | — |
| Dense only | — | — | — |
| Hybrid (RRF) | — | — | — |
| Hybrid + cross-encoder rerank | — | — | — |

_This table is produced by the ablation run, not written by hand._

---

## The evaluation set

### How it was generated

The obvious way to build a retrieval eval set — show an LLM a chunk, ask it to
write a question — produces questions that are semantic paraphrases of the
passage that answers them. Dense retrieval then looks excellent and BM25 looks
useless, because the generator deliberately avoided lexical overlap. **Any
hybrid-vs-dense ablation run on such a set measures the question generator, not
the retrieval system.**

So `src/generate_evalset.py` does four things differently:

1. Generates from the **full document**, not a single chunk, so the question
   isn't anchored to one passage's wording.
2. Rewrites a share of questions into **search-box phrasing** — terse,
   keyword-shaped, the way a person actually types.
3. Deliberately includes **exact-identifier queries** (`SEBI circular 18038 …`),
   the class where dense retrieval fails and BM25 wins. Circular numbers carry
   almost no semantic signal, so two different serials embed to nearly the same
   point. Omitting these guarantees a misleading ablation. They're built from the
   document's own serial, confirmed present in its header, and rewritten by hand
   into natural phrasing during review.
4. Tags every query by type, so recall is reported **by subgroup** rather than
   blended.

Generation used a reasoning model at its default temperature — deliberately.
At temperature zero a model converges on the single most obvious question per
document, and an eval set of obvious questions flatters retrieval.
Reproducibility comes from committing and reviewing the output, not from
sampling settings.

188 candidates were attempted across the corpus. 12 were dropped because the
quoted answer passage couldn't be located verbatim in the source text, and 7
because the answer fell inside the letterhead, leaving **169 generated
candidates**.

### How it was reviewed

Every record is written with `verified: false` and counts for nothing until a
person confirms it. `verify_evalset.py` shows each query with its gold span
highlighted in context, flags any query whose source circular has a related or
newer one, and records one decision:

| Decision | Used when |
|---|---|
| Accept | the highlighted text answers the query |
| Edit | the span is right, the query is vague — rewritten, original kept |
| Widen | another document answers it equally — second span, `require: "any"` |
| Re-span | the query is right, the span is wrong — re-pointed, original kept |
| Replace | the query is poor — rejected, and a new record authored in its place |
| Reject | boilerplate, ambiguous across documents, wrong span, or trivial |

Rejected records stay in the file with a reason, so the rejection rate is
reportable rather than silently discarded. Re-reviewing a decided record moves
the earlier decision into `review_history` rather than overwriting it. Every
record carries `query_source`, so the mix of generated and designed queries is
visible.

Four patterns account for most of what review caught in the generated set:

- **References to an unseen document.** *"When does this circular come into
  effect?"* — written by a model looking at one circular, meaningless to a user
  who isn't. Rewritten, or rejected where several circulars would answer.
- **Boilerplate.** Contact blocks, instructions to load contract files,
  directions to market infrastructure institutions, statements of legal
  authority — text that recurs almost verbatim across circulars and has no
  single right answer. Rejected.
- **Ambiguity across a chain.** *"When does the revised list take effect?"* has
  three correct answers in a chain of three revisions. Rejected, or rewritten to
  name the circular it means.
- **Stale answers.** A question generated from a circular that a later one
  changed carries the old answer as ground truth. Re-pointed to the current
  circular. See [Supersession](#supersession).

| | Count |
|---|---|
| Generated candidates | 169 |
| Designed supersession queries | 32 |
| Held out — generation only | 3 |
| Accepted / rejected | _from `verify_evalset.py --stats` when review completes_ |

### Gold labels are character spans, not chunk IDs

```json
{"query_id": 1015,
 "query": "Until when can investment advisers keep showing certified past performance data to clients?",
 "gold_spans": [{"doc_id": "SEBI-10557", "start": 2856, "end": 2980}],
 "require": "all"}
```

A chunk ID is only valid for one chunking configuration — re-chunk at 256 tokens
and every label is meaningless. A character span points at the **source
document**, which doesn't change. Label once, evaluate every chunking
configuration forever. `require` is `"all"` when every span must be retrieved
(multi-hop) and `"any"` when any one suffices (the same fact stated in two
places).

Every offset is relative to the exact bytes in `data/documents/`. Re-extracting
with a different PDF parser shifts whitespace and silently invalidates the
entire goldset — nothing errors, the numbers just quietly become wrong.

---

## Supersession

Regulatory circulars amend each other. Single-document generation can't see
that, so a query generated from a circular that was later changed carries a
**stale answer as ground truth** — and a retriever that correctly returns the
newer circular is scored as a miss. The eval would penalise correct behaviour.

`build_crossrefs.py` finds related circulars two ways, because each misses what
the other catches:

- **Citations.** Circular A cites B if B's serial appears in A's text. This
  catches SEBI extensions and amendments, which name the circular they modify.
- **Shared subject lines.** This catches chains whose linking circular isn't in
  the corpus. In the collateral-timings chain, two circulars each cite one we
  don't have, so no citation connects them — but all three share a subject.

Each circular's date is read from its header, and the review screen names the
newest member of any family, warning when a query's gold span sits in an older
one. That's fine only when the query names the older circular explicitly.

Across the corpus this found nine citations and two subject families, making
six chains. They change in different ways, and each breaks retrieval
differently:

| Chain | Kind of change | What goes wrong |
|---|---|---|
| Collateral timings · 3 NSE | **Partial modification** | Current state lives in two documents. Two cut-offs moved 8 PM → 9 PM → 8 PM, so the oldest circular gives the right number from a superseded source and the middle one gives the wrong number |
| Cross-margin ETF lists · 3 NSE | **Full replacement** | The newest list governs everything. ETFs were removed and symbols renamed; the old symbol survives only inside the new scheme name, luring keyword search to the stale list |
| PaRRVA enrolment · 2 SEBI | **Extension** | The deadline moved a month. The extension restates the old date, so even the right document carries the stale answer |
| ETF price bands · 2 SEBI | **Extension of start date** | Only the date moved, so the older circular's content is still current — but it also describes the regime it replaced, so the stale answer sits inside the correct document |
| Mutual fund borrowing · 3 SEBI | **Deferral, then partial supersession** | The last circular supersedes only the intraday half of the first; the other half still governs. It was issued in July but took effect in September, so in August the older rules were in force despite a newer circular existing |
| F&O launch · 2 NSE | **Deferred detail** | The first circular defers lot sizes to a later one. Retrieving it yields "not yet announced" — not wrong when written, useless now |

**32 designed queries** (IDs 1001–1032) test these chains directly, each with a
recorded `stale_trap` — the document that gives a wrong, outdated or unsourced
answer if retrieved instead. They include *silent* cases, where the newer
circular never mentions a fact so the older one still governs; *flips*, where the
answer changes from no to yes; *point-in-time* queries, where naming the older
circular makes its answer the correct one; and multi-hop queries whose answer
needs both circulars.

Span-level scoring matters here. When a value is unchanged across circulars, the
stale document gives the right number — but a compliance answer citing a
superseded circular is still wrong. Span scoring catches that; an answer-only
judge would not.

**Removal questions are held out.** "Is this ETF still eligible?" has an answer
that is an *absence* from the current list, which no span can mark. Recording
such a query with no gold spans would score every retrieval as a hit. They live
in `data/generation_only.jsonl`, for generation evaluation against a reference
answer.

---

## Table twins

`extract.py` writes each table twice — the page's flattened text, and labelled
rows with column names attached — so a table-row answer exists in two places in
the same document. A gold span naming only one would score a retriever that
returns the other as a miss. Because the two forms sit ~150 tokens apart, that
error would fall mostly on the 256-token configuration and bias the chunking
comparison.

`add_table_twins.py` finds the second form and adds it as an alternative span
(`require: "any"`). It matches on values with column labels stripped, across
one- and two-line windows — the ETF lists' column header contains a line break,
so every labelled row wraps. Multi-span records are left alone, since a flat span
list can't express "either form of span *k*".

---

## Metrics, and why these ones

**recall@5** — what the generator actually sees. The headline, because a
retrieval miss is unrecoverable: no reranker and no prompt can surface a chunk
that was never retrieved.

**recall@20** — the reranker's ceiling. The gap between @5 and @20 is the
maximum available gain from reranking, and it separates "the retriever can't
find it" from "the retriever found it and ranked it badly." Those need different
fixes and look identical on recall@5 alone.

**Recall at a fixed token budget** — the fair comparison across chunk sizes.
Top-5 of 512-token chunks is twice the context of top-5 of 256-token chunks, so
fixed-k quietly favours larger chunks.

**Recall by query type** — blended recall averages away the signal that
justifies hybrid retrieval in the first place, and the supersession slice is
reported separately for the same reason.

**McNemar's test on paired outcomes** — comparing two configurations on the same
queries means most queries carry no information. Conditioning on the discordant
pairs is more sensitive than comparing two proportions, because it removes the
variance from query difficulty, which is shared.

**Bootstrap CIs rather than a holdout** — at this sample size, splitting would
leave both halves too small to measure or compare anything.

---

## Architecture

```
query
  ├── BM25                                              ─┐
  ├── dense (exhaustive KNN, text-embedding-3-small)    ─┴─ RRF fusion
  │                                                        (Azure AI Search, native)
  ├── cross-encoder rerank over top-20        (sentence-transformers, local)
  │
  │   ── built above this line · in progress below ──
  │
  ├── GATE: retrieval-score threshold         → refuse, no LLM call
  ├── generation with forced citation schema  (constrained decoding)
  └── GATE: citation + quote verification     → refuse if unsupported
```

Refusal is to be enforced in code before the generator is invoked, not
requested in the prompt. Citation validity is a set-membership check and quote
verification a substring match — both deterministic, both free, both run before
anything reaches the user.

The search indexes hold nothing that isn't reproducible from the repo: chunk
text and offsets are committed, the schema is code, and vectors regenerate in
about ninety seconds. Rebuilding on a fresh Azure service is `create_indexes.py`
followed by `index.py`.

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

**Python 3.12.** Newer versions may lack prebuilt wheels for some dependencies.

**PyTorch installs the CPU build.** `requirements.txt` puts the PyTorch CPU index
first so pip resolves torch from there rather than PyPI — the default PyPI wheel
bundles CUDA (~2.5GB unpacked) and will exhaust memory on a constrained WSL
setup. No GPU is used anywhere in this project.

```bash
python -c "import torch; print('cuda bundled:', torch.version.cuda)"   # → None
```

**Two dependency files, different jobs.** `requirements.txt` is the spec — loose
version floors plus the index configuration; install from this.
`requirements.lock.txt` is exact pinned versions from a verified clean install,
for reproducing results when something doesn't match.

Environment (`.env`, gitignored — see `.env.example`):

```
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
AZURE_SEARCH_KEY=<admin key>

AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_KEY=<key>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-10-21
```

`AZURE_OPENAI_ENDPOINT` is the bare host. The Foundry portal also surfaces an
`/openai/v1` path for use with the OpenAI-compatible client; the `AzureOpenAI`
client appends its own routing and will return 404 if that suffix is included.

`gpt-5-mini` is a reasoning model: it rejects any `temperature` other than the
default, and spends output tokens on reasoning before producing visible text, so
calls use `max_completion_tokens` with generous headroom.

**Two regions, deliberately.** Azure AI Search runs in Central India; the model
deployments are in Sweden Central, where they're available for the deployment
types this project uses. The search service stays local because that's where
query latency is measured and where the data lives.

Cross-region embedding adds roughly 250ms per query. That cost lands on indexing
(paid once, for ~877 chunks across three configurations) and on evaluation runs,
but not on reported retrieval latency, which is measured inside the search
service after the query vector arrives.

**Embedding throughput.** `index.py` embeds roughly 275K tokens. Azure OpenAI
Standard deployments bill per token consumed rather than per token of
provisioned capacity, so the tokens-per-minute setting affects run time, not
cost — provision generously. On a low TPM deployment the API returns 429 with a
60-second `Retry-After`; `index.py` handles rate limits separately from transient
errors and honours the server's hint rather than backing off exponentially.

---

## Pipeline

Scripts import from `src/` but resolve data paths relative to the repo root, so
run them from the root with `PYTHONPATH=src`:

```bash
# corpus
PYTHONPATH=src python src/extract.py                 # PDFs → frozen text, table-aware
PYTHONPATH=src python src/validate_corpus.py         # identity + duplicate gate; exits 1 on failure
PYTHONPATH=src python src/chunk.py                   # three chunk sets, offsets verified
PYTHONPATH=src python src/create_indexes.py          # three indexes, exhaustive KNN
PYTHONPATH=src python src/index.py                   # embed chunks and upload

# eval set
PYTHONPATH=src python src/generate_evalset.py        # draft candidates -- run once; overwrites goldset.jsonl
PYTHONPATH=src python src/build_crossrefs.py         # related circulars, dated, for the review screen
PYTHONPATH=src python src/verify_evalset.py          # human review (--ids, --from, --list, --stats)
PYTHONPATH=src python src/add_table_twins.py         # second span for table answers (--apply)
PYTHONPATH=src python src/fix_respanned_passages.py  # one-off repair, safe to re-run

# results
PYTHONPATH=src python src/run_ablation.py            # → reports/ablation.md
```

The designed queries live in `data/supersession_slice*.jsonl`, one file per
chain, and are appended to `goldset.jsonl` for review.

---

## Known limitations

- **Eval set provenance.** Most questions are LLM-generated and human-reviewed;
  the supersession slice is designed. None are drawn from real user logs. Each
  record carries `query_source` so the proportion is visible rather than implied.
- **Sample size.** At the current set size each query moves recall by under one
  percentage point. Differences smaller than a few queries are not
  distinguishable from noise, which is why configuration comparisons use paired
  tests rather than raw deltas.
- **Corpus scale.** See Corpus — at ~225 chunks, approximate nearest-neighbour
  search and a large reranking pool both stop earning their cost.
- **Single-document generation.** The generator sees one circular at a time, so
  it produces no multi-hop queries and can't see supersession. Both are covered
  only by the 32 designed queries, which makes them a small slice to draw
  conclusions from.
- **Supersession detection.** Citation matching skips serials under four digits,
  which match regulation numbers and amounts too often. Subject-line matching
  catches only families whose circulars share an exact subject — true of
  recurring NSE notices, rarely of SEBI circulars.
- **Issue date is not effective date.** `build_crossrefs.py` orders related
  circulars by the date in their header, which is when they were issued. A
  circular can be issued well before it takes effect — in the mutual fund
  borrowing chain, the newest was issued in July and took effect in September —
  so "newest" is a prompt for review, not a statement of which rule is in force
  on a given date.
- **Removal questions.** Queries whose answer is an absence can't be expressed
  as spans, so they're excluded from retrieval evaluation.
- **Duplicated tables.** Emitting both forms of every table was meant to help
  retrieval, but it duplicates content and complicates labelling. A cleaner
  extraction would keep only the labelled rows; it wasn't changed because every
  gold-span offset is pinned to the current text.
- **Table headers on continuation pages.** When a table continues onto a new
  page, `extract.py` takes that page's first data row as the header, so labelled
  rows there carry another row's values as labels. Values are unaffected and twin
  detection still pairs them, but in one case the symbol cell was lost entirely.
- **Tables inserted at page ends.** Tables are written after each page's text,
  so a table on a page where a paragraph continues onto the next lands mid
  sentence. Address blocks are occasionally mistaken for tables, making this
  more common than it should be.
- **Annexures outside the corpus text.** Some circulars refer to annexures, such
  as strike-price schemes, that didn't extract with the body. Questions whose
  answer lives there can point only at the reference, not the content.
- **Extraction artifacts inside identifiers.** PDF extraction occasionally
  inserts spaces within circular IDs (`NCL/CMPT/ 74926`). Retrieval is
  unaffected — the analyzer tokenizes on both slashes and whitespace — but it
  breaks any attempt to parse full identifiers from the text, which is why
  exact-ID queries and identity checks work from the document's serial number.
- **No semantic ranker.** Azure AI Search's L2 semantic reranker is not
  available on the Free tier, so reranking runs locally with
  `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU — roughly 50–150ms for a pool of
  20. A managed reranker would move that cost server-side.
- **Cross-region embedding.** End-to-end query latency includes a ~250ms
  network hop that a single-region deployment would not. Reported retrieval
  latency excludes it.
- **Dated API version.** Scripts use the `AzureOpenAI` client with a pinned API
  version. Azure's v1 endpoint removes the need to pin one; migrating is a
  client-construction change in each script.
