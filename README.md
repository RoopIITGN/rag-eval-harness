# rag-eval-harness

Hybrid retrieval over Indian securities-market regulatory circulars, with an
evaluation harness that scores retrieval and generation independently.

Built on 47 public SEBI and NSE circulars. Azure AI Search for hybrid BM25 +
dense retrieval with RRF fusion, a local cross-encoder reranker, and an
evaluation layer that measures each stage separately so failures can be
attributed to the right one.

> **What this is.** A from-scratch, public reproduction of a retrieval
> architecture I built in production at Garuda Yashas Capital, rebuilt here on
> public documents with an LLM-generated, human-reviewed evaluation set. The
> numbers below are from this repo and differ from the production system —
> different corpus, different eval set, smaller scale. The architecture,
> methodology and failure analysis are the same.

**Status: in progress.** Retrieval is built, indexed and measured against 214
verified queries. Generation and the refusal gates are next. Building in the
open — see commit history.

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

`data/corpus_manifest.txt` lists the document IDs. PDFs are not redistributed;
`data/documents/` is committed deliberately — those files are part of the
dataset, not a build artefact.

### What the corpus actually contains

Not 47 independent documents. Three kinds of relationship, each breaking
retrieval differently, and each found differently.

**Supersession chains — a later circular changes an earlier one**

| Subject | Circulars | What changes | Found by |
|---|---|---|---|
| Collateral timings | NSE-CMPT 72357 → 75321 → 75463 | Partial modification; two cut-offs move 8 PM → 9 PM → 8 PM | subject |
| Cross-margin ETF lists | NSE-CMPT 74926 → 75424 → 75922 | Full replacement; four ETFs dropped, three symbols renamed | citation + subject |
| Index quantity freeze limits | NSE-FAOP 74469 → 74942 → 75521 → 76068 | Monthly reissue; BANKNIFTY 900 → 600, NIFTYFPI added | subject |
| Mutual fund borrowing | SEBI 6961 → 7885 → 16006 | Deferral, then supersession of only the intraday half | citation |
| Intraday position limits | SEBI 41 → 122 | "No penalty until further directions" replaced by a penalised FutEq framework | **neither** |
| PaRRVA enrolment | SEBI 10557 → 18038 | Extension; deadline 3 Aug → 3 Sep 2026 | citation |
| ETF price bands | SEBI 13804 → 19839 | Start date deferred 1 Sep → 7 Sep 2026 | citation |
| Indus Towers dividend | NSE-FAOP 75563 → 75644 | Strikes deferred, then published three days later | subject |
| F&O launch, three stocks | NSE-FAOP 75371 → 75958 | Lot sizes deferred, then published | citation + subject |

**Template families — same wording, different subject, no supersession**

| Subject | Circulars | Why it matters |
|---|---|---|
| F&O exclusions | NSE-FAOP 74363, 74774, 75912 | Identical but for the stock and dates; the eligibility basis silently changed between June and August 2026 |
| F&O introductions | NSE-FAOP 74408, 75239, 75371, 75958 | Two stock batches and one index launch |
| Dividend adjustments | NSE-FAOP 75444, 75563, 75644, 75731 | Four paragraphs identical across all four |
| Cost accountant audits | SEBI 7933, 7934 | Same day, same text, one for advisers and one for research analysts |
| Extension of timeline | SEBI 10421, 13567, 18038, 19839 | Four unrelated regulations, near-identical framing |

**Cross-document dependencies — the answer needs two circulars**

| Question | Circulars | Why |
|---|---|---|
| When do pre-open auction changes go live, and what must members install? | SEBI 2765 + NSE-FAOP 76186 | SEBI sets the date, NSE sets the software |
| Why can index funds borrow for under-executed sell trades only from August 2026? | SEBI 6961 + SEBI 2765 | The rule names no date; the date is in the other circular |

In the families nothing is superseded — but a question that doesn't name its
stock, its intermediary or its regulation has three or four correct answers.
That's a different failure from staleness and needs a different fix: anchoring
the query rather than re-pointing the label.

### Two ingestion faults, and what caught them

**A scanned PDF.** `extract.py` runs quality checks on every extraction —
character count, alphabetic ratio, presence of a circular identifier — and
flagged one document as an image rather than text. It was replaced rather than
OCR'd: OCR errors would be baked into the frozen text, and every gold-span
offset is a character position in those exact bytes. See
`reports/extraction_log.txt`.

**A mislabelled file.** One circular was saved under the serial number it
*cited* rather than its own, so `SEBI-10557.txt` held circular 18038 — a
duplicate, with the real 10557 missing. Nothing in the pipeline noticed, because
a document's identity came from a filename a person typed. The citation graph
exposed it: two circulars citing each other with the same date is impossible in
real regulatory data.

The fix is `validate_corpus.py`, which runs after every extraction and exits
non-zero on failure. It checks that each file's serial appears in the document's
own header — before its subject line, where a circular states its identity —
rather than merely somewhere in the text; that no two files have identical
content; and it reports near-duplicates for inspection. The four eval queries
generated from the duplicate were rejected with that reason recorded, and the
real circular turned out to form a genuine supersession pair with 18038.

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

Generated by `src/run_ablation.py` into [`reports/ablation.md`](reports/ablation.md),
which also reports recall by query type, recall at a fixed token budget, paired
McNemar comparisons, and the queries no configuration retrieves. 214 verified
queries, retrieval pool 20, chunking configuration `512-recursive`.

| Mode | recall@5 | 95% CI | recall@20 |
|---|---|---|---|
| BM25 only | 88.3% | 84.1–92.5% | 99.5% |
| Dense only | 90.7% | 86.4–94.4% | 97.2% |
| Hybrid (RRF) | **93.9%** | 90.7–96.7% | 97.2% |
| Hybrid + cross-encoder rerank | 86.4% | 81.8–90.7% | 97.2% |

Retrieval runs once per query, configuration and mode; the ranked lists are
saved to `reports/runs.jsonl` and every metric is computed from that file.
recall@5 and recall@20 are the same retrieval at two cutoffs, never a second
query. The run file is committed, so the numbers can be recomputed — or argued
with — without Azure credentials.

### Hybrid beats keyword search, but not dense search

| Comparison | won | lost | discordant | p |
|---|---|---|---|---|
| hybrid vs BM25 | 13 | 1 | 14 | **0.002** |
| hybrid vs dense | 11 | 4 | 15 | 0.118 |

Hybrid's 3.2-point lead over dense alone sits inside the noise at this sample
size, so it isn't claimed. The lead over BM25 is real. Reporting the second
result without the first would be the easy mistake here: the headline table
alone makes hybrid look like a clear two-way win.

### Reranking makes retrieval worse

Cross-encoder reranking costs 7.5 points of recall@5 — 19 queries lost against
3 gained, p = 0.001. The same direction holds at `512-fixed`. It helps only at
`256-fixed`, and barely.

The subgroup table shows where it goes wrong: multi-hop falls from 79.2% to
62.5% and exact-identifier queries from 93.0% to 88.4%.
`cross-encoder/ms-marco-MiniLM-L-6-v2` is trained on web passages and has no
purchase on a circular serial or a table row. With a pool of 20 and recall@20
already at 97.2%, its only job is ordering — and it does that worse than RRF.

So reranking is measured and reported, not shipped. A managed reranker trained
on this kind of text, or a larger pool where there is more to reorder, might
change that; this one doesn't earn its place.

### Chunk size wins at fixed k, and loses at equal tokens

| Configuration | recall@5 (hybrid) | 1280 tokens | 2560 tokens |
|---|---|---|---|
| 256-fixed | 82.2% | **83.2%** | **90.7%** |
| 512-fixed | 92.5% | 74.8% | 89.3% |
| 512-recursive | 93.9% | 72.9% | 87.9% |

At fixed k, 512-token chunks beat 256 by 11 points (p = 0.007) — but top-5 of
512-token chunks is twice the context of top-5 of 256-token chunks. Given the
same token budget the ranking reverses, and the smaller chunks win by 10 points
at 1280 tokens, because they place the answer more precisely.

Which is right depends on the constraint. If the generator's context is the
scarce resource, 256 is the better configuration. If latency and call count
dominate, 512 is. A single recall@5 number would have hidden the trade entirely.

**Recursive splitting shows no advantage over fixed splitting at the same size**
— 7 queries against 2, p = 0.180. The gain is chunk size, not the separator
hierarchy.

### recall@5 by query type — 512-recursive

| Mode | exact_id (43) | keyword (35) | natural (84) | supersession (28) | multi_hop (24) |
|---|---|---|---|---|---|
| bm25 | 86.0% | 97.1% | 95.2% | 85.7% | 58.3% |
| dense | 93.0% | 94.3% | 95.2% | 82.1% | 75.0% |
| hybrid | **93.0%** | **100.0%** | **97.6%** | **89.3%** | **79.2%** |
| hybrid+rerank | 88.4% | 94.3% | 90.5% | 82.1% | 62.5% |

**Dense beats BM25 on exact-identifier queries**, 93.0% against 86.0% — the
opposite of the premise these queries were built on. The review process is why:
the generator's `SEBI circular 18038 What extended deadline has SEBI set…` was
rewritten by hand into `What new PaRRVA enrolment deadline did SEBI circular
18038 set?`. That carries a serial *and* semantic content, so dense handles it.
The claim that identifiers are where keyword search wins doesn't survive
realistic phrasing at this corpus size.

Multi-hop is the hardest slice everywhere, which is expected: `require: "all"`
means every span must be retrieved, and a query needing two circulars fails if
either is missed.

### One thing fusion costs

BM25 reaches 99.5% at recall@20; hybrid reaches 97.2%. Fusing two rankings into
a fixed pool of 20 drops documents BM25 had at ranks 15–20. Fusion is not free
at the pool boundary — worth revisiting with a larger pool.

Every one of the 214 verified queries is retrieved by at least one
configuration, so nothing in the eval set is unreachable.

---

---

## The evaluation set

214 verified queries. 142 came from an LLM generator and human review; 72 were
designed by hand to cover supersession, multi-hop and in-document traps that
single-document generation cannot produce.

| | |
|---|---|
| Generated candidates | 169 |
| Designed queries | 72 |
| **Accepted** | **214** |
| Rejected | 27 |
| Held out for generation evaluation only | 3 |

By type: natural 84, keyword 35, exact_id 43, supersession 28, multi_hop 24.

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
3. Deliberately includes **exact-identifier queries**, the class where dense
   retrieval fails and BM25 wins. Circular numbers carry almost no semantic
   signal, so two serials embed to nearly the same point. Omitting these
   guarantees a misleading ablation. They're built from the document's own
   serial, confirmed present in its header, and rewritten by hand into natural
   phrasing during review.
4. Tags every query by type, so recall is reported **by subgroup** rather than
   blended.

Generation used a reasoning model at its default temperature, deliberately. At
temperature zero a model converges on the single most obvious question per
document, and an eval set of obvious questions flatters retrieval.
Reproducibility comes from committing and reviewing the output, not from
sampling settings.

188 candidates were attempted. 12 were dropped because the quoted answer
passage couldn't be located verbatim in the source text and 7 because the answer
fell inside the letterhead, leaving 169 for review.

### How it was reviewed

Every record is written with `verified: false` and counts for nothing until a
person confirms it. `verify_evalset.py` shows each query with its gold span
highlighted in context, flags any query whose source circular has a related or
newer one, and records one decision — accept, edit, widen, re-span, replace or
reject. Rejected records stay in the file with a reason. Re-reviewing a decided
record moves the earlier decision into `review_history` rather than overwriting
it.

**The number worth reporting is not the rejection rate.** 27 of 169 generated
candidates were rejected — 16 as boilerplate, 5 as ambiguous across documents, 4
from the mislabelled file, 2 other. But **96 of the accepted queries had to be
reworded**, 45% of everything kept. Almost all were unanchored: *"this
circular"*, *"these securities"*, *"the extended deadline"* — questions that make
sense only to someone holding the document the generator was looking at. A
further 10 had their gold span re-pointed to a newer circular, and 5 were
widened to a second document.

That is the real cost of LLM-generated eval sets: not that the questions are
wrong, but that they are written from inside a document and silently assume the
reader is too.

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
(multi-hop) and `"any"` when any one suffices (the same fact in two places).

Coverage accumulates across chunks: a span split by a chunk boundary counts once
the retrieved set jointly covers 80% of it. Strict single-chunk matching would
penalise small chunks for something the generator wouldn't care about.

Every offset is relative to the exact bytes in `data/documents/`. Re-extracting
with a different PDF parser shifts whitespace and silently invalidates the
entire goldset — nothing errors, the numbers just quietly become wrong.

---

## Supersession

A query generated from a circular that was later changed carries a **stale
answer as ground truth** — and a retriever that correctly returns the newer
circular is scored as a miss. The eval would penalise correct behaviour.

`build_crossrefs.py` finds related circulars two ways, because each misses what
the other catches:

- **Citations.** Circular A cites B if B's serial appears in A's text. This
  catches SEBI extensions and amendments, which name what they modify.
- **Shared subject lines.** This catches chains whose linking circular isn't in
  the corpus — in the collateral-timings chain, two circulars each cite one we
  don't have, so no citation connects them, but all three share a subject. NSE
  futures-and-options circulars state their subject as an unlabelled heading
  after the addressee rather than a `Sub:` line; reading only labelled subjects
  missed twelve documents, including an entire four-circular monthly series.

Each circular's date is read from its header, and the review screen names the
newest member of any family, warning when a query's gold span sits in an older
one. That's fine only when the query names the older circular explicitly.

**72 designed queries** (IDs 1001–1072) test these directly, each with a recorded
`stale_trap` — the document that gives a wrong, outdated or unsourced answer if
retrieved instead. They cover *silent* cases, where the newer circular never
mentions a fact so the older one still governs; *flips*, where the answer changes
from no to yes; *deferrals*, where the older circular answers "not yet
announced"; *point-in-time* queries, where naming the older circular makes its
answer correct; and multi-hop queries needing both circulars.

Several traps are **inside a single document**: a circular that quotes the
regime it replaces, one that names its mock date seventeen times against five
mentions of the live date, one that carries a consultation proposal never
adopted alongside the limits actually set, and one whose supersession takes
effect three months before its own provisions do.

Span-level scoring matters here. When a value is unchanged across circulars the
stale document gives the right number — but a compliance answer citing a
superseded circular is still wrong. Span scoring catches that; an answer-only
judge would not.

**Removal questions are held out.** "Is this ETF still eligible?" has an answer
that is an *absence* from the current list, which no span can mark. Recording
such a query with no gold spans would score every retrieval as a hit. They live
in `data/generation_only.jsonl`.

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
windows of up to four lines — a labelled row wraps once per column header
containing a line break, and the dividend strike tables have three. Across 222
table rows in the corpus it pairs 195 with no mismatches.

---

## Metrics, and why these ones

**recall@5** — what the generator actually sees. The headline, because a
retrieval miss is unrecoverable: no reranker and no prompt can surface a chunk
that was never retrieved.

**recall@20** — the reranker's ceiling. The gap between @5 and @20 separates
"the retriever can't find it" from "the retriever found it and ranked it badly."
Those need different fixes and look identical on recall@5 alone.

**Recall at a fixed token budget** — the fair comparison across chunk sizes.
Top-5 of 512-token chunks is twice the context of top-5 of 256-token chunks, so
fixed-k quietly favours larger chunks. This turned out to reverse the chunk-size
conclusion entirely, which is the clearest argument for measuring it.

**Recall by query type** — blended recall averages away the signal that
justifies hybrid retrieval, and the designed slices are reported separately for
the same reason.

**McNemar's test on paired outcomes** — comparing two runs on the same queries
means most queries carry no information. Conditioning on the discordant pairs is
more sensitive than comparing two proportions, because it removes the variance
from query difficulty, which is shared. It is also what stopped two of the four
headline differences from being claimed: hybrid over dense, and recursive over
fixed splitting, are both inside the noise.

**Bootstrap CIs rather than a holdout** — at this sample size, splitting would
leave both halves too small to measure or compare anything.

---

## Architecture

```
query
  ├── BM25                                              ─┐
  ├── dense (exhaustive KNN, text-embedding-3-small)    ─┴─ RRF fusion
  │                                                        (Azure AI Search, native)
  ├── cross-encoder rerank over top-20        (measured, NOT shipped -- see Results)
  │
  │   ── built above this line · in progress below ──
  │
  ├── GATE: retrieval-score threshold         → refuse, no LLM call
  ├── generation with forced citation schema  (constrained decoding)
  └── GATE: citation + quote verification     → refuse if unsupported
```

Refusal is to be enforced in code before the generator is invoked, not requested
in the prompt. Citation validity is a set-membership check and quote
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
`requirements.lock.txt` is exact pinned versions from a verified clean install.

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
`/openai/v1` path for the OpenAI-compatible client; the `AzureOpenAI` client
appends its own routing and will return 404 if that suffix is included.

`gpt-5-mini` is a reasoning model: it rejects any `temperature` other than the
default, and spends output tokens on reasoning before producing visible text, so
calls use `max_completion_tokens` with generous headroom.

**Two regions, deliberately.** Azure AI Search runs in Central India; the model
deployments are in Sweden Central, where they're available for the deployment
types this project uses. The search service stays local because that's where
query latency is measured and where the data lives. Cross-region embedding adds
roughly 250ms per query — paid at indexing and during evaluation runs, but not
inside reported retrieval latency, which is measured in the search service after
the query vector arrives.

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
PYTHONPATH=src python src/run_ablation.py            # → reports/ablation.md (--report reuses runs.jsonl)
```

Designed queries live in `data/*_slice_*.jsonl`, one file per chain, and are
appended to `goldset.jsonl` for review.

---

## Known limitations

- **Eval set provenance.** 142 questions are LLM-generated and human-reviewed;
  72 are designed. None are drawn from real user logs. Each record carries
  `query_source` so the proportion is visible rather than implied.
- **Sample size.** At 214 queries each one moves recall by 0.47 points. Two of
  the four headline comparisons — hybrid over dense, recursive over fixed — have
  p-values above 0.1 and are reported as inconclusive rather than as wins. A
  larger eval set, not a better retriever, is what would settle them.
- **Reranker choice.** `ms-marco-MiniLM-L-6-v2` is a general web-passage model
  and is the wrong tool for this text; its failure here is a result about that
  model at this pool size, not about cross-encoder reranking in general.
- **Reranking pool.** Fixed at 20. BM25 alone reaches 99.5% at recall@20 while
  hybrid reaches 97.2%, so RRF is dropping documents at the pool boundary. A
  larger pool would give both fusion and the reranker more to work with, and is
  the first thing to vary next.
- **Corpus scale.** At ~225 chunks, approximate nearest-neighbour search and a
  large reranking pool both stop earning their cost.
- **Single-document generation.** The generator sees one circular at a time, so
  it produces no multi-hop queries and cannot see supersession. Both rest
  entirely on the 72 designed queries.
- **Supersession detection.** Citation matching skips serials under four digits,
  which match regulation numbers and amounts too often. Subject matching catches
  only families sharing an exact subject. Neither can see a chain whose
  circulars never cite each other and whose subjects differ — SEBI 41 → 122 —
  nor one whose superseded documents are outside the corpus, as when a single
  nomination circular supersedes eighteen others, none of them here.
- **Issue date is not effective date.** `build_crossrefs.py` orders circulars by
  the date in their header. A circular can be issued well before it takes
  effect — in the mutual fund borrowing chain the newest was issued in July and
  took effect in September — so "newest" is a prompt for review, not a statement
  of which rule is in force on a given date.
- **Removal questions.** Queries whose answer is an absence can't be expressed
  as spans and are excluded from retrieval evaluation.
- **Duplicated tables.** Emitting both forms of every table was meant to help
  retrieval, but it duplicates content and complicates labelling. A cleaner
  extraction would keep only the labelled rows; it wasn't changed because every
  gold-span offset is pinned to the current text.
- **Table headers on continuation pages.** When a table continues onto a new
  page, `extract.py` takes that page's first data row as the header, so labelled
  rows there carry another row's values as labels. Values are unaffected and
  twin detection still pairs them, but in one case a symbol cell was lost.
- **Tables inserted at page ends.** Tables are written after each page's text, so
  a table on a page where a paragraph continues onto the next lands mid
  sentence. Address blocks are occasionally mistaken for tables.
- **Annexures outside the corpus text.** Some circulars refer to annexures that
  didn't extract with the body; questions whose answer lives there can point
  only at the reference.
- **Extraction artifacts inside identifiers.** PDF extraction occasionally
  inserts spaces within circular IDs (`NCL/CMPT/ 74926`). Retrieval is
  unaffected — the analyzer tokenizes on slashes and whitespace — but it breaks
  parsing identifiers from text, which is why exact-ID queries and identity
  checks work from the serial number.
- **Extraction choice.** `pdfplumber` plus hand-written table handling accounts
  for most of the data-quality entries above. Azure AI Document Intelligence's
  layout model handles tables spanning pages properly and would be the change
  worth making; it wasn't made here because re-extracting would shift every
  gold-span offset.
- **No semantic ranker.** Azure AI Search's L2 semantic reranker is not
  available on the Free tier, so reranking runs locally with
  `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU — roughly 50–150ms for a pool of
  20. A managed reranker would move that cost server-side.
- **Cross-region embedding.** End-to-end query latency includes a ~250ms network
  hop a single-region deployment would not. Reported retrieval latency excludes
  it.
- **Dated API version.** Scripts pin an `AzureOpenAI` API version. Azure's v1
  endpoint removes the need to pin one; migrating is a client-construction
  change in each script.
