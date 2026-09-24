# Generation

214 verified queries. Generator `gpt-5-mini`, judge `gpt-5` — different models, same family, which is noted as a limitation. Retrieval is `512-recursive` + `hybrid`, top 5.

## Answered and refused

| | count | share |
|---|---|---|
| Answered | 190 | 88.8% |
| Refused | 24 | 11.2% |

Refusal reasons: some citations failed verification 17, no citation survived verification 4, insufficient context 3.

## Groundedness

**94.7%** of answered queries (180/190) are judged fully supported by the retrieved chunks.

Groundedness is attribution, not correctness. An answer drawn confidently from the wrong retrieved chunk scores as grounded — which is why retrieval is measured separately, and why the supersession slice matters.

| Query type | answered | grounded |
|---|---|---|
| exact_id | 39 | 94.9% |
| keyword | 31 | 96.8% |
| multi_hop | 20 | 90.0% |
| natural | 76 | 96.1% |
| supersession | 24 | 91.7% |

## Refusal correctness

Retrieval already tells us, per query, whether the gold span was in the top 5. So refusals can be scored without a judge.

| Retrieval | Pipeline | count | |
|---|---|---|---|
| found the answer | answered | 180 | as intended |
| found the answer | refused | 21 | lost answer |
| missed | refused | 3 | correct refusal |
| missed | answered | 10 | **answered without the source** |

The last row is the one that matters. An answer produced when retrieval missed is either drawn from the model's own knowledge or from the wrong chunk, and citation verification is the only thing standing between it and the user.

## Citation verification

21 of 214 queries had at least one citation fail verification — a quote that does not appear in the chunk it cites, or a chunk that was never retrieved.

## Variance

Neither model accepts `temperature=0`, so both are non-deterministic. A sample of 30 queries was answered twice and judged twice.

| | flips | of | rate |
|---|---|---|---|
| Generator: answered vs refused | 6 | 30 | 20.0% |
| Judge: grounded vs not, same answer | 0 | 24 | 0.0% |

These are the floor. A measured difference smaller than the judge flip rate is not a difference, and any future change to the prompt or the generator has to clear it before it can be called an improvement.

## Ungrounded answers — 10

| Query | Type | First unsupported claim |
|---|---|---|
| 8. When did the ETF list in NSE circular 74926 come into force? | exact_id | "The list came into effect on July 29, 2026" (as applied to the list in circular 74926). T |
| 21. What minimum quantity is required for the ETF with symbol BS | natural | That under the revised list (circular dated August 24, 2026), the minimum quantity require |
| 77. Under the Regulation 9C introduced on October 27, 2025, what | natural | "Debenture trustees must transfer activities not regulated by SEBI to separate business un |
| 84. What are the entity-level intraday FutEq position limits for | natural | SEBI's September 01, 2025 circular sets entity-level intraday FutEq limits at Net 1,000 c |
| 125. From what date does SEBI circular 19251 apply? | exact_id | That SEBI circular 19251 applies from August 20, 2026 (the context does not identify any c |
| 1001. Until what time can we request an end-of-day release of pled | supersession | "I have relied on the July 22, 2026 circular." (The context shows a circular dated July 21 |
| 1009. What symbol is the Axis technology ETF listed under for cros | supersession | That as of August 24, 2026, the Axis technology ETF is listed under the symbol ITAXIS. |
| 1023. Which intraday borrowing conditions were in force for mutual | multi_hop | The answer states intraday borrowings could be used only for repurchase or redemption of u |
| 1027. gold silver ETF initial price band flex step | keyword | "The cooling-off period is 15 minutes after trades at or above 5.90%." (Context adds an ex |
| 1032. What condition did NSE attach to launching F&O contracts on  | multi_hop | The contracts were made available for trading on August 26, 2026. |

