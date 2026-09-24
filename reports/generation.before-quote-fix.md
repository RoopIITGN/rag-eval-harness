# Generation

214 verified queries. Generator `gpt-5-mini`, judge `gpt-5` — different models, same family, which is noted as a limitation. Retrieval is `512-recursive` + `hybrid`, top 5.

## Answered and refused

| | count | share |
|---|---|---|
| Answered | 169 | 79.0% |
| Refused | 45 | 21.0% |

Refusal reasons: some citations failed verification 20, model returned no tool call 14, insufficient context 6, no citation survived verification 5.

## Groundedness

**98.8%** of answered queries (167/169) are judged fully supported by the retrieved chunks.

Groundedness is attribution, not correctness. An answer drawn confidently from the wrong retrieved chunk scores as grounded — which is why retrieval is measured separately, and why the supersession slice matters.

| Query type | answered | grounded |
|---|---|---|
| exact_id | 35 | 97.1% |
| keyword | 28 | 96.4% |
| multi_hop | 13 | 100.0% |
| natural | 75 | 100.0% |
| supersession | 18 | 100.0% |

## Refusal correctness

Retrieval already tells us, per query, whether the gold span was in the top 5. So refusals can be scored without a judge.

| Retrieval | Pipeline | count | |
|---|---|---|---|
| found the answer | answered | 160 | as intended |
| found the answer | refused | 41 | lost answer |
| missed | refused | 4 | correct refusal |
| missed | answered | 9 | **answered without the source** |

The last row is the one that matters. An answer produced when retrieval missed is either drawn from the model's own knowledge or from the wrong chunk, and citation verification is the only thing standing between it and the user.

## Citation verification

25 of 214 queries had at least one citation fail verification — a quote that does not appear in the chunk it cites, or a chunk that was never retrieved.

## Variance

Neither model accepts `temperature=0`, so both are non-deterministic. A sample of 30 queries was answered twice and judged twice.

| | flips | of | rate |
|---|---|---|---|
| Generator: answered vs refused | 1 | 30 | 3.3% |
| Judge: grounded vs not, same answer | 0 | 25 | 0.0% |

These are the floor. A measured difference smaller than the judge flip rate is not a difference, and any future change to the prompt or the generator has to clear it before it can be called an improvement.

## Ungrounded answers — 2

| Query | Type | First unsupported claim |
|---|---|---|
| 125. From what date does SEBI circular 19251 apply? | exact_id | That SEBI circular 19251 applies from August 20, 2026. |
| 1059. index options intraday position snapshot timing | keyword | That the snapshot requirement (including one between 14:45–15:30) is based on SEBI circula |

