# Example outputs

Captured by `src/capture_examples.py` against the live pipeline.

## supersession

**Q:** By what time must members deposit cash and FDR as collateral?

Current position: the cut-off for deposit (addition) of Cash and FDR is 9:00 PM. An earlier circular had set the cut-off at 7:30 PM. I relied on Circular Ref. No: 090/2026 (dated July 21, 2026) for the current timing.

| chunk | circular | quote |
|---|---|---|
| chunk_2 | NSE-CMPT-75321 | Cash and FDR Addition 7:30 PM 9:00 PM |
| chunk_1 | NSE-CMPT-72357 | The cut-off time for deposit of Cash and Fixed Deposit Receipt shall be 7:30 pm  |
| chunk_2 | NSE-CMPT-75321 | Circular Ref. No: 090/2026 |

_Retrieved: NSE-CMPT-72357, NSE-CMPT-75321, NSE-CMPT-72357, NSE-CMPT-75463, NSE-CMPT-72357_

## multi-hop

**Q:** When did the BANKNIFTY quantity freeze limit change, and what did it change from and to?

The current BANKNIFTY quantity freeze limit is 600. It changed on July 01, 2026 from 900 to 600.

| chunk | circular | quote |
|---|---|---|
| chunk_3 | NSE-FAOP-76068 | 1 BANKNIFTY 600 |
| chunk_5 | NSE-FAOP-74942 | w.e.f. July 01, 2026: |
| chunk_1 | NSE-FAOP-74469 | 1 BANKNIFTY 900 |
| chunk_5 | NSE-FAOP-74942 | 1 BANKNIFTY 600 |

_Retrieved: NSE-FAOP-74469, NSE-FAOP-76068, NSE-FAOP-76068, NSE-FAOP-75521, NSE-FAOP-74942_

## correct refusal

**Q:** Why can equity index funds only borrow for under-executed sell trades from August 2026 onwards?

Current position: borrowings by equity-oriented index funds and equity-oriented ETFs for under-executed sell trades are permitted only to participate in the Closing Auction Session, effective August 03, 2026. This is because Regulation 42(1) allows borrowing for settlement of under-executed sell trades, and SEBI clarified that such borrowings are "permissible only for the purpose of participation by such funds in the Closing Auction Session" from the session’s applicability.

| chunk | circular | quote |
|---|---|---|
| chunk_3 | SEBI-6961 | is permissible only for the purpose of participation by such funds in the Closin |
| chunk_1 | SEBI-6961 | effective from August 03, 2026. |
| chunk_2 | SEBI-6961 | for settlement of trades by equity oriented index funds and equity oriented exch |

_Retrieved: SEBI-6961, SEBI-6961, SEBI-6961, SEBI-7885, SEBI-16006_

## refusal

**Q:** What is the capital of France?

**Refused** — insufficient context


_Retrieved: SEBI-19251, SEBI-8965, SEBI-17111, SEBI-17111, SEBI-16006_

## in-document trap

**Q:** What is the intraday net position limit for index options?

Current position: The intraday net position limit for index options is ₹5,000 crore. This limit is on a FutEq (futures-equivalent) basis for each entity.

| chunk | circular | quote |
|---|---|---|
| chunk_2 | SEBI-122 | 4.1. Intraday Net position limit (FutEq basis) for each entity shall be ₹5,000 c |
| chunk_2 | SEBI-122 | Intraday Net position limit (FutEq basis) for each entity shall be ₹5,000 cr. |

_Retrieved: SEBI-122, SEBI-122, SEBI-122, SEBI-41, SEBI-41_

