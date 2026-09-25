"""Capture real pipeline outputs for the README -- a reader has no credentials."""
import json
from pathlib import Path
import generate

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("supersession", "By what time must members deposit cash and FDR as collateral?"),
    ("multi-hop",    "When did the BANKNIFTY quantity freeze limit change, and what did it change from and to?"),
    ("multi-hop, source in another circular", "Why can equity index funds only borrow for under-executed sell trades from August 2026 onwards?"),
    ("refusal",      "What is the capital of France?"),
    ("in-document trap", "What is the intraday net position limit for index options?"),
]
out = ["# Example outputs\n",
       "Captured by `src/capture_examples.py` against the live pipeline.\n"]
for label, q in CASES:
    r = generate.answer(q)
    out.append(f"## {label}\n\n**Q:** {q}\n")
    if r.answered:
        out.append(f"{r.answer}\n")
        out.append("| chunk | circular | quote |")
        out.append("|---|---|---|")
        for c in r.citations:
            out.append(f"| {c['chunk_id']} | {c['doc_id']} | {c['source_quote'][:80]} |")
    else:
        out.append(f"**Refused** — {r.refusal_reason}\n")
    out.append(f"\n_Retrieved: {', '.join(c['doc_id'] for c in r.chunks)}_\n")
    print(f"  {label}: {'answered' if r.answered else 'refused'}")
(ROOT / "reports/examples.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("-> reports/examples.md")
