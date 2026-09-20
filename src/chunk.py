"""Produce three chunk sets from the frozen text, with verified offsets."""
import json
from pathlib import Path
from chunkers import fixed_split, recursive_split, verify, ENC

if ENC is None:
    raise SystemExit("tiktoken unavailable - token counts would be estimates. Fix before chunking.")

CONFIGS = {
    "256-fixed":     (fixed_split,     256, 32),
    "512-fixed":     (fixed_split,     512, 64),
    "512-recursive": (recursive_split, 512, 64),
}

docs = sorted(Path("data/documents").glob("*.txt"))
if not docs:
    raise SystemExit("No documents. Run extract.py first.")

for name, (fn, size, overlap) in CONFIGS.items():
    rows, tokens = [], 0
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        chunks = fn(text, max_tokens=size, overlap_tokens=overlap)
        verify(text, chunks)          # offsets must round-trip, or stop here
        for i, c in enumerate(chunks):
            rows.append({
                "id": f"{doc.stem}_{i}",
                "doc_id": doc.stem,
                "content": c["text"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "token_count": c["token_count"],
            })
            tokens += c["token_count"]

    out = Path(f"data/chunks_{name}.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"{name:16s} {len(rows):>4d} chunks   "
          f"mean {tokens // max(1, len(rows)):>4d} tokens   -> {out}")
