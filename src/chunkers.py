"""
Chunkers that preserve character offsets into the source document.

Offsets are non-negotiable here: gold labels are (doc_id, char_start, char_end)
spans in the source text, so every chunk must know where it came from or the
labels can't be matched against retrieval results.
"""

from __future__ import annotations

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:                                    # offline / no cache
    ENC = None


def ntok(s: str) -> int:
    """Token count. Falls back to a ~4-chars-per-token estimate."""
    return len(ENC.encode(s)) if ENC else max(1, round(len(s) / 4))


def _hard_cut(text: str, start: int, max_tokens: int):
    """Split with no separator available. Token-exact when tiktoken is
    present, character-approximate otherwise."""
    if ENC:
        ids = ENC.encode(text)
        out, cursor = [], 0
        for i in range(0, len(ids), max_tokens):
            piece = ENC.decode(ids[i:i + max_tokens])
            out.append((piece, start + cursor, start + cursor + len(piece)))
            cursor += len(piece)
        return out
    width = max_tokens * 4
    return [(text[i:i + width], start + i, start + min(i + width, len(text)))
            for i in range(0, len(text), width)]


# --------------------------------------------------------------- fixed

def fixed_split(text: str, max_tokens: int = 512, overlap_tokens: int = 64):
    """Cut every max_tokens tokens, striding by (max_tokens - overlap).

    No regard for document structure -- this is the baseline that recursive
    splitting is meant to beat.
    """
    step = max_tokens - overlap_tokens
    chunks = []

    if ENC:
        ids = ENC.encode(text)
        for i in range(0, len(ids), step):
            window = ids[i:i + max_tokens]
            if not window:
                break
            cs = len(ENC.decode(ids[:i]))
            ce = cs + len(ENC.decode(window))
            chunks.append({"text": text[cs:ce], "char_start": cs,
                           "char_end": ce, "token_count": len(window)})
            if i + max_tokens >= len(ids):
                break
    else:
        width, stride = max_tokens * 4, step * 4
        for i in range(0, len(text), stride):
            seg = text[i:i + width]
            if not seg:
                break
            chunks.append({"text": seg, "char_start": i,
                           "char_end": min(i + width, len(text)),
                           "token_count": ntok(seg)})
            if i + width >= len(text):
                break
    return chunks


# ----------------------------------------------------------- recursive

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_keep_offsets(text: str, start: int, max_tokens: int, sep_idx: int):
    """Recursively split [start, start+len(text)) into pieces that fit.

    Returns a list of (piece_text, abs_start, abs_end). Pieces are the
    finest-grained units; merging happens afterwards.
    """
    if ntok(text) <= max_tokens:
        return [(text, start, start + len(text))]

    if sep_idx >= len(SEPARATORS) - 1:
        return _hard_cut(text, start, max_tokens)   # last resort

    sep = SEPARATORS[sep_idx]
    if sep not in text:
        # This separator doesn't occur; try the next finer one
        return _split_keep_offsets(text, start, max_tokens, sep_idx + 1)

    pieces, cursor = [], 0
    parts = text.split(sep)
    for i, part in enumerate(parts):
        # Keep the separator attached to the preceding part so offsets
        # stay contiguous and no characters are lost
        seg = part + (sep if i < len(parts) - 1 else "")
        if not seg:
            continue
        abs_start = start + cursor
        if ntok(seg) > max_tokens:
            pieces.extend(_split_keep_offsets(seg, abs_start, max_tokens, sep_idx + 1))
        else:
            pieces.append((seg, abs_start, abs_start + len(seg)))
        cursor += len(seg)
    return pieces


def _merge(pieces, max_tokens: int, overlap_tokens: int):
    """Greedily merge adjacent pieces up to max_tokens, with overlap.

    Without this step you get a pile of tiny fragments. Split down, merge up.
    """
    chunks, buf = [], []

    def flush():
        if not buf:
            return
        cs, ce = buf[0][1], buf[-1][2]
        chunks.append({
            "text": "".join(p[0] for p in buf),
            "char_start": cs,
            "char_end": ce,
            "token_count": ntok("".join(p[0] for p in buf)),
        })

    for piece in pieces:
        candidate = buf + [piece]
        if ntok("".join(p[0] for p in candidate)) > max_tokens and buf:
            flush()
            # Carry back trailing pieces as overlap
            back, acc = [], 0
            for p in reversed(buf):
                t = ntok(p[0])
                if acc + t > overlap_tokens:
                    break
                back.insert(0, p)
                acc += t
            buf = back + [piece]
        else:
            buf = candidate
    flush()
    return chunks


def recursive_split(text: str, max_tokens: int = 512, overlap_tokens: int = 64):
    """Split at the coarsest separator that works, then merge back up."""
    pieces = _split_keep_offsets(text, 0, max_tokens, 0)
    return _merge(pieces, max_tokens, overlap_tokens)


# ---------------------------------------------------------------- check

def verify(text: str, chunks: list[dict]) -> None:
    """Every chunk's offsets must actually index the source text.

    If this fails, gold-span matching is silently broken.
    """
    for c in chunks:
        assert text[c["char_start"]:c["char_end"]] == c["text"], \
            f"offset mismatch at {c['char_start']}"
    # Chunks must cover the document (gaps mean unretrievable text)
    covered = set()
    for c in chunks:
        covered.update(range(c["char_start"], c["char_end"]))
    missing = len(text) - len(covered)
    assert missing == 0, f"{missing} characters not covered by any chunk"
