"""Detect training/RAG text that overlaps held-out evaluation items."""
import re

def ngrams(text: str, n: int = 8) -> set:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

def build_eval_index(eval_texts, n: int = 8) -> set:
    idx = set()
    for t in eval_texts: idx |= ngrams(t, n)
    return idx

def split_contaminated(chunks: list[dict], eval_index: set, n: int = 8):
    clean, dirty = [], []
    for c in chunks:
        (dirty if ngrams(c["text"], n) & eval_index else clean).append(c)
    return clean, dirty
