"""Minimal BM25 retriever (pure Python) for tests and small corpora.
Swap for pgvector/OpenSearch in production; keep the .search() interface."""
import math, re
from collections import Counter

def _tok(t): return re.findall(r"[a-z0-9]+", t.lower())

class BM25:
    def __init__(self, chunks: list[dict], k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(_tok(c["text"])) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _tok(query); scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    f = d[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
            if s > 0: scored.append((s, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]
