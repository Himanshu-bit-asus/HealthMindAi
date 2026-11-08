"""Corpus ingestion: license gate -> normalise -> de-identify -> chunk -> dedupe.
Regex de-identification is a first pass only; it does not guarantee removal of
all personal data. Use a dedicated PHI tool plus human spot checks on real records."""
from __future__ import annotations
import hashlib, json, re, unicodedata
from pathlib import Path
from data.sources import allowed

PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR_LIKE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_LIKE]"),
    (re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}(?!\d)"), "[PHONE]"),      # 10-digit, e.g. India
    (re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\d)"), "[PHONE]"),  # US-style
]

def normalise(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()

def deidentify(t: str) -> str:
    for rx, tag in PII: t = rx.sub(tag, t)
    return t

def chunk(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    w = text.split()
    if size <= overlap: raise ValueError("size must exceed overlap")
    out, i = [], 0
    while i < len(w):
        out.append(" ".join(w[i:i + size]))
        if i + size >= len(w): break
        i += size - overlap
    return out

def shingles(t: str, n: int = 5) -> set:
    w = t.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0

def ingest(docs: list[dict], purpose: str = "rag", accept_unverified: bool = False,
           near_dup: float = 0.9, version: str = "v1") -> tuple[list[dict], dict]:
    """docs: [{'source','doc_id','text'}]. Returns (chunks, report)."""
    rep = {"input_docs": len(docs), "blocked_source": 0, "exact_dup": 0,
           "near_dup": 0, "kept": 0, "version": version}
    seen_hash: set[str] = set(); kept_sh: list[set] = []; out = []
    for d in docs:
        if not allowed(d["source"], purpose, accept_unverified):
            rep["blocked_source"] += 1; continue
        text = deidentify(normalise(d["text"]))
        for k, c in enumerate(chunk(text)):
            h = hashlib.sha256(c.lower().encode()).hexdigest()
            if h in seen_hash: rep["exact_dup"] += 1; continue
            sh = shingles(c)
            if any(jaccard(sh, o) >= near_dup for o in kept_sh):
                rep["near_dup"] += 1; continue
            seen_hash.add(h); kept_sh.append(sh)
            out.append({"chunk_id": h[:16], "source": d["source"], "doc_id": d["doc_id"],
                        "chunk_index": k, "text": c, "corpus_version": version,
                        "provenance": "unverified_license" if accept_unverified else "verified"})
    rep["kept"] = len(out)
    return out, rep

def write(chunks, report, outdir):
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    (p / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    (p / "report.json").write_text(json.dumps(report, indent=2))
