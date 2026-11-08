from data.ingest import ingest, deidentify, chunk, jaccard, shingles
from data.contamination import build_eval_index, split_contaminated
from data.sources import allowed

def test_eval_only_never_allowed():
    assert not allowed("medqa", "rag", accept_unverified=True)
    assert not allowed("medqa", "train", accept_unverified=True)

def test_unverified_license_blocked_by_default():
    assert not allowed("medlineplus", "rag")
    assert allowed("medlineplus", "rag", accept_unverified=True)

def test_unknown_source_blocked():
    assert not allowed("random_site", "rag", accept_unverified=True)

def test_deidentify():
    t = deidentify("Mail a.b@x.com or call +91 98765 43210, id 1234 5678 9012.")
    assert "a.b@x.com" not in t and "98765" not in t and "1234 5678 9012" not in t

def test_chunk_overlap_and_end():
    words = " ".join(f"w{i}" for i in range(700))
    cs = chunk(words, 300, 50)
    assert len(cs) == 3 and cs[0].split()[-50:] == cs[1].split()[:50]
    assert cs[-1].split()[-1] == "w699"

def test_ingest_report_and_dedupe():
    base = " ".join(f"term{i}" for i in range(100))
    docs = [{"source": "medlineplus", "doc_id": "1", "text": base},
            {"source": "medlineplus", "doc_id": "2", "text": base},          # exact dup
            {"source": "medqa", "doc_id": "3", "text": "held out"},          # blocked
            {"source": "medlineplus", "doc_id": "4", "text": base + " extra"}]  # near dup
    out, rep = ingest(docs, accept_unverified=True)
    assert rep["blocked_source"] == 1 and rep["kept"] == 1
    assert rep["exact_dup"] >= 1 and rep["near_dup"] >= 1
    assert out[0]["provenance"] == "unverified_license"

def test_contamination_detected():
    q = "A 45 year old man presents with chest pain radiating to the left arm and sweating"
    idx = build_eval_index([q])
    chunks = [{"text": "Notes: " + q + " was the case."}, {"text": "Totally unrelated nutrition text about iron"}]
    clean, dirty = split_contaminated(chunks, idx)
    assert len(dirty) == 1 and len(clean) == 1
