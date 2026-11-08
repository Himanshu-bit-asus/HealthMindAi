"""Teacher generation + review workflow.
Teacher answers ONLY from retrieved evidence, must cite it, and every record
starts as AI-generated / pending review. Only clinician-approved records may be
exported for training. The teacher is not a source of medical truth."""
from __future__ import annotations
import json, random, re
from datetime import datetime, timezone
from rewards.medical_rewards import parse_answer, OVERCONFIDENT

DOMAIN_REVIEWERS = {
    "medical": {"doctor"}, "nutrition": {"dietitian"},
    "fitness": {"fitness_professional"}, "medication": {"pharmacist", "doctor"},
}
SAFETY_TERMS = re.compile(
    r"\b(dose|dosage|mg|overdose|pregnan\w*|breastfeed\w*|child(ren)?|infant|newborn|"
    r"chest pain|stroke|suicid\w*|self[- ]harm|bleeding|allerg\w*|anaphyla\w*|insulin|"
    r"warfarin|interaction|stop taking|discontinue|kidney|dialysis|seizure)\b", re.I)
STOP = set("this that with from have been which their there about would could should these those into than then when what also may can not are the and for".split())

SYSTEM = ("You are an educational medical teacher. Answer ONLY using the numbered evidence. "
          "If the evidence is insufficient, say so and set escalation_required true. "
          "Do not diagnose or give prescribing/dose-change instructions. Return JSON only with keys: "
          "answer (string), evidence_ids (list of ids you used), uncertainty (string), "
          "escalation_required (boolean).")

def build_prompt(question: str, evidence: list[dict]) -> str:
    ev = "\n".join(f"[{e['chunk_id']}] {e['text']}" for e in evidence)
    return f"{SYSTEM}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {question}"

def is_safety_critical(question: str, answer: str = "") -> bool:
    return bool(SAFETY_TERMS.search(question) or SAFETY_TERMS.search(answer))

def grounding_score(answer: str, cited_texts: list[str]) -> float:
    """Fraction of the answer's content words found in the cited evidence."""
    words = {w for w in re.findall(r"[a-z]{4,}", answer.lower()) if w not in STOP}
    if not words: return 0.0
    ev = set(re.findall(r"[a-z]{4,}", " ".join(cited_texts).lower()))
    return len(words & ev) / len(words)

def generate_record(question, domain, retriever, teacher_fn, k=3, min_grounding=0.6):
    """Returns (record | None, reject_reason | None)."""
    if domain not in DOMAIN_REVIEWERS: raise ValueError(f"unknown domain {domain}")
    evidence = retriever.search(question, k)
    if not evidence: return None, "no_evidence"
    d = parse_answer(teacher_fn(build_prompt(question, evidence)))
    if d is None: return None, "invalid_format"
    by_id = {e["chunk_id"]: e for e in evidence}
    cited = set(d["evidence_ids"])
    if not cited: return None, "no_citation"
    if cited - set(by_id): return None, "fabricated_citation"
    if OVERCONFIDENT.search(d["answer"]): return None, "overconfident"
    g = grounding_score(d["answer"], [by_id[c]["text"] for c in cited])
    if g < min_grounding: return None, f"weak_grounding_{g:.2f}"
    sc = is_safety_critical(question, d["answer"])
    return {
        "question": question, "domain": domain, "answer": d["answer"],
        "evidence_ids": sorted(cited), "uncertainty": d["uncertainty"],
        "escalation_required": d["escalation_required"] or sc,   # safety topics default to escalate
        "grounding": round(g, 3), "safety_critical": sc,
        "provenance": "ai_generated", "review_status": "pending",
        "required_reviewer": sorted(DOMAIN_REVIEWERS[domain]),
    }, None

def build_review_queue(records, qc_fraction=0.2, seed=42):
    """All safety-critical records + a seeded random QC sample of the rest."""
    rng = random.Random(seed)
    crit = [r for r in records if r["safety_critical"]]
    rest = [r for r in records if not r["safety_critical"]]
    n = min(len(rest), max(1, round(len(rest) * qc_fraction))) if rest else 0
    return crit + rng.sample(rest, n)

def apply_review(record, decision, reviewer_id, reviewer_role, corrected_answer=None, note=""):
    """decision: 'approve' | 'reject' | 'correct'. Role must match the domain."""
    if reviewer_role not in DOMAIN_REVIEWERS[record["domain"]]:
        raise PermissionError(f"{reviewer_role} may not review {record['domain']} records")
    if decision not in ("approve", "reject", "correct"): raise ValueError("bad decision")
    r = dict(record)
    if decision == "correct":
        if not corrected_answer: raise ValueError("correction text required")
        r["answer"], r["provenance"] = corrected_answer, "expert_corrected"
    elif decision == "approve":
        r["provenance"] = "expert_approved"
    r["review_status"] = "rejected" if decision == "reject" else "approved"
    r["review"] = {"reviewer_id": reviewer_id, "role": reviewer_role, "note": note,
                   "at": datetime.now(timezone.utc).isoformat()}
    return r

def export_sft(records) -> list[dict]:
    """Only expert-approved/corrected records become training data."""
    out = []
    for r in records:
        if r["review_status"] == "approved" and r["provenance"] in ("expert_approved", "expert_corrected"):
            out.append({"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": json.dumps({
                    "answer": r["answer"], "evidence_ids": r["evidence_ids"],
                    "uncertainty": r["uncertainty"], "escalation_required": r["escalation_required"]})}],
                "provenance": r["provenance"], "domain": r["domain"]})
    return out

def qc_report(reviewed) -> dict:
    """Teacher quality as judged by your clinicians (approve vs correct/reject)."""
    n = len(reviewed)
    if n == 0: return {"n": 0}
    ok = sum(1 for r in reviewed if r["provenance"] == "expert_approved")
    return {"n": n, "approved_unchanged": ok / n,
            "corrected": sum(r["provenance"] == "expert_corrected" for r in reviewed) / n,
            "rejected": sum(r["review_status"] == "rejected" for r in reviewed) / n}
