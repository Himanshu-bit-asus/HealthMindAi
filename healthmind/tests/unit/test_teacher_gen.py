import json
from data.retrieval import BM25
from data.teacher_gen import (generate_record, build_review_queue, apply_review,
                              export_sft, qc_report, is_safety_critical, build_prompt)
import pytest

CH = [{"chunk_id": "c1", "text": "HbA1c reflects average blood glucose exposure over approximately two to three months."},
      {"chunk_id": "c2", "text": "Iron deficiency anemia is common and ferritin is a marker of iron stores."}]
R = BM25(CH)

def teacher(**o):
    d = {"answer": "HbA1c reflects average blood glucose exposure over about two to three months.",
         "evidence_ids": ["c1"], "uncertainty": "Depends on lab and clinical context.",
         "escalation_required": False}; d.update(o)
    return lambda prompt: json.dumps(d)

def test_retrieval_ranks_relevant_first():
    assert R.search("what does HbA1c measure", 1)[0]["chunk_id"] == "c1"

def test_prompt_contains_evidence_ids():
    assert "[c1]" in build_prompt("q", CH)

def test_good_record_pending_and_ai_labeled():
    rec, why = generate_record("What does HbA1c measure?", "medical", R, teacher())
    assert why is None and rec["provenance"] == "ai_generated" and rec["review_status"] == "pending"
    assert rec["required_reviewer"] == ["doctor"] and rec["grounding"] >= 0.6

@pytest.mark.parametrize("kw,reason", [
    ({"evidence_ids": ["zzz"]}, "fabricated_citation"),
    ({"evidence_ids": []}, "no_citation"),
    ({"answer": "You definitely have diabetes."}, "overconfident"),
    ({"answer": "Quantum entanglement powers metabolism pathways entirely."}, "weak_grounding"),
])
def test_rejections(kw, reason):
    rec, why = generate_record("What does HbA1c measure?", "medical", R, teacher(**kw))
    assert rec is None and why.startswith(reason)

def test_invalid_format_and_no_evidence():
    assert generate_record("HbA1c?", "medical", R, lambda p: "nope")[1] == "invalid_format"
    assert generate_record("zzzz qqqq", "medical", R, teacher())[1] == "no_evidence"

def test_safety_critical_forces_escalation():
    assert is_safety_critical("Can I stop taking insulin?")
    rec, _ = generate_record("What does HbA1c measure in pregnancy?", "medical", R, teacher())
    assert rec["safety_critical"] and rec["escalation_required"]

def mk(i, crit=False):
    return {"question": f"q{i}", "domain": "medical", "answer": "a", "evidence_ids": ["c1"],
            "uncertainty": "u", "escalation_required": False, "safety_critical": crit,
            "provenance": "ai_generated", "review_status": "pending"}

def test_queue_includes_all_critical_and_is_reproducible():
    recs = [mk(i, crit=(i < 3)) for i in range(30)]
    q1, q2 = build_review_queue(recs, 0.2, seed=1), build_review_queue(recs, 0.2, seed=1)
    assert sum(r["safety_critical"] for r in q1) == 3 and len(q1) == 3 + 5 and q1 == q2

def test_role_enforced():
    with pytest.raises(PermissionError):
        apply_review(mk(1), "approve", "u1", "dietitian")

def test_correction_requires_text_and_relabels():
    with pytest.raises(ValueError): apply_review(mk(1), "correct", "u1", "doctor")
    r = apply_review(mk(1), "correct", "u1", "doctor", corrected_answer="fixed")
    assert r["provenance"] == "expert_corrected" and r["answer"] == "fixed" and r["review"]["reviewer_id"] == "u1"

def test_export_only_approved():
    a = apply_review(mk(1), "approve", "u1", "doctor")
    rej = apply_review(mk(2), "reject", "u1", "doctor")
    pending = mk(3)
    out = export_sft([a, rej, pending])
    assert len(out) == 1 and out[0]["provenance"] == "expert_approved"

def test_qc_report():
    recs = [apply_review(mk(1), "approve", "u", "doctor"),
            apply_review(mk(2), "correct", "u", "doctor", corrected_answer="x"),
            apply_review(mk(3), "reject", "u", "doctor"),
            apply_review(mk(4), "approve", "u", "doctor")]
    q = qc_report(recs)
    assert q["n"] == 4 and q["approved_unchanged"] == .5 and q["corrected"] == .25 and q["rejected"] == .25
