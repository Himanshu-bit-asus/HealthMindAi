from datetime import datetime, timedelta, timezone
import pytest
from review.service import ReviewService, Reviewer, ReviewError

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
class Clock:
    def __init__(self): self.t = T0
    def __call__(self): return self.t

def rec(i, crit=False, domain="medical"):
    return {"question": f"q{i}", "domain": domain, "answer": "ai answer", "evidence_ids": ["c1"],
            "uncertainty": "u", "escalation_required": crit, "safety_critical": crit,
            "provenance": "ai_generated", "review_status": "pending"}

D1, D2, D3 = Reviewer("d1", "doctor", True), Reviewer("d2", "doctor", True), Reviewer("d3", "doctor", True)

def mk():
    c = Clock(); return ReviewService(now=c), c

def test_unverified_reviewer_refused():
    s, _ = mk(); s.enqueue([rec(1)])
    with pytest.raises(ReviewError, match="not verified"):
        s.claim_next(Reviewer("x", "doctor", False))

def test_role_filtering():
    s, _ = mk(); s.enqueue([rec(1)])
    assert s.claim_next(Reviewer("n1", "dietitian", True)) is None
    assert s.claim_next(D1) is not None

def test_safety_critical_first():
    s, _ = mk(); s.enqueue([rec(1), rec(2, crit=True)])
    assert s.claim_next(D1)["record"]["safety_critical"] is True

def test_lease_exclusive_and_expires():
    s, c = mk(); s.enqueue([rec(1)])
    s.claim_next(D1)
    assert s.claim_next(D2) is None
    c.t += timedelta(minutes=16)
    assert s.claim_next(D2) is not None

def test_cannot_submit_without_lease():
    s, c = mk(); [i] = s.enqueue([rec(1)])
    with pytest.raises(ReviewError, match="lease"): s.submit(D1, i, "approve")
    s.claim_next(D1); c.t += timedelta(minutes=20)
    with pytest.raises(ReviewError, match="lease"): s.submit(D1, i, "approve")

def test_single_review_finalises_normal_item():
    s, _ = mk(); [i] = s.enqueue([rec(1)])
    s.claim_next(D1); item = s.submit(D1, i, "approve")
    assert item["status"] == "approved" and len(s.export_training()) == 1

def test_wrong_role_cannot_submit():
    s, _ = mk(); [i] = s.enqueue([rec(1)])
    item = s.repo.get(i); item["lease_holder"], item["lease_until"] = "n1", T0 + timedelta(hours=1); s.repo.save(item)
    with pytest.raises(PermissionError): s.submit(Reviewer("n1", "dietitian", True), i, "approve")

def test_safety_critical_needs_two_agreeing_reviewers():
    s, _ = mk(); [i] = s.enqueue([rec(1, crit=True)])
    s.claim_next(D1); assert s.submit(D1, i, "approve")["status"] == "pending"
    assert s.export_training() == []                      # not exportable after one review
    assert s.claim_next(D1) is None                       # same reviewer can't take it again
    s.claim_next(D2); assert s.submit(D2, i, "approve")["status"] == "approved"
    assert len(s.export_training()) == 1

def test_disagreement_is_disputed_then_adjudicated():
    s, _ = mk(); [i] = s.enqueue([rec(1, crit=True)])
    s.claim_next(D1); s.submit(D1, i, "approve")
    s.claim_next(D2); assert s.submit(D2, i, "reject")["status"] == "disputed"
    assert s.export_training() == []
    with pytest.raises(ReviewError, match="different"): s.adjudicate(D1, i, "approve")
    assert s.adjudicate(D3, i, "reject")["status"] == "rejected"
    assert s.export_training() == []

def test_two_matching_corrections_finalise_with_correction():
    s, _ = mk(); [i] = s.enqueue([rec(1, crit=True)])
    s.claim_next(D1); s.submit(D1, i, "correct", "Better answer.")
    s.claim_next(D2); item = s.submit(D2, i, "correct", "better  answer.")
    out = s.export_training()
    assert item["status"] == "approved" and out[0]["provenance"] == "expert_corrected"

def test_audit_log_records_actions():
    s, _ = mk(); [i] = s.enqueue([rec(1)])
    s.claim_next(D1); s.submit(D1, i, "approve")
    assert [e["action"] for e in s.repo.audit] == ["enqueue", "claim", "submit"]

def test_stats():
    s, _ = mk(); s.enqueue([rec(1), rec(2)])
    assert s.stats() == {"pending": 2}
