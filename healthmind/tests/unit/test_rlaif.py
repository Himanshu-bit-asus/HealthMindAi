from data.rlaif import (validate_judgment, route, pair_key, PROVISIONAL, REVIEW, JudgeCache)

def J(w="A", c=0.9, s=False, e=True):
    return validate_judgment({"winner": w, "confidence": c, "safety_issue": s,
                              "evidence_supported": e, "reason": "x"})

def test_validate_rejects_bad_output():
    assert validate_judgment("garbage") is None
    assert validate_judgment({"winner": "C", "confidence": .9, "safety_issue": False, "evidence_supported": True}) is None
    assert validate_judgment({"winner": "A", "confidence": 1.7, "safety_issue": False, "evidence_supported": True}) is None
    assert validate_judgment({"winner": "A", "confidence": .9, "safety_issue": "no", "evidence_supported": True}) is None

def test_validate_accepts_json_string():
    assert J().winner == "A"
    assert validate_judgment('{"winner":"B","confidence":0.9,"safety_issue":false,"evidence_supported":true}').winner == "B"

def test_provisional_when_consistent_and_confident():
    assert route(J("A"), J("B"), False)[0] == PROVISIONAL

def test_position_bias_goes_to_review():
    assert route(J("A"), J("A"), False)[0] == REVIEW     # same label after swap = inconsistent

def test_each_review_trigger():
    assert route(J("A"), J("B"), True)[0] == REVIEW                  # safety-critical topic
    assert route(J("abstain"), J("abstain"), False)[0] == REVIEW
    assert route(J("A", s=True), J("B"), False)[0] == REVIEW
    assert route(J("A", e=False), J("B"), False)[0] == REVIEW
    assert route(J("A", c=.5), J("B"), False)[0] == REVIEW
    assert route(None, J("B"), False)[0] == REVIEW

def test_pair_key_order_independent():
    assert pair_key("q", "a1", "a2") == pair_key("q", "a2", "a1")
    assert pair_key("q", "a1", "a2") != pair_key("q2", "a1", "a2")

def test_cache():
    c = JudgeCache(); k = pair_key("q", "a", "b")
    assert c.get(k) is None; c.put(k, "v"); assert c.get(k) == "v"
