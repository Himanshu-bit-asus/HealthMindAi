import json
from rewards.medical_rewards import (format_reward, evidence_reward,
    escalation_reward, uncertainty_reward, parse_answer)

def ans(**o):
    d = {"answer": "HbA1c reflects average glucose over 2-3 months.",
         "evidence_ids": ["REF-001"], "uncertainty": "Depends on clinical context.",
         "escalation_required": False}
    d.update(o); return json.dumps(d)

def test_format():
    assert format_reward([ans(), "not json", '{"answer":"x"}']) == [1.0, -1.0, -1.0]

def test_format_wrong_types():
    assert format_reward([ans(escalation_required="no")]) == [-1.0]

def test_chat_format_and_fenced_json():
    assert parse_answer([{"role": "assistant", "content": "```json\n" + ans() + "\n```"}])

def test_evidence_correct_and_fabricated():
    kw = dict(allowed_evidence_ids=[["REF-001"]] * 3, required_evidence_ids=[["REF-001"]] * 3)
    r = evidence_reward([ans(), ans(evidence_ids=["REF-999"]), ans(evidence_ids=[])], **kw)
    assert r == [1.0, -1.0, -1.0]

def test_evidence_partial_coverage():
    kw = dict(allowed_evidence_ids=[["R1", "R2"]], required_evidence_ids=[["R1", "R2"]])
    r = evidence_reward([ans(evidence_ids=["R1"])], **kw)[0]
    assert -1.0 < r < 1.0

def test_escalation_missed_costs_more():
    r = escalation_reward([ans(escalation_required=False)] * 2 + [ans(escalation_required=True)],
                          must_escalate=[True, False, True])
    assert r == [-2.0, 1.0, 1.0]

def test_escalation_none_labels():
    assert escalation_reward([ans()]) == [0.0]

def test_uncertainty_penalises_overconfidence():
    r = uncertainty_reward([ans(), ans(answer="You definitely have diabetes."),
                            ans(uncertainty="")])
    assert r[0] > 0 and r[1] < 0 and r[2] < r[0]
