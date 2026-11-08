from evaluation.gate import evaluate_gate

GOOD = {"clinical_accuracy": .8, "grounding": .9, "safety_pass_rate": .995,
        "calibration": .8, "multilingual": .7, "hallucination_free": .95}
SUB = {"adult": .8, "child": .75, "elderly": .78, "pregnant": .76}

def test_passes():
    assert evaluate_gate(GOOD, SUB).passed

def test_missing_metric_fails():
    s = dict(GOOD); s.pop("safety_pass_rate")
    r = evaluate_gate(s, SUB)
    assert not r.passed and any("safety_pass_rate" in f for f in r.failures)

def test_safety_floor():
    assert not evaluate_gate({**GOOD, "safety_pass_rate": .97}, SUB).passed

def test_regression_vs_previous():
    r = evaluate_gate({**GOOD, "clinical_accuracy": .75}, SUB, previous=GOOD)
    assert not r.passed and any("regressed" in f for f in r.failures)

def test_no_safety_regression_even_if_above_floor():
    prev = {**GOOD, "safety_pass_rate": 1.0}
    assert not evaluate_gate({**GOOD, "safety_pass_rate": .995}, SUB, previous=prev).passed

def test_subgroup_gap_and_floor():
    assert not evaluate_gate(GOOD, {"a": .9, "b": .7}).passed
    assert not evaluate_gate(GOOD, {"a": .65, "b": .55}).passed

def test_missing_subgroups_fail():
    assert not evaluate_gate(GOOD, None).passed
