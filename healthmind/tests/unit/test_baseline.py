import math
from evaluation.baseline import parse_choice, parse_conf, ece, run_mcq

def test_parse():
    assert parse_choice("Answer: C") == "C" and parse_choice("The answer is (b).") == "B"
    assert parse_choice("no idea") is None
    assert parse_conf("Confidence: 80%") == 0.8 and parse_conf("confidence 0.7") == 0.7
    assert parse_conf("nothing") is None

def test_ece_perfect_and_overconfident():
    assert ece([1.0, 1.0], [1, 1]) == 0.0
    assert math.isclose(ece([0.9] * 10, [1] * 5 + [0] * 5), 0.4)

ITEMS = [{"question": "q1", "options": {"A": "x", "B": "y"}, "answer": "A", "subgroup": "adult"},
         {"question": "q2", "options": {"A": "x", "B": "y"}, "answer": "B", "subgroup": "child"}]

def test_run_mcq():
    r = run_mcq(ITEMS, lambda p: "Answer: A\nConfidence: 90%")
    assert r["clinical_accuracy"] == 0.5
    assert r["subgroups"] == {"adult": 1.0, "child": 0.0}
    assert math.isclose(r["calibration_ece"], 0.4)

def test_unparseable_counts():
    r = run_mcq(ITEMS, lambda p: "I cannot say")
    assert r["unparseable_rate"] == 1.0 and r["calibration_ece"] is None

def test_strip_reasoning():
    from evaluation.baseline import strip_reasoning
    assert strip_reasoning("<think>maybe A or B</think>Answer: B") == "Answer: B"
    assert strip_reasoning("<think>still thinking A") == ""
    assert strip_reasoning("Answer: C") == "Answer: C"

def test_reasoning_letters_not_scored():
    out = lambda p: "<think>It could be A. Answer: A?</think>Answer: B\nConfidence: 70%"
    assert run_mcq([ITEMS[1]], out)["clinical_accuracy"] == 1.0
