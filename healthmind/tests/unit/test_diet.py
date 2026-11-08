import pytest
from tools.diet_calculator import plan, bmr_mifflin

def test_mifflin_known_value():
    # male 30y 80kg 180cm: 10*80 + 6.25*180 - 150 + 5 = 1780
    assert bmr_mifflin("male", 80, 180, 30) == 1780

def test_plan_energy_balance():
    p = plan("male", 30, 80, 180, "moderate", "maintain")
    assert p["tdee"] == round(1780 * 1.55)
    kcal = p["protein_g"] * 4 + p["carb_g"] * 4 + p["fat_g"] * 9
    assert abs(kcal - p["target_kcal"]) < 20
    assert not p["needs_clinician_review"]

def test_calorie_floor():
    assert plan("female", 40, 45, 150, "sedentary", "lose")["target_kcal"] >= 1200

def test_flags():
    p = plan("female", 30, 70, 165, conditions=["Diabetes"], pregnant=True)
    assert p["needs_clinician_review"]
    assert "pregnancy" in p["flags"] and "condition:diabetes" in p["flags"]

def test_minor_flagged():
    assert "minor" in plan("male", 15, 55, 165)["flags"]

@pytest.mark.parametrize("kw", [dict(age=5), dict(weight_kg=5), dict(sex="x"), dict(activity="zzz")])
def test_invalid_inputs(kw):
    args = dict(sex="male", age=30, weight_kg=80, height_cm=180); args.update(kw)
    with pytest.raises(ValueError): plan(**args)
