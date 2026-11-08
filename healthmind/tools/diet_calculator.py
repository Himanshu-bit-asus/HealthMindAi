"""Deterministic calorie/macro calculator. The LLM turns these numbers into a
meal plan; it never does the arithmetic. Output is general guidance, and cases
in the flags list must be reviewed by a clinician/dietitian."""
from __future__ import annotations

ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55,
            "active": 1.725, "very_active": 1.9}
GOAL_ADJ = {"lose": -0.15, "maintain": 0.0, "gain": 0.10}
RISK_CONDITIONS = {"diabetes", "kidney_disease", "liver_disease",
                   "heart_disease", "eating_disorder", "hypertension"}


def bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def bmr_mifflin(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + (5 if sex == "male" else -161)


def plan(sex, age, weight_kg, height_cm, activity="moderate", goal="maintain",
         conditions=(), pregnant=False) -> dict:
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    if not (10 <= age <= 100) or not (25 <= weight_kg <= 300) or not (100 <= height_cm <= 230):
        raise ValueError("input outside plausible range")
    if activity not in ACTIVITY or goal not in GOAL_ADJ:
        raise ValueError("unknown activity or goal")

    b = bmi(weight_kg, height_cm)
    bmr = bmr_mifflin(sex, weight_kg, height_cm, age)
    tdee = bmr * ACTIVITY[activity]
    floor = 1500 if sex == "male" else 1200          # never plan below floor
    kcal = max(tdee * (1 + GOAL_ADJ[goal]), floor)

    protein_g = round((1.6 if goal != "maintain" else 1.2) * weight_kg)
    fat_g = round(kcal * 0.27 / 9)
    carb_g = max(0, round((kcal - protein_g * 4 - fat_g * 9) / 4))

    flags = []
    if age < 18: flags.append("minor")
    if age >= 65: flags.append("older_adult")
    if pregnant: flags.append("pregnancy")
    if b < 18.5 or b >= 35: flags.append("bmi_extreme")
    flags += [f"condition:{c.lower()}" for c in conditions if c.lower() in RISK_CONDITIONS]
    return {"bmi": round(b, 1), "bmr": round(bmr), "tdee": round(tdee),
            "target_kcal": round(kcal), "protein_g": protein_g, "fat_g": fat_g,
            "carb_g": carb_g, "needs_clinician_review": bool(flags), "flags": flags}
