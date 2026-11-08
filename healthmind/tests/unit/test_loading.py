from training.loading import quant_plan
from training.hw import pick_precision

def test_no_qlora():
    assert quant_plan({}, pick_precision(8)) == {"qlora": False, "label": "none"}

def test_qlora_plan_follows_hardware_dtype():
    assert quant_plan({"qlora": True}, pick_precision(8))["bnb"]["bnb_4bit_compute_dtype"] == "bfloat16"
    p = quant_plan({"qlora": True}, pick_precision(7))
    assert p["bnb"]["bnb_4bit_compute_dtype"] == "float16" and p["bnb"]["bnb_4bit_quant_type"] == "nf4"
    assert p["bnb"]["load_in_4bit"] and p["label"] == "nf4"
