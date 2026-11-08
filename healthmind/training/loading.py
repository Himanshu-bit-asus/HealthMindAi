"""Model loading with optional QLoRA (4-bit NF4 base + LoRA adapters).
quant_plan is pure and unit-tested; load_model needs GPU + bitsandbytes (UNTESTED here)."""
from __future__ import annotations
import os


def quant_plan(cfg: dict, hw: dict) -> dict:
    if not cfg.get("qlora"):
        return {"qlora": False, "label": "none"}
    return {"qlora": True, "label": "nf4",
            "bnb": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": hw["dtype"]}}


def load_model(cfg: dict, hw: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    plan = quant_plan(cfg, hw)
    local = int(os.environ.get("LOCAL_RANK", 0))
    kw = dict(torch_dtype=getattr(torch, hw["dtype"]), attn_implementation=hw["attn"])
    if plan["qlora"]:
        b = dict(plan["bnb"]); b["bnb_4bit_compute_dtype"] = getattr(torch, b["bnb_4bit_compute_dtype"])
        kw.update(quantization_config=BitsAndBytesConfig(**b),
                  device_map={"": local})            # one full quantised copy per DDP rank
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kw)
    if plan["qlora"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model
