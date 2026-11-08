"""DPO stage (SKELETON, UNTESTED: needs GPU + pinned TRL/PEFT versions).
TRL argument names change between releases; verify against the version you pin.
Launch with: torchrun --nproc_per_node=<gpus> training/dpo.py --config configs/dpo.yaml
"""
import argparse, os, yaml, torch
from pathlib import Path
from training.adapter_chain import verify_parent, write_manifest, hash_dir
from training.hw import detect
from training.loading import load_model, quant_plan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--train_data", default="/opt/ml/input/data/train")
    ap.add_argument("--out", default="/opt/ml/model")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); t = cfg["train"]

    # 1. Chain check BEFORE loading anything heavy.
    QL = "nf4" if cfg.get("qlora") else "none"
    parent = verify_parent("dpo", cfg["parent_adapter"], cfg["base_model"], quantization=QL)

    from datasets import load_from_disk
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer
    set_seed(cfg["seed"]); hw = detect()

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = load_model(cfg, hw)
    if parent is not None:   # continue from the SFT adapter, never from scratch
        model = PeftModel.from_pretrained(model, cfg["parent_adapter"], is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                                 lora_dropout=cfg["lora"]["dropout"],
                                 target_modules=cfg["lora"]["target_modules"],
                                 task_type="CAUSAL_LM")

    ds = load_from_disk(a.train_data)   # pre-tokenised/validated in the data stage
    args = DPOConfig(
        output_dir="/opt/ml/checkpoints", bf16=hw["bf16"], fp16=hw["fp16"], seed=cfg["seed"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], beta=t["beta"], max_length=t["max_length"],
        num_train_epochs=t["num_epochs"], save_steps=t["save_steps"],
        gradient_checkpointing=True, report_to="mlflow")
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)

    ckpt = Path("/opt/ml/checkpoints")
    resume = any(ckpt.glob("checkpoint-*")) if ckpt.exists() else False
    trainer.train(resume_from_checkpoint=resume or None)

    if trainer.is_world_process_zero():
        out = Path(a.out) / "adapter"; trainer.save_model(str(out))
        write_manifest(out, stage="dpo", base_model=cfg["base_model"],
                       parent_stage=parent.stage if parent else "base",
                       parent_hash=parent.adapter_hash if parent else None,
                       dataset_version=cfg["dataset_version"], seed=cfg["seed"], quantization=QL)

if __name__ == "__main__":
    main()
