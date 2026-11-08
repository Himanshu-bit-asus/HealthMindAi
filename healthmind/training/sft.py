"""SFT stage (SKELETON, UNTESTED: needs GPU + pinned TRL/PEFT). Run only if
training.sft_data.sft_needed(...) says the baseline has a real gap.
torchrun --nproc_per_node=<gpus> training/sft.py --config configs/sft.yaml"""
import argparse, json, yaml, torch
from pathlib import Path
from training.adapter_chain import verify_parent, write_manifest
from training.hw import detect
from training.loading import load_model, quant_plan
from training.monitor import make_vram_callback, loss_report

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="/opt/ml/model"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); t = cfg["train"]
    verify_parent("sft", None, cfg["base_model"])
    QL = "nf4" if cfg.get("qlora") else "none"           # SFT starts from base by design
    manifest = json.loads((Path(cfg["dataset_dir"]) / "dataset_manifest.json").read_text())

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(cfg["seed"]); hw = detect()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = load_model(cfg, hw)
    ds = load_dataset("json", data_files={"train": f"{cfg['dataset_dir']}/train.jsonl",
                                          "validation": f"{cfg['dataset_dir']}/val.jsonl"})
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                      task_type="CAUSAL_LM")
    args = SFTConfig(output_dir="/opt/ml/checkpoints", bf16=hw["bf16"], fp16=hw["fp16"], seed=cfg["seed"],
                     per_device_train_batch_size=t["per_device_batch_size"],
                     gradient_accumulation_steps=t["gradient_accumulation_steps"],
                     learning_rate=t["learning_rate"], num_train_epochs=t["num_epochs"],
                     max_length=t["max_length"], packing=t["packing"], save_steps=t["save_steps"],
                     eval_strategy="steps", eval_steps=t["save_steps"],
                     gradient_checkpointing=True, report_to="mlflow")
    # NOTE: verify in your TRL version that loss is computed on assistant tokens only.
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], processing_class=tok, peft_config=lora)
    trainer.add_callback(make_vram_callback())
    ckpt = Path("/opt/ml/checkpoints")
    trainer.train(resume_from_checkpoint=(any(ckpt.glob("checkpoint-*")) if ckpt.exists() else False) or None)
    if trainer.is_world_process_zero():
        rep = loss_report(trainer.state.log_history)
        rep["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "train_report.json").write_text(json.dumps(rep, indent=2)); print(rep)
        out = Path(a.out) / "adapter"; trainer.save_model(str(out))
        write_manifest(out, stage="sft", base_model=cfg["base_model"], parent_stage="base",
                       parent_hash=None, dataset_version=manifest["dataset_version"], seed=cfg["seed"],
                       quantization=QL)

if __name__ == "__main__":
    main()
