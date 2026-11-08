"""Baseline entry point, data-parallel over all GPUs: run with
  torchrun --nproc_per_node=<gpus> evaluation/baseline_job.py
Each rank loads a full model copy on its GPU and scores items[rank::world].
UNTESTED on GPU (shard/aggregate logic is unit-tested).
Input : /opt/ml/processing/input/eval.jsonl   Output: /opt/ml/processing/output/baseline.json"""
import json, os, time, torch
from pathlib import Path
from evaluation.baseline import score_items_batched, aggregate, shard
from training.hw import detect

MODEL = os.environ.get("BASE_MODEL", "Intelligent-Internet/II-Medical-8B")
REVISION = os.environ.get("MODEL_REVISION")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
BATCH = int(os.environ.get("BATCH_SIZE", "8"))
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))   # >0 = quick smoke test to measure real speed
OUT = Path("/opt/ml/processing/output")

def main():
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group("nccl")
    hw = detect()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=getattr(torch, hw["dtype"]),
        attn_implementation=hw["attn"], device_map={"": local}).eval()

    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    def gen_batch(prompts: list[str]) -> list[str]:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        n = enc["input_ids"].shape[1]
        return [tok.decode(o[n:], skip_special_tokens=True) for o in out]

    items = [json.loads(l) for l in Path("/opt/ml/processing/input/eval.jsonl").read_text().splitlines() if l.strip()]
    if MAX_ITEMS: items = items[:MAX_ITEMS]
    t0 = time.time()
    recs = score_items_batched(shard(items, rank, world), gen_batch, BATCH)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"part-{rank}.json").write_text(json.dumps(
        {"recs": recs, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
         "gpu": torch.cuda.get_device_name(local)}))
    if world > 1: dist.barrier()
    if rank == 0:
        parts = [json.loads((OUT / f"part-{r}.json").read_text()) for r in range(world)]
        res = aggregate([r for p in parts for r in p["recs"]])
        res.update({"model": MODEL, "revision": REVISION, "precision": hw, "gpus": world, "batch_size": BATCH, "max_new_tokens": MAX_NEW,
                    "gpu": parts[0]["gpu"], "wall_seconds": round(time.time() - t0, 1),
                    "peak_vram_gb_per_gpu": [p["peak_vram_gb"] for p in parts]})
        (OUT / "baseline.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    if world > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
