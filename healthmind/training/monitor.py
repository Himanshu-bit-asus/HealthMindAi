"""Training health checks. loss_report is pure/unit-tested; the VRAM callback needs a GPU."""
import math

def loss_report(log_history: list[dict]) -> dict:
    losses = [h["loss"] for h in log_history if "loss" in h]
    if not losses: return {"n": 0, "ok": False, "reason": "no loss values logged"}
    bad = any(math.isnan(x) or math.isinf(x) for x in losses)
    first, last = losses[0], losses[-1]
    ok = (not bad) and (last < first)
    return {"n": len(losses), "first": first, "last": last, "has_nan_or_inf": bad,
            "decreased": (not bad) and last < first, "ok": ok,
            "reason": "ok" if ok else ("nan/inf loss" if bad else "loss did not decrease")}

def make_vram_callback():
    from transformers import TrainerCallback
    import torch
    class VRAM(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if torch.cuda.is_available():
                gb = torch.cuda.max_memory_allocated() / 2**30
                print(f"[vram] step={state.global_step} peak_allocated_gb={gb:.2f}")
                try:
                    import mlflow; mlflow.log_metric("peak_vram_gb", gb, step=state.global_step)
                except Exception: pass
    return VRAM()
