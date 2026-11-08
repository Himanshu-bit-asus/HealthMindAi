"""Pick numeric precision / attention from GPU compute capability.
V100 (capability 7.0) has no native bf16 and no FlashAttention-2; A10G/A100/H100 (>=8.0) do."""
def pick_precision(capability_major: int, flash_attn_installed: bool = False) -> dict:
    if capability_major >= 8:
        return {"dtype": "bfloat16", "bf16": True, "fp16": False,
                "attn": "flash_attention_2" if flash_attn_installed else "sdpa"}
    return {"dtype": "float16", "bf16": False, "fp16": True, "attn": "sdpa"}

def detect() -> dict:
    import torch
    cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    try:
        import flash_attn  # noqa: F401
        fa = True
    except Exception:
        fa = False
    return pick_precision(cap, fa)
