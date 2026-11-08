"""Adapter-chain integrity: base -> optional SFT -> DPO -> GRPO.

Every stage writes a manifest holding the sha256 of its adapter files and the
hash of its parent. A stage refuses to start if the parent is missing, has the
wrong stage, or its hash no longer matches (prevents silently restarting from
the base model or using a stale/corrupt adapter).
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

MANIFEST = "adapter_manifest.json"
ALLOWED_PARENTS = {
    "sft": {"base"},
    "dpo": {"base", "sft"},      # SFT is optional
    "grpo": {"dpo"},             # GRPO must follow DPO
}


class AdapterChainError(RuntimeError):
    pass


@dataclass
class Manifest:
    stage: str
    base_model: str
    adapter_hash: str
    parent_stage: str
    parent_hash: str | None
    dataset_version: str
    seed: int
    quantization: str = "none"      # none | nf4 (old manifests default to none)


def is_smoke(m: "Manifest") -> bool:
    return str(m.dataset_version).upper().startswith("SMOKE")


def assert_releasable(adapter_dir) -> None:
    """Call before registering/deploying. Smoke-test adapters are never releasable."""
    m = read_manifest(adapter_dir)
    if is_smoke(m): raise AdapterChainError("smoke-test adapter cannot be released or registered")


def hash_dir(path: str | Path) -> str:
    """Deterministic sha256 over all files (names + bytes), manifest excluded."""
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name != MANIFEST)
    if not files:
        raise AdapterChainError(f"No adapter files found in {path}")
    for f in files:
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def write_manifest(adapter_dir, *, stage, base_model, parent_stage,
                   parent_hash, dataset_version, seed, quantization="none") -> Manifest:
    if stage not in ALLOWED_PARENTS:
        raise AdapterChainError(f"Unknown stage {stage!r}")
    if parent_stage not in ALLOWED_PARENTS[stage]:
        raise AdapterChainError(f"{stage} cannot follow {parent_stage}")
    m = Manifest(stage, base_model, hash_dir(adapter_dir), parent_stage,
                 parent_hash, dataset_version, seed, quantization)
    (Path(adapter_dir) / MANIFEST).write_text(json.dumps(asdict(m), indent=2))
    return m


def read_manifest(adapter_dir) -> Manifest:
    p = Path(adapter_dir) / MANIFEST
    if not p.exists():
        raise AdapterChainError(f"Missing {MANIFEST} in {adapter_dir}")
    return Manifest(**json.loads(p.read_text()))


def verify_parent(stage: str, parent_dir, base_model: str, quantization: str | None = None,
                  allow_smoke: bool = False) -> Manifest | None:
    """Call at the start of a training stage. Returns the parent manifest
    (None when training directly on the base model, allowed for sft/dpo only)."""
    if stage not in ALLOWED_PARENTS:
        raise AdapterChainError(f"Unknown stage {stage!r}")
    if parent_dir is None:
        if "base" not in ALLOWED_PARENTS[stage]:
            raise AdapterChainError(
                f"{stage} requires a parent adapter; refusing to start from base model")
        return None
    pm = read_manifest(parent_dir)
    if pm.stage not in ALLOWED_PARENTS[stage]:
        raise AdapterChainError(f"{stage} cannot use a {pm.stage} adapter as parent")
    if is_smoke(pm) and not allow_smoke:
        raise AdapterChainError("parent adapter was trained on smoke-test data; refusing to continue from it")
    if pm.base_model != base_model:
        raise AdapterChainError(
            f"Base model mismatch: parent={pm.base_model} current={base_model}")
    if quantization is not None and pm.quantization != quantization:
        raise AdapterChainError(
            f"Quantization mismatch: parent={pm.quantization} current={quantization}")
    actual = hash_dir(parent_dir)
    if actual != pm.adapter_hash:
        raise AdapterChainError("Parent adapter files do not match manifest hash")
    return pm
