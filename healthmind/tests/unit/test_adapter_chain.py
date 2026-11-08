import pytest
from training.adapter_chain import (AdapterChainError, write_manifest, verify_parent, hash_dir)

BASE = "Intelligent-Internet/II-Medical-8B"

def make(tmp_path, name, content=b"w"):
    d = tmp_path / name; d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(content)
    return d

def test_valid_chain_sft_dpo_grpo(tmp_path):
    sft = make(tmp_path, "sft")
    m1 = write_manifest(sft, stage="sft", base_model=BASE, parent_stage="base",
                        parent_hash=None, dataset_version="v1", seed=1)
    assert verify_parent("dpo", sft, BASE).adapter_hash == m1.adapter_hash
    dpo = make(tmp_path, "dpo", b"x")
    write_manifest(dpo, stage="dpo", base_model=BASE, parent_stage="sft",
                   parent_hash=m1.adapter_hash, dataset_version="v1", seed=1)
    assert verify_parent("grpo", dpo, BASE).stage == "dpo"

def test_dpo_may_skip_sft():
    assert verify_parent("dpo", None, BASE) is None

def test_grpo_cannot_start_from_base():
    with pytest.raises(AdapterChainError, match="refusing to start from base"):
        verify_parent("grpo", None, BASE)

def test_grpo_cannot_follow_sft(tmp_path):
    sft = make(tmp_path, "sft")
    write_manifest(sft, stage="sft", base_model=BASE, parent_stage="base",
                   parent_hash=None, dataset_version="v1", seed=1)
    with pytest.raises(AdapterChainError, match="cannot use a sft"):
        verify_parent("grpo", sft, BASE)

def test_tampered_parent_detected(tmp_path):
    dpo = make(tmp_path, "dpo")
    write_manifest(dpo, stage="dpo", base_model=BASE, parent_stage="base",
                   parent_hash=None, dataset_version="v1", seed=1)
    (dpo / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(AdapterChainError, match="do not match"):
        verify_parent("grpo", dpo, BASE)

def test_base_model_mismatch(tmp_path):
    dpo = make(tmp_path, "dpo")
    write_manifest(dpo, stage="dpo", base_model=BASE, parent_stage="base",
                   parent_hash=None, dataset_version="v1", seed=1)
    with pytest.raises(AdapterChainError, match="Base model mismatch"):
        verify_parent("grpo", dpo, "other/model")

def test_missing_manifest(tmp_path):
    d = make(tmp_path, "x")
    with pytest.raises(AdapterChainError, match="Missing"):
        verify_parent("grpo", d, BASE)

def test_hash_is_deterministic(tmp_path):
    d = make(tmp_path, "a")
    assert hash_dir(d) == hash_dir(d)


def test_quantization_recorded_and_mismatch_blocked(tmp_path):
    d = make(tmp_path, "sft")
    write_manifest(d, stage="sft", base_model=BASE, parent_stage="base", parent_hash=None,
                   dataset_version="v1", seed=1, quantization="nf4")
    assert verify_parent("dpo", d, BASE, quantization="nf4").quantization == "nf4"
    with pytest.raises(AdapterChainError, match="Quantization mismatch"):
        verify_parent("dpo", d, BASE, quantization="none")

def test_old_manifest_without_field_reads_as_none(tmp_path):
    import json
    d = make(tmp_path, "old")
    write_manifest(d, stage="dpo", base_model=BASE, parent_stage="base", parent_hash=None,
                   dataset_version="v1", seed=1)
    m = json.loads((d / "adapter_manifest.json").read_text()); m.pop("quantization")
    (d / "adapter_manifest.json").write_text(json.dumps(m))
    assert verify_parent("grpo", d, BASE, quantization="none").quantization == "none"
