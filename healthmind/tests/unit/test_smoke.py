import json, pytest
from scripts.make_smoke_data import make
from training.sft_data import prepare, SFTDataError
from training.monitor import loss_report
from training.adapter_chain import (write_manifest, verify_parent, assert_releasable, AdapterChainError)

BASE = "Qwen/Qwen3-0.6B"

def test_smoke_data_deterministic_and_labelled():
    a, b = make(seed=0), make(seed=0)
    assert a == b and len(a[0]) == 64 and len(a[1]) == 8
    assert all(e["provenance"] == "synthetic_smoke_test" for e in a[0])
    json.loads(a[0][0]["messages"][1]["content"])      # assistant reply is valid JSON in our schema

def test_real_sft_pipeline_refuses_smoke_data():
    tr, va = make()
    with pytest.raises(SFTDataError, match="lack expert approval"):
        prepare(tr + va, [], min_examples=1)

def test_loss_report():
    assert loss_report([{"loss": 2.0}, {"loss": 1.0}])["ok"]
    assert loss_report([{"loss": 2.0}, {"loss": float("nan")}])["reason"] == "nan/inf loss"
    assert not loss_report([{"loss": 1.0}, {"loss": 1.5}])["ok"]
    assert not loss_report([{"eval_loss": 1.0}])["ok"]

def mk(tmp_path, version):
    d = tmp_path / "a"; d.mkdir(); (d / "adapter_model.safetensors").write_bytes(b"w")
    write_manifest(d, stage="sft", base_model=BASE, parent_stage="base", parent_hash=None,
                   dataset_version=version, seed=1)
    return d

def test_smoke_adapter_cannot_be_continued_or_released(tmp_path):
    d = mk(tmp_path, "SMOKE-TEST-NOT-FOR-RELEASE")
    with pytest.raises(AdapterChainError, match="smoke-test data"): verify_parent("dpo", d, BASE)
    assert verify_parent("dpo", d, BASE, allow_smoke=True)
    with pytest.raises(AdapterChainError, match="cannot be released"): assert_releasable(d)

def test_real_adapter_is_releasable(tmp_path):
    d = mk(tmp_path, "abc123def456"); assert_releasable(d)
