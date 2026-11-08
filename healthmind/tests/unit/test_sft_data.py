import pytest
from training.sft_data import prepare, sft_needed, SFTDataError

def ex(i, prov="expert_approved", text=None):
    return {"messages": [{"role": "user", "content": text or f"question number {i} about topic {i} details"},
                         {"role": "assistant", "content": f"answer {i} grounded in evidence {i} thoroughly"}],
            "provenance": prov, "domain": "medical" if i % 2 else "nutrition"}

def many(n=250): return [ex(i) for i in range(n)]

def test_rejects_unapproved_provenance():
    with pytest.raises(SFTDataError, match="lack expert approval"):
        prepare(many(250) + [ex(999, "ai_generated")], [])

def test_minimum_size_guard():
    with pytest.raises(SFTDataError, match="only"):
        prepare(many(50), [])

def test_split_sizes_and_manifest():
    tr, va, rep = prepare(many(250), [], val_fraction=0.1)
    assert len(tr) == 225 and len(va) == 25 and rep["train"] == 225
    assert set(rep["by_domain"]) == {"medical", "nutrition"}

def test_deterministic_and_order_independent():
    a = prepare(many(250), [], seed=1); b = prepare(list(reversed(many(250))), [], seed=1)
    assert a[2]["dataset_version"] == b[2]["dataset_version"] and a[0] == b[0]

def test_seed_changes_split_not_version():
    a, b = prepare(many(250), [], seed=1), prepare(many(250), [], seed=2)
    assert a[0] != b[0] and a[2]["dataset_version"] == b[2]["dataset_version"]

def test_eval_overlap_removed():
    leak = "A 45 year old man presents with chest pain radiating to the left arm and sweating"
    tr, va, rep = prepare(many(250) + [ex(900, text=leak)], [leak])
    assert rep["dropped"]["eval_overlap"] == 1 and len(tr) + len(va) == 250

def test_duplicates_removed():
    tr, va, rep = prepare(many(250) + [ex(0)], [])
    assert rep["dropped"]["duplicate"] == 1

def test_sft_needed_logic():
    need, why = sft_needed({"clinical_accuracy": 0.9, "grounding": 0.9}, {"clinical_accuracy": .7, "grounding": .8})
    assert not need and why == []
    need, why = sft_needed({"clinical_accuracy": 0.6}, {"clinical_accuracy": .7, "grounding": .8})
    assert need and len(why) == 2          # below floor, and missing metric counts as a gap
