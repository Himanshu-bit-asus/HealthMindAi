from training.hw import pick_precision
from evaluation.baseline import shard, score_items, aggregate, run_mcq

def test_v100_gets_fp16_sdpa():
    p = pick_precision(7, flash_attn_installed=True)
    assert p == {"dtype": "float16", "bf16": False, "fp16": True, "attn": "sdpa"}

def test_ampere_gets_bf16():
    assert pick_precision(8)["bf16"] and pick_precision(8)["attn"] == "sdpa"
    assert pick_precision(9, True)["attn"] == "flash_attention_2"

def test_no_gpu_falls_back_safely():
    assert pick_precision(0)["fp16"]

ITEMS = [{"question": f"q{i}", "options": {"A": "x", "B": "y"}, "answer": "A" if i % 3 else "B",
          "subgroup": "adult" if i % 2 else "child"} for i in range(23)]
GEN = lambda p: "Answer: A\nConfidence: 80%"

def test_shards_partition_items():
    parts = [shard(ITEMS, r, 8) for r in range(8)]
    flat = [i["question"] for p in parts for i in p]
    assert sorted(flat) == sorted(i["question"] for i in ITEMS) and len(flat) == len(set(flat))

def test_sharded_result_equals_unsharded():
    recs = [r for k in range(8) for r in score_items(shard(ITEMS, k, 8), GEN)]
    assert aggregate(recs) == run_mcq(ITEMS, GEN)

def test_batched_equals_unbatched_and_checks_length():
    from evaluation.baseline import score_items_batched
    import pytest
    batch_gen = lambda ps: [GEN(p) for p in ps]
    assert aggregate(score_items_batched(ITEMS, batch_gen, 5)) == run_mcq(ITEMS, GEN)
    with pytest.raises(ValueError):
        score_items_batched(ITEMS, lambda ps: ["Answer: A"], 4)

def test_l4_gets_bf16():
    assert pick_precision(8)["dtype"] == "bfloat16"     # L4 = compute capability 8.9 -> major 8
