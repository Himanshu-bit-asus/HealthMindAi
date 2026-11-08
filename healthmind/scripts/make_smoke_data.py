"""Generate a tiny SYNTHETIC dataset to test the training code path (loading, loss, VRAM,
checkpoint resume). It contains NO medical content and is NOT training data for HealthMind.
Output dataset_version starts with 'SMOKE' so the adapter chain refuses to use or release it."""
import argparse, json, random
from pathlib import Path

def make(n_train=64, n_val=8, seed=0):
    rng = random.Random(seed)
    def ex():
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        ans = {"answer": f"{a} + {b} = {a + b}.", "evidence_ids": [],
               "uncertainty": "Simple arithmetic; no uncertainty.", "escalation_required": False}
        return {"messages": [{"role": "user", "content": f"What is {a} + {b}? Reply as JSON."},
                             {"role": "assistant", "content": json.dumps(ans)}],
                "provenance": "synthetic_smoke_test", "domain": "smoke"}
    rows = [ex() for _ in range(n_train + n_val)]
    return rows[:n_train], rows[n_train:]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="smoke_data"); a = ap.parse_args()
    tr, va = make(); p = Path(a.out); p.mkdir(parents=True, exist_ok=True)
    (p / "train.jsonl").write_text("\n".join(map(json.dumps, tr)))
    (p / "val.jsonl").write_text("\n".join(map(json.dumps, va)))
    (p / "dataset_manifest.json").write_text(json.dumps(
        {"dataset_version": "SMOKE-TEST-NOT-FOR-RELEASE", "smoke_test": True, "seed": 0,
         "train": len(tr), "val": len(va)}, indent=2))
    print(f"wrote {len(tr)} train / {len(va)} val synthetic examples to {p}")

if __name__ == "__main__":
    main()
