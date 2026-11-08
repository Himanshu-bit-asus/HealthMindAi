"""Prepare SFT data from review-approved records only.
Guards: provenance must be expert_approved/expert_corrected, no eval overlap,
minimum size, deterministic split, content-hash dataset version."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from data.contamination import build_eval_index, ngrams

OK_PROVENANCE = {"expert_approved", "expert_corrected"}
MIN_EXAMPLES = 200


class SFTDataError(RuntimeError): pass


def _text(ex) -> str:
    return " ".join(m["content"] for m in ex["messages"])


def prepare(examples: list[dict], eval_texts: list[str], seed: int = 42,
            val_fraction: float = 0.1, min_examples: int = MIN_EXAMPLES):
    bad = [e for e in examples if e.get("provenance") not in OK_PROVENANCE]
    if bad:
        raise SFTDataError(f"{len(bad)} example(s) lack expert approval (e.g. {bad[0].get('provenance')!r})")
    idx = build_eval_index(eval_texts)
    seen, kept, dropped = set(), [], {"duplicate": 0, "eval_overlap": 0}
    for e in examples:
        key = hashlib.sha256(_text(e).lower().encode()).hexdigest()
        if key in seen: dropped["duplicate"] += 1; continue
        if ngrams(_text(e), 8) & idx: dropped["eval_overlap"] += 1; continue
        seen.add(key); kept.append(e)
    if len(kept) < min_examples:
        raise SFTDataError(f"only {len(kept)} usable examples (< {min_examples}); get more approved data first")
    kept.sort(key=lambda e: hashlib.sha256(_text(e).encode()).hexdigest())   # order-independent
    version = hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()[:12]  # content only, before shuffle
    random.Random(seed).shuffle(kept)
    n_val = max(1, round(len(kept) * val_fraction))
    val, train = kept[:n_val], kept[n_val:]
    return train, val, {"dataset_version": version, "seed": seed, "train": len(train),
                        "val": len(val), "dropped": dropped,
                        "by_domain": {d: sum(e.get("domain") == d for e in kept)
                                      for d in {e.get("domain") for e in kept}}}


def write(train, val, report, outdir):
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        (p / name).write_text("\n".join(json.dumps(r) for r in rows))
    (p / "dataset_manifest.json").write_text(json.dumps(report, indent=2))


def sft_needed(baseline: dict, floors: dict) -> tuple[bool, list[str]]:
    """Skip SFT when the baseline already meets every floor (avoids redundant training)."""
    why = [f"{m}: {baseline.get(m)} < {f}" for m, f in floors.items()
           if baseline.get(m) is None or baseline[m] < f]
    return bool(why), why
