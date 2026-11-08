"""Baseline harness. Model-agnostic: pass generate_fn(prompt)->str.
Measures MCQ accuracy, calibration (ECE), per-subgroup accuracy, abstention.
It does NOT cover hallucination, safety or multilingual quality by itself;
those need their own clinician-built sets."""
from __future__ import annotations
import re
from collections import defaultdict

LETTER = re.compile(r"\b(?:answer\s*(?:is|:)?\s*)?\(?([A-E])\)?\b", re.I)
CONF = re.compile(r"confidence\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(%?)", re.I)

def strip_reasoning(text: str) -> str:
    """Reasoning models emit a <think>...</think> block; score only the final answer."""
    if "</think>" in text: return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text: return ""          # unfinished reasoning = no final answer
    return text


def parse_choice(text: str):
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-E])\b", text, re.I) or LETTER.search(text)
    return m.group(1).upper() if m else None

def parse_conf(text: str):
    m = CONF.search(text)
    if not m: return None
    v = float(m.group(1)); v = v / 100 if (m.group(2) or v > 1) else v
    return v if 0 <= v <= 1 else None

def ece(confs, correct, bins: int = 10) -> float:
    n = len(confs)
    if n == 0: return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total

def shard(items: list, rank: int, world: int) -> list:
    return items[rank::world]


def build_prompt(it: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
    return f"{it['question']}\n{opts}\nGive 'Answer: <letter>' and 'Confidence: <0-100>%'."


def _record(it: dict, raw: str) -> dict:
    out = strip_reasoning(raw)
    ch = parse_choice(out)
    return {"ok": int(ch == it["answer"]), "parsed": ch is not None,
            "conf": parse_conf(out), "subgroup": it.get("subgroup", "all")}


def score_items(items: list[dict], generate_fn) -> list[dict]:
    """items: {'question','options':{'A':..},'answer':'B','subgroup':optional}"""
    return [_record(it, generate_fn(build_prompt(it))) for it in items]


def score_items_batched(items: list[dict], generate_batch_fn, batch_size: int = 8) -> list[dict]:
    """generate_batch_fn(list[str]) -> list[str]. Batching raises throughput on memory-bound GPUs."""
    recs = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        outs = generate_batch_fn([build_prompt(it) for it in chunk])
        if len(outs) != len(chunk): raise ValueError("batch function returned wrong number of outputs")
        recs += [_record(it, o) for it, o in zip(chunk, outs)]
    return recs


def aggregate(recs: list[dict]) -> dict:
    n = len(recs)
    confs = [r["conf"] for r in recs if r["conf"] is not None]
    conf_ok = [r["ok"] for r in recs if r["conf"] is not None]
    by_group = defaultdict(list)
    for r in recs: by_group[r["subgroup"]].append(r["ok"])
    return {
        "n": n,
        "clinical_accuracy": sum(r["ok"] for r in recs) / n if n else float("nan"),
        "unparseable_rate": sum(not r["parsed"] for r in recs) / n if n else float("nan"),
        "calibration_ece": ece(confs, conf_ok) if confs else None,
        "calibration_coverage": len(confs) / n if n else 0.0,
        "subgroups": {g: sum(v) / len(v) for g, v in by_group.items()},
    }


def run_mcq(items: list[dict], generate_fn) -> dict:
    return aggregate(score_items(items, generate_fn))
