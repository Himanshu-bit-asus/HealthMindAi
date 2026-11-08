"""GRPO reward functions (TRL style: take completions + dataset columns as
kwargs, return one float per completion). These score STRUCTURE, evidence
discipline and escalation only. They do NOT verify clinical correctness, which
needs clinician-validated evaluation sets. Never weaken them to speed training."""
from __future__ import annotations
import json, re

REQUIRED_KEYS = ("answer", "evidence_ids", "uncertainty", "escalation_required")
OVERCONFIDENT = re.compile(
    r"\b(definitely|guaranteed|certainly|100\s*%|without (any )?doubt|you (do )?have)\b", re.I)


def _text(c) -> str:
    if isinstance(c, str): return c
    if isinstance(c, list) and c: return str(c[-1].get("content", ""))   # chat format
    return ""


def parse_answer(completion) -> dict | None:
    t = _text(completion).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        d = json.loads(t)
    except Exception:
        return None
    if not isinstance(d, dict) or any(k not in d for k in REQUIRED_KEYS): return None
    if not isinstance(d["answer"], str) or not isinstance(d["uncertainty"], str): return None
    if not isinstance(d["evidence_ids"], list) or not all(isinstance(x, str) for x in d["evidence_ids"]): return None
    if not isinstance(d["escalation_required"], bool): return None
    return d


def format_reward(completions, **kw):
    return [1.0 if parse_answer(c) else -1.0 for c in completions]


def evidence_reward(completions, allowed_evidence_ids=None, required_evidence_ids=None, **kw):
    n = len(completions)
    allowed_evidence_ids = allowed_evidence_ids or [[]] * n
    required_evidence_ids = required_evidence_ids or [[]] * n
    out = []
    for c, allowed, req in zip(completions, allowed_evidence_ids, required_evidence_ids):
        d = parse_answer(c)
        if d is None:
            out.append(-1.0); continue
        cited, allowed, req = set(d["evidence_ids"]), set(allowed), set(req)
        if cited - allowed:                       # fabricated / out-of-scope reference
            out.append(-1.0); continue
        if not req:
            out.append(1.0 if not cited or cited <= allowed else -1.0); continue
        hit = len(cited & req)
        p = hit / len(cited) if cited else 0.0
        r = hit / len(req)
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        out.append(2.0 * f1 - 1.0)
    return out


def escalation_reward(completions, must_escalate=None, **kw):
    if must_escalate is None: return [0.0] * len(completions)
    out = []
    for c, exp in zip(completions, must_escalate):
        d = parse_answer(c)
        if d is None: out.append(-1.0)
        elif exp and not d["escalation_required"]: out.append(-2.0)   # missed escalation costs more
        else: out.append(1.0 if d["escalation_required"] == bool(exp) else -1.0)
    return out


def uncertainty_reward(completions, **kw):
    out = []
    for c in completions:
        d = parse_answer(c)
        if d is None: out.append(-1.0); continue
        score = 0.5 if d["uncertainty"].strip() else -0.5
        if OVERCONFIDENT.search(d["answer"]): score -= 1.0
        out.append(score)
    return out


REWARD_FUNCS = [format_reward, evidence_reward, escalation_reward, uncertainty_reward]
