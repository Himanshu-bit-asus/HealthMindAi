"""RLAIF routing. AI judgments are weak supervision: nothing here produces a
training pair on its own. Output is 'provisional', which still needs clinician
sampling/approval, or 'clinician_review'."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass

MIN_CONFIDENCE = 0.80
PROVISIONAL, REVIEW = "provisional", "clinician_review"


@dataclass
class Judgment:
    winner: str            # "A" | "B" | "abstain"
    confidence: float
    safety_issue: bool
    evidence_supported: bool
    reason: str


def validate_judgment(raw) -> Judgment | None:
    """Strictly validate judge JSON; anything malformed returns None."""
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        if d["winner"] not in ("A", "B", "abstain"): return None
        c = float(d["confidence"])
        if not 0.0 <= c <= 1.0: return None
        if not isinstance(d["safety_issue"], bool) or not isinstance(d["evidence_supported"], bool): return None
        return Judgment(d["winner"], c, d["safety_issue"], d["evidence_supported"], str(d.get("reason", "")))
    except Exception:
        return None


def pair_key(question: str, a: str, b: str) -> str:
    """Order-independent id so identical candidate pairs are judged only once."""
    parts = sorted([a.strip(), b.strip()])
    return hashlib.sha256(json.dumps([question.strip(), parts]).encode()).hexdigest()


def flip(winner: str) -> str:
    return {"A": "B", "B": "A"}.get(winner, winner)


def route(j: Judgment | None, j_swapped: Judgment | None, safety_critical: bool,
          min_conf: float = MIN_CONFIDENCE) -> tuple[str, str]:
    """j = judged as (A,B); j_swapped = judged with positions swapped (detects position bias)."""
    if j is None or j_swapped is None: return REVIEW, "invalid judge output"
    if safety_critical: return REVIEW, "safety-critical topic"
    if j.winner == "abstain" or j_swapped.winner == "abstain": return REVIEW, "judge abstained"
    if j.winner != flip(j_swapped.winner): return REVIEW, "inconsistent under position swap"
    if j.safety_issue or j_swapped.safety_issue: return REVIEW, "safety issue flagged"
    if not (j.evidence_supported and j_swapped.evidence_supported): return REVIEW, "evidence not supported"
    if min(j.confidence, j_swapped.confidence) < min_conf: return REVIEW, "low confidence"
    return PROVISIONAL, "consistent, confident, evidence-supported"


class JudgeCache:
    """Skip re-judging identical pairs (in-memory; persist to S3/disk in production)."""
    def __init__(self): self._d: dict[str, tuple] = {}
    def get(self, key): return self._d.get(key)
    def put(self, key, value): self._d[key] = value
