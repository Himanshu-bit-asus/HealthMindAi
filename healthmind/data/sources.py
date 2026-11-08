"""Corpus source registry. Licenses are deliberately 'VERIFY': confirm each
source's current terms yourself before use. Nothing is trainable until
commercial_ok is set to True after you check."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Source:
    name: str
    role: str                 # "rag" | "train" | "eval_only"
    license: str = "VERIFY"
    commercial_ok: bool | None = None   # None = unchecked -> blocked

SOURCES = {s.name: s for s in [
    Source("pmc_open_access", "rag"),
    Source("medlineplus", "rag"),
    Source("dailymed_labels", "rag"),
    Source("openstax_anatomy", "rag"),
    Source("usda_fooddata", "rag"),
    Source("ifct_india", "rag"),
    Source("medqa", "eval_only"),
    Source("medmcqa", "eval_only"),
    Source("pubmedqa", "eval_only"),
]}

def allowed(source: str, purpose: str, accept_unverified: bool = False) -> bool:
    """purpose: 'rag' or 'train'. eval_only sources are never usable for either."""
    s = SOURCES.get(source)
    if s is None or s.role == "eval_only": return False
    if purpose == "train" and s.role != "train" and s.role != "rag": return False
    return s.commercial_ok is True or accept_unverified
