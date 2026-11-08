"""Expert-review service. Framework-free so it can sit behind your Express/FastAPI
backend. SECURITY: the Reviewer (id, role, verified) must come from your server-side
auth + credential records, never from request-body fields.

Rules enforced here:
- unverified reviewers cannot act; role must match the record's domain
- items are leased so two reviewers don't work the same item
- safety-critical records need TWO independent reviewers who agree
- disagreement -> 'disputed' (needs adjudication by a third reviewer)
- a reviewer can't review the same item twice
- every action is written to an audit log
- only finalised-approved items can be exported for training"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from data.teacher_gen import apply_review, export_sft, DOMAIN_REVIEWERS

LEASE = timedelta(minutes=15)


class ReviewError(RuntimeError): pass


@dataclass(frozen=True)
class Reviewer:
    id: str
    role: str
    verified: bool      # credential verified by your admin process


class InMemoryRepo:
    """Reference repo. Implement the same methods over Supabase/Postgres (see infra/review_schema.sql)."""
    def __init__(self): self.items, self.audit, self._ids = {}, [], itertools.count(1)
    def new_id(self): return f"item-{next(self._ids)}"
    def save(self, item): self.items[item["id"]] = item
    def get(self, item_id):
        if item_id not in self.items: raise ReviewError("unknown item")
        return self.items[item_id]
    def all(self): return list(self.items.values())
    def log(self, event): self.audit.append(event)


class ReviewService:
    def __init__(self, repo=None, now=lambda: datetime.now(timezone.utc)):
        self.repo, self.now = repo or InMemoryRepo(), now

    # ---- helpers
    def _check(self, rv: Reviewer):
        if not rv.verified: raise ReviewError("reviewer credentials not verified")
    def _audit(self, action, rv, item_id, **extra):
        self.repo.log({"at": self.now().isoformat(), "action": action,
                       "reviewer": rv.id if rv else None, "item": item_id, **extra})
    def _eligible(self, item, rv):
        rec = item["record"]
        return (item["status"] == "pending" and rv.role in DOMAIN_REVIEWERS[rec["domain"]]
                and rv.id not in {r["review"]["reviewer_id"] for r in item["reviews"]}
                and (item["lease_holder"] in (None, rv.id) or item["lease_until"] <= self.now()))

    # ---- API
    def enqueue(self, records) -> list[str]:
        ids = []
        for rec in records:
            item = {"id": self.repo.new_id(), "record": rec, "status": "pending",
                    "needed": 2 if rec["safety_critical"] else 1, "reviews": [],
                    "lease_holder": None, "lease_until": None}
            self.repo.save(item); ids.append(item["id"]); self._audit("enqueue", None, item["id"])
        return ids

    def claim_next(self, rv: Reviewer):
        self._check(rv)
        cands = [i for i in self.repo.all() if self._eligible(i, rv)]
        if not cands: return None
        cands.sort(key=lambda i: (not i["record"]["safety_critical"], i["id"]))   # safety first
        item = cands[0]
        item["lease_holder"], item["lease_until"] = rv.id, self.now() + LEASE
        self.repo.save(item); self._audit("claim", rv, item["id"])
        return item

    def submit(self, rv: Reviewer, item_id, decision, corrected_answer=None, note=""):
        self._check(rv)
        item = self.repo.get(item_id)
        if item["status"] != "pending": raise ReviewError(f"item is {item['status']}")
        if item["lease_holder"] != rv.id or item["lease_until"] <= self.now():
            raise ReviewError("you do not hold a valid lease on this item")
        if rv.id in {r["review"]["reviewer_id"] for r in item["reviews"]}:
            raise ReviewError("reviewer already reviewed this item")
        reviewed = apply_review(item["record"], decision, rv.id, rv.role, corrected_answer, note)  # role check
        item["reviews"].append(reviewed)
        item["lease_holder"] = item["lease_until"] = None
        if len(item["reviews"]) >= item["needed"]: self._finalise(item)
        self.repo.save(item); self._audit("submit", rv, item_id, decision=decision, status=item["status"])
        return item

    @staticmethod
    def _sig(r):
        return (r["review_status"], " ".join((r["answer"] if r["provenance"] == "expert_corrected" else "").lower().split()))

    def _finalise(self, item):
        sigs = {self._sig(r) for r in item["reviews"]}
        if len(sigs) > 1: item["status"] = "disputed"; return
        first = item["reviews"][0]
        item["status"] = first["review_status"]          # approved | rejected
        item["final"] = first

    def adjudicate(self, rv: Reviewer, item_id, decision, corrected_answer=None, note=""):
        self._check(rv)
        item = self.repo.get(item_id)
        if item["status"] != "disputed": raise ReviewError("item is not disputed")
        if rv.id in {r["review"]["reviewer_id"] for r in item["reviews"]}:
            raise ReviewError("adjudicator must be a different reviewer")
        final = apply_review(item["record"], decision, rv.id, rv.role, corrected_answer, note)
        item["reviews"].append(final); item["final"] = final; item["status"] = final["review_status"]
        self.repo.save(item); self._audit("adjudicate", rv, item_id, decision=decision)
        return item

    def stats(self):
        out = {}
        for i in self.repo.all(): out[i["status"]] = out.get(i["status"], 0) + 1
        return out

    def export_training(self):
        return export_sft([i["final"] for i in self.repo.all() if i["status"] == "approved"])
