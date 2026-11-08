"""FastAPI router (UNTESTED here: fastapi is not installed in my sandbox; syntax-checked only).
Mount with: app.include_router(router, prefix="/review")
Replace get_reviewer with a dependency that validates the caller's token and loads
(id, role, verified) from YOUR credential tables. Never read role from the request body."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from review.service import ReviewService, Reviewer, ReviewError

router = APIRouter()
service = ReviewService()          # swap InMemoryRepo for a Supabase-backed repo in production


def get_reviewer() -> Reviewer:
    raise HTTPException(501, "wire get_reviewer to your auth (JWT -> reviewer record)")


class Submission(BaseModel):
    decision: str                   # approve | reject | correct
    corrected_answer: str | None = None
    note: str = ""


def _view(item):                    # never leak other reviewers' decisions to a reviewer
    r = item["record"]
    return {"id": item["id"], "question": r["question"], "domain": r["domain"],
            "ai_answer": r["answer"], "evidence_ids": r["evidence_ids"],
            "uncertainty": r["uncertainty"], "safety_critical": r["safety_critical"],
            "provenance": r["provenance"], "lease_until": item["lease_until"]}


@router.get("/next")
def next_item(rv: Reviewer = Depends(get_reviewer)):
    try: item = service.claim_next(rv)
    except ReviewError as e: raise HTTPException(403, str(e))
    return _view(item) if item else {"item": None}


@router.post("/{item_id}/submit")
def submit(item_id: str, body: Submission, rv: Reviewer = Depends(get_reviewer)):
    try: item = service.submit(rv, item_id, body.decision, body.corrected_answer, body.note)
    except (ReviewError, PermissionError, ValueError) as e: raise HTTPException(400, str(e))
    return {"id": item_id, "status": item["status"]}


@router.get("/stats")
def stats(rv: Reviewer = Depends(get_reviewer)):
    return service.stats()
