"""corpus chunks + questions -> pending records + review queue + reject report.
usage: python -m scripts.run_teacher_gen --chunks chunks.jsonl --questions q.jsonl --out out/
questions.jsonl lines: {"question": "...", "domain": "medical|nutrition|fitness|medication"}"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from data.retrieval import BM25
from data.teacher_gen import generate_record, build_review_queue
from data.teacher_client import PrivacyError, BudgetExceeded, TeacherError


def run(chunks, questions, teacher_fn, qc_fraction=0.2, seed=42):
    retriever = BM25(chunks)
    records, rejects = [], Counter()
    for q in questions:
        try:
            rec, why = generate_record(q["question"], q["domain"], retriever, teacher_fn)
        except PrivacyError:
            rejects["privacy_blocked"] += 1; continue
        except TeacherError:
            rejects["teacher_error"] += 1; continue
        except BudgetExceeded:
            rejects["budget_stop"] += 1; break
        if rec: records.append(rec)
        else: rejects[why.split("_0")[0].split("_1")[0]] += 1
    return records, build_review_queue(records, qc_fraction, seed), dict(rejects)


def main():
    from data.teacher_client import make_teacher_fn
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True); ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max_calls", type=int, default=200)
    a = ap.parse_args()
    load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    fn = make_teacher_fn(os.environ["JUDGE_BASE_URL"], os.environ["JUDGE_MODEL"],
                         cache_dir=f"{a.out}/cache", max_calls=a.max_calls)
    records, queue, rejects = run(load(a.chunks), load(a.questions), fn)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    w = lambda n, rows: (out / n).write_text("\n".join(json.dumps(r) for r in rows))
    w("records.jsonl", records); w("review_queue.jsonl", queue)
    (out / "report.json").write_text(json.dumps(
        {"generated": len(records), "in_review_queue": len(queue), "rejected": rejects,
         "api_calls": fn.state["calls"], "cache_hits": fn.state["cache_hits"]}, indent=2))
    print(f"generated={len(records)} queue={len(queue)} rejects={rejects}")

if __name__ == "__main__":
    main()
