import json, threading, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from data.teacher_client import make_teacher_fn, PrivacyError, BudgetExceeded, TeacherError
from scripts.run_teacher_gen import run

ANSWER = {"answer": "HbA1c reflects average blood glucose exposure over about two to three months.",
          "evidence_ids": ["c1"], "uncertainty": "Depends on context.", "escalation_required": False}

def serve(script):
    """script: list of status codes returned in order (last repeats)."""
    hits = {"n": 0, "auth": None}
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            code = script[min(hits["n"], len(script) - 1)]; hits["n"] += 1
            hits["auth"] = self.headers.get("Authorization")
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
            body = {"choices": [{"message": {"content": json.dumps(ANSWER)}}]} if code == 200 else {"error": "x"}
            self.wfile.write(json.dumps(body).encode())
        def log_message(self, *a): pass
    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}/v1", hits

def test_success_and_auth_header(monkeypatch=None):
    import os; os.environ["JUDGE_API_KEY"] = "secret"
    s, url, hits = serve([200])
    fn = make_teacher_fn(url, "m", backoff=0.01)
    assert json.loads(fn("hello"))["evidence_ids"] == ["c1"] and hits["auth"] == "Bearer secret"
    s.shutdown()

def test_retries_on_503_then_succeeds():
    s, url, hits = serve([503, 503, 200])
    fn = make_teacher_fn(url, "m", backoff=0.01)
    assert fn("x") and hits["n"] == 3
    s.shutdown()

def test_no_retry_on_401():
    s, url, hits = serve([401])
    with pytest.raises(TeacherError):
        make_teacher_fn(url, "m", backoff=0.01)("x")
    assert hits["n"] == 1
    s.shutdown()

def test_gives_up_after_retries():
    s, url, hits = serve([500])
    with pytest.raises(TeacherError):
        make_teacher_fn(url, "m", retries=2, backoff=0.01)("x")
    assert hits["n"] == 3
    s.shutdown()

def test_cache_avoids_second_call():
    s, url, hits = serve([200])
    fn = make_teacher_fn(url, "m", cache_dir=tempfile.mkdtemp(), backoff=0.01)
    fn("same"); fn("same")
    assert hits["n"] == 1 and fn.state["cache_hits"] == 1
    s.shutdown()

def test_budget():
    s, url, hits = serve([200])
    fn = make_teacher_fn(url, "m", max_calls=1, backoff=0.01)
    fn("a")
    with pytest.raises(BudgetExceeded): fn("b")
    s.shutdown()

def test_privacy_guard_blocks_before_network():
    s, url, hits = serve([200])
    with pytest.raises(PrivacyError):
        make_teacher_fn(url, "m")("patient email a.b@x.com has HbA1c 7")
    assert hits["n"] == 0
    s.shutdown()

def test_pipeline_end_to_end_with_fake_teacher():
    chunks = [{"chunk_id": "c1", "text": "HbA1c reflects average blood glucose exposure over approximately two to three months."}]
    qs = [{"question": "What does HbA1c measure?", "domain": "medical"},
          {"question": "zzzz qqqq", "domain": "medical"},
          {"question": "Contact a.b@x.com about HbA1c", "domain": "medical"}]
    def fn(prompt):
        from data.teacher_client import PrivacyError
        from data.ingest import deidentify
        if deidentify(prompt) != prompt: raise PrivacyError("x")
        return json.dumps(ANSWER)
    recs, queue, rej = run(chunks, qs, fn)
    assert len(recs) == 1 and rej.get("no_evidence") == 1
