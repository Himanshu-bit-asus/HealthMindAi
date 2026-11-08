"""OpenAI-compatible teacher client (stdlib only). Features: retries with
backoff on 429/5xx/timeouts, no retry on auth/4xx, on-disk response cache
(identical prompts are never paid for twice), call budget, and a privacy guard
that refuses to send text still containing personal identifiers."""
from __future__ import annotations
import hashlib, json, os, time, urllib.error, urllib.request
from pathlib import Path
from data.ingest import deidentify


class PrivacyError(RuntimeError): pass
class BudgetExceeded(RuntimeError): pass
class TeacherError(RuntimeError): pass


def make_teacher_fn(base_url: str, model: str, api_key_env: str = "JUDGE_API_KEY",
                    cache_dir: str | None = None, max_calls: int | None = None,
                    retries: int = 3, backoff: float = 1.0, timeout: float = 60.0,
                    max_tokens: int = 800):
    key = os.environ.get(api_key_env, "")
    cache = Path(cache_dir) if cache_dir else None
    if cache: cache.mkdir(parents=True, exist_ok=True)
    state = {"calls": 0, "cache_hits": 0}

    def call(prompt: str) -> str:
        if deidentify(prompt) != prompt:
            raise PrivacyError("prompt contains personal identifiers; refusing to send externally")
        h = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
        if cache and (cache / h).exists():
            state["cache_hits"] += 1
            return (cache / h).read_text()
        if max_calls is not None and state["calls"] >= max_calls:
            raise BudgetExceeded(f"call budget {max_calls} reached")
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        last = None
        for attempt in range(retries + 1):
            try:
                state["calls"] += 1
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    text = json.loads(r.read())["choices"][0]["message"]["content"]
                if cache: (cache / h).write_text(text)
                return text
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    raise TeacherError(f"HTTP {e.code} (not retrying)") from None
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError) as e:
                last = type(e).__name__
            if attempt < retries: time.sleep(backoff * 2 ** attempt)
        raise TeacherError(f"teacher failed after {retries + 1} attempts: {last}")

    call.state = state
    return call
