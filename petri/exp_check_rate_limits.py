"""Print your EXACT current rate limits on Anthropic and OpenRouter.

  uv run exp_check_rate_limits.py                 # both
  uv run exp_check_rate_limits.py --model=claude-opus-4-8
  uv run exp_check_rate_limits.py --skip-anthropic   # OpenRouter only (FREE)

Anthropic: there is no read-only limits endpoint, so this makes ONE tiny generation
(max_tokens=1, ~$0.0001) purely to read the `anthropic-ratelimit-*` response headers —
the live RPM / input-TPM / output-TPM limits AND how much headroom is left right now.
Pass --skip-anthropic to avoid the call. (The plan-level view is also in the Console:
Settings -> Limits.)

OpenRouter: GET /api/v1/key is free — returns your credit limit/usage and surge rate_limit.
CAVEAT: for paid models the limit that actually throttles you under concurrency is usually
the UPSTREAM provider's per-account limit, not this number — watch the per-call
X-RateLimit-* headers during a real run for the binding value.

Keys are read from the same .env the pipeline uses (ANTHROPIC_API_KEY, OPENROUTER_API_KEY).
"""

import os
import pathlib
import sys

import httpx   # verifies TLS via certifi (urllib uses the macOS system store and fails to find CAs)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))
from petri_paths import ENV_FILE
from dotenv import load_dotenv

load_dotenv(ENV_FILE)


def _arg(name, default=None):
    for a in sys.argv[1:]:
        if a == name:
            return True
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def check_anthropic(model: str) -> None:
    print("=" * 70)
    print(f"ANTHROPIC  (reading rate-limit headers via a 1-token call to {model})")
    print("=" * 70)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  ANTHROPIC_API_KEY not set — skipping.\n")
        return
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
        resp = client.messages.with_raw_response.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    except Exception as e:
        print(f"  call failed ({type(e).__name__}: {e})\n")
        return
    rl = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower()}
    if not rl:
        print("  no anthropic-ratelimit-* headers on the response (unexpected).\n")
        return
    for k in sorted(rl):
        print(f"  {k:45s} {rl[k]}")
    if resp.headers.get("retry-after"):
        print(f"  {'retry-after':45s} {resp.headers['retry-after']}")
    print("  (limits = your tier's ceiling; remaining = headroom right now; reset = when it refills)\n")


def check_openrouter() -> None:
    print("=" * 70)
    print("OPENROUTER  (GET /api/v1/key — free)")
    print("=" * 70)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("  OPENROUTER_API_KEY not set — skipping.\n")
        return
    try:
        r = httpx.get("https://openrouter.ai/api/v1/key",
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
    except Exception as e:
        print(f"  request failed ({type(e).__name__}: {e})\n")
        return
    for k in sorted(data):
        print(f"  {k:20s} {data[k]}")
    print("  CAVEAT: for paid models the binding limit is usually the UPSTREAM provider's,")
    print("  not this surge rate_limit — watch per-call X-RateLimit-* headers in a real run.\n")


def main() -> None:
    model = _arg("--model", "claude-opus-4-8")
    if "--skip-anthropic" not in sys.argv:
        check_anthropic(str(model))
    else:
        print("(skipping Anthropic — no billable call made)\n")
    check_openrouter()


if __name__ == "__main__":
    main()
