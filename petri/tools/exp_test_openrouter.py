"""OpenRouter connectivity + credit check.

Prints your account credit balance and this key's limits (free GET calls), then does ONE tiny
chat completion (costs a fraction of a cent) to confirm the full request path the pipeline uses.
Distinguishes the likely causes of a 402: zero balance vs a per-key spend cap vs a bad key vs a
rate limit.

Run: uv run tools/exp_test_openrouter.py
"""
from pathlib import Path

import httpx

env = Path(__file__).resolve().parent.parent.parent / ".env"   # supermats/mats/.env
key = next(l.split("=", 1)[1].strip().strip('"')
           for l in env.read_text().splitlines() if l.startswith("OPENROUTER_API_KEY"))
H = {"Authorization": f"Bearer {key}"}


def _get(path: str):
    r = httpx.get(f"https://openrouter.ai/api/v1/{path}", headers=H, timeout=30)
    ct = r.headers.get("content-type", "")
    return r.status_code, (r.json() if ct.startswith("application/json") else r.text)


# 1) ACCOUNT credit balance (free GET) — the direct answer to "do I have credits?"
sc, body = _get("credits")
if sc == 200:
    d = body.get("data", {})
    tc, tu = d.get("total_credits"), d.get("total_usage")
    rem = (tc - tu) if (isinstance(tc, (int, float)) and isinstance(tu, (int, float))) else "?"
    print(f"CREDITS: purchased={tc}  used={tu}  REMAINING={rem}")
else:
    print(f"CREDITS check FAILED — HTTP {sc}: {str(body)[:300]}")

# 2) KEY info (free GET) — a per-KEY spend limit can 402 even with account credit left
sc, body = _get("key")
if sc == 200:
    d = body.get("data", {})
    print(f"KEY: label={d.get('label')!r}  spend_limit={d.get('limit')}  usage={d.get('usage')}  "
          f"free_tier={d.get('is_free_tier')}  rate_limit={d.get('rate_limit')}")
elif sc == 401:
    print("KEY: 401 UNAUTHORIZED — OPENROUTER_API_KEY is invalid or revoked.")
else:
    print(f"KEY check FAILED — HTTP {sc}: {str(body)[:300]}")

# 3) tiny COMPLETION — the exact path the pipeline hits (costs ~nothing when it works).
# NB: a HTTP 200 is the real "auth + credits OK" signal. Reasoning models can return 200 with
# content=None (budget spent on reasoning / truncated by max_tokens), so treat any 200 as
# working and report the text only if present -- don't mistake empty content for a failure.
try:
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": "deepseek/deepseek-v4-pro-20260423",
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 32,
        },
        headers=H,
        timeout=60,
    )
    if r.status_code != 200:
        print(f"COMPLETION: NOT WORKING — HTTP {r.status_code}: {r.text[:300]}")
    else:
        j = r.json()
        if j.get("error"):                       # some errors come back as 200 + an error body
            print(f"COMPLETION: NOT WORKING — 200 with error body: {j['error']}")
        else:
            ch = (j.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = (msg.get("content") or "").strip()
            if text:
                print("COMPLETION: WORKING (HTTP 200) — response:", text)
            else:
                extra = " (model emitted only reasoning)" if msg.get("reasoning") else ""
                print(f"COMPLETION: WORKING (HTTP 200) — auth + credits OK; no text returned"
                      f"{extra} (finish_reason={ch.get('finish_reason')!r}).")
except Exception as e:
    print(f"COMPLETION: ERROR contacting OpenRouter — {type(e).__name__}: {e}")
