"""One tiny OpenRouter call to check whether the budget cap has been lifted.

Costs a fraction of a cent when it works. Run: uv run tools/exp_test_openrouter.py
"""
from pathlib import Path

import httpx

env = Path(__file__).resolve().parent.parent.parent / ".env"
key = next(l.split("=", 1)[1].strip().strip('"')
           for l in env.read_text().splitlines() if l.startswith("OPENROUTER_API_KEY"))

try:
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": "deepseek/deepseek-v4-pro-20260423",
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 5,
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    if r.status_code == 200:
        print("WORKING — response:", r.json()["choices"][0]["message"]["content"].strip())
    else:
        print(f"NOT WORKING — HTTP {r.status_code}: {r.text[:300]}")
except Exception as e:
    print(f"NOT WORKING — {e}")
