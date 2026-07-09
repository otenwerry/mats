"""Free/local: back out exact per-token OpenRouter prices from the cost_details OpenRouter
stamped on the (un-condensed) early calls of our eval logs. Reads only the FIRST eval file
per dir and early-stops once every seen model has >=TARGET priced calls, so it's fast.
"""
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

LOGS = Path("/Users/owenterry/supermats/mats-local/petri/logs")
TARGET = 25
rows = defaultdict(list)
cache_examples = {}

dirs = [d for d in sorted(LOGS.iterdir(), key=lambda p: -p.stat().st_mtime) if d.is_dir()]
for d in dirs:
    fs = sorted(d.glob("*.eval"))
    if not fs:
        continue
    try:
        log = read_eval_log(str(fs[0]))
    except Exception as e:
        print(f"  skip {d.name}: {e}")
        continue
    n_new = 0
    for s in (log.samples or []):
        for e in (s.events or []):
            if getattr(e, "event", None) != "model":
                continue
            call = getattr(e, "call", None)
            resp = getattr(call, "response", None) if call else None
            if not isinstance(resp, dict):
                continue
            u = resp.get("usage")
            if not isinstance(u, dict) or u.get("cost") is None:
                continue
            model = getattr(e, "model", resp.get("model", "?"))
            pd = u.get("prompt_tokens_details") or {}
            cd = u.get("cost_details") or {}
            rows[model].append(dict(
                pt=u.get("prompt_tokens", 0), ct=(pd.get("cached_tokens") or 0),
                cw=(pd.get("cache_write_tokens") or 0), comp=u.get("completion_tokens", 0),
                cost=u.get("cost"), pcost=cd.get("upstream_inference_prompt_cost"),
                ccost=cd.get("upstream_inference_completions_cost")))
            n_new += 1
            if (pd.get("cached_tokens") or 0) > 0 and model not in cache_examples:
                cache_examples[model] = {"prompt_details": pd, "cost_details": cd}
    print(f"  {d.name}: +{n_new} priced calls  (models: {sorted(rows)})")
    if rows and all(len(v) >= TARGET for v in rows.values()) and len(rows) >= 5:
        # keep going a couple dirs to catch rarer models, but we have the common ones
        pass

for model in sorted(rows):
    rs = rows[model]
    clean = [r for r in rs if r["ct"] == 0 and r["cw"] == 0 and r["pt"] and r["pcost"]]
    in_rates = sorted(round(r["pcost"] / r["pt"] * 1e6, 4) for r in clean)
    out_rates = sorted(round(r["ccost"] / r["comp"] * 1e6, 4) for r in rs if r["comp"] and r["ccost"])
    in_med = in_rates[len(in_rates) // 2] if in_rates else None
    cread = []
    for r in rs:
        if r["ct"] and r["pcost"] is not None and in_med is not None:
            resid = r["pcost"] - (r["pt"] - r["ct"]) * in_med / 1e6
            cread.append(round(resid / r["ct"] * 1e6, 4))
    cread.sort()
    print(f"\n=== {model}  ({len(rs)} priced calls) ===")
    print(f"  input  $/M: median={in_med}  n={len(in_rates)}  lo/hi={(in_rates[:1]+in_rates[-1:]) if in_rates else []}")
    print(f"  output $/M: median={out_rates[len(out_rates)//2] if out_rates else None}  n={len(out_rates)}")
    if cread:
        print(f"  cache_read $/M: median={cread[len(cread)//2]}  n={len(cread)}  lo/hi={cread[:1]+cread[-1:]}")

print("\n=== cache_details shape ===")
for m, ex in cache_examples.items():
    print(f"{m}: prompt_details={ex['prompt_details']} cost_details={ex['cost_details']}")
