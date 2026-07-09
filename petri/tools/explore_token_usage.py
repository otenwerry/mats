"""Token usage by model role (auditor/target/judge) per run dir. Free, local-only:
reads eval-log headers (log.stats.model_usage), no API calls."""

import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

LOGS = Path("/Users/owenterry/supermats/mats-local/petri/logs")
DIRS = [
    "v2-5targets-3ep-20260629-225553",   # Opus auditor baseline
    "v2-5targets-1ep-20260629-183349",
    "v2-4targets-1ep-20260702-143507",   # GLM auditor
    "v2-4targets-1ep-20260702-152119",   # DeepSeek auditor
    "v2-4targets-1ep-20260702-160733",   # Kimi auditor
]

for d in DIRS:
    per_model = defaultdict(lambda: [0, 0, 0, 0])  # model -> [input, output, cache_read, cache_write]
    n = 0
    roles = {}
    for f in sorted((LOGS / d).glob("*.eval")):
        log = read_eval_log(str(f), header_only=True)
        n += len(log.eval.dataset.sample_ids or []) * (log.eval.config.epochs or 1)
        for role, spec in (log.eval.model_roles or {}).items():
            roles[getattr(spec, "model", str(spec))] = role
        for model, u in (log.stats.model_usage or {}).items():
            row = per_model[model]
            row[0] += u.input_tokens
            row[1] += u.output_tokens
            row[2] += u.input_tokens_cache_read or 0
            row[3] += u.input_tokens_cache_write or 0
    print(f"\n=== {d}  (~{n} trajectories) ===")
    print("  NOTE: auditor+target/judge with the SAME model slug are merged in model_usage")
    for model, (inp, out, cr, cw) in sorted(per_model.items(), key=lambda kv: -(kv[1][0] + kv[1][2])):
        role = roles.get(model, "judge?")
        print(f"  {role:8s} {model:55s} in={inp/1e6:8.1f}M  cache_read={cr/1e6:8.1f}M  cache_write={cw/1e6:7.1f}M  out={out/1e6:6.1f}M")
