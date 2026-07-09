"""Per-model thinking-tag usage report over the CURRENT sweep (sweep 5).

Read-only: reads .eval logs, counts the target's <thinking> scratchpad tags and any
native reasoning blocks. No API calls.

Sweep 5 = every audit run dir (name starts with "v2-") NOT in the retired sweeps 1-4
explicit sets (mirrors make_viewer.SWEEPS catch-all). Continuation/resample/rollback
dirs are excluded (they're derived runs, not fresh audits).

Metrics per target model, across all its sweep-5 trajectories (= samples):
  - n trajectories
  - % of trajectories with >=1 <thinking> tag
  - among those, mean/median count of <thinking> tags per trajectory
  - mean target turns per trajectory (so thinking-per-turn is interpretable)
  - native reasoning blocks seen (ContentReasoning) -- should be ~0 (reasoning is OFF)
"""
import glob
import os
import re
import statistics
from collections import defaultdict

from inspect_ai.log import read_eval_log

LOGS = "/Users/owenterry/supermats/mats-local/petri/logs"

# Retired sweeps 1-4 explicit dir sets (from make_viewer.SWEEPS). Everything else that
# starts with "v2-" is the current catch-all sweep (sweep 5 / index.html).
RETIRED = {
    # sweep 4 "auditors"
    "v2-4targets-1ep-20260702-143507", "v2-4targets-1ep-20260702-143741",
    "v2-4targets-1ep-20260702-152119", "v2-4targets-1ep-20260702-160733",
    "v2-4targets-2ep-20260702-143430", "v2-4targets-allow-5ep-20260703-160853",
    "v2-4targets-allow-5ep-20260703-222430", "v2-4targets-allow-5ep-20260703-222456",
    # sweep 3 "fixed_sp"
    "v2-5targets-1ep-20260629-183349", "v2-5targets-3ep-20260629-225553",
    # sweep 2 "opus"
    "v2-5targets-1ep-20260629-163441", "v2-6targets-1ep-20260626-145400",
    "v2-6targets-1ep-20260626-164347", "v2-6targets-1ep-20260626-231246",
    "v2-deepseek-v4-pro-1ep-20260625-193414", "v2-kimi-k2.6-1ep-20260625-222445",
    # sweep 1 "sonnet"
    "full-v2", "pilot", "pilot-v2-opus-4-6",
    "v2-3targets-1ep-20260617-213105", "v2-3targets-1ep-20260617-230739",
    "v2-3targets-2ep-20260618-224209", "v2-6targets-2ep-20260623-233416",
    "v2-opus-4-6-2ep-20260617-184240", "v2-qwen3-32b-1ep-20260622-170315",
    "v2-qwen3-32b-1ep-20260623-232217", "v2-qwen3.7-max-1ep-20260617-173841",
    "v2-qwen3.7-max-2ep-20260617-175944",
}

# Match BOTH the prompt's <thinking> and the open models' native <think> opener, so
# a model that reasons in its native tag isn't miscounted as "no thinking".
THINK_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)


def sweep5_dirs():
    out = []
    for p in sorted(glob.glob(LOGS + "/*/")):
        name = os.path.basename(p.rstrip("/"))
        if name.startswith("v2-") and name not in RETIRED:
            out.append(p.rstrip("/"))
    return out


def target_text_and_reasoning(msg):
    """Return (concatenated ContentText, n_reasoning_blocks) for a target assistant msg."""
    c = msg.content
    if isinstance(c, str):
        return c, 0
    text_parts, n_reason = [], 0
    for b in c:
        t = type(b).__name__
        if t == "ContentText":
            text_parts.append(b.text or "")
        elif t == "ContentReasoning":
            n_reason += 1
    return "\n".join(text_parts), n_reason


def has_tool_calls(msg):
    tc = getattr(msg, "tool_calls", None)
    return bool(tc)


def main():
    dirs = sweep5_dirs()
    print(f"Sweep-5 audit dirs ({len(dirs)}):")
    for d in dirs:
        print("  ", os.path.basename(d))
    print()

    # per model: list of per-trajectory dicts
    per_model = defaultdict(list)
    contributing_dirs = defaultdict(set)

    for d in dirs:
        evals = sorted(glob.glob(d + "/*.eval"))
        if not evals:
            continue
        print(f"[reading] {os.path.basename(d)}  ({len(evals)} eval file(s))")
        for f in evals:
            try:
                log = read_eval_log(f)
            except Exception as e:
                print(f"    !! skipping {os.path.basename(f)}: {type(e).__name__}: {e}")
                continue
            tgt_model = (log.eval.metadata or {}).get("target_model") or "UNKNOWN"
            for s in (log.samples or []):
                tgt_events = [e for e in (s.events or [])
                              if type(e).__name__ == "ModelEvent" and e.role == "target"]
                n_turns = len(tgt_events)
                n_think = 0
                n_reason = 0
                chars = 0
                for e in tgt_events:
                    try:
                        msg = e.output.choices[0].message
                    except (IndexError, AttributeError):
                        continue
                    text, nr = target_text_and_reasoning(msg)
                    n_think += len(THINK_RE.findall(text))
                    n_reason += nr
                    chars += len(text)
                per_model[tgt_model].append(
                    {"turns": n_turns, "think": n_think, "reason": n_reason, "chars": chars})
                contributing_dirs[tgt_model].add(os.path.basename(d))

    print("\n" + "=" * 78)
    print("PER-MODEL THINKING-TAG REPORT (current sweep)")
    print("=" * 78)
    header = (f"{'model':<26}{'traj':>5}{'%any':>7}{'think/traj*':>12}"
              f"{'think/turn':>11}{'turns/traj':>11}{'chars/turn':>11}{'nativeCoT':>10}")
    print(header)
    print("-" * 93)
    for model in sorted(per_model):
        rows = per_model[model]
        n = len(rows)
        with_think = [r for r in rows if r["think"] > 0]
        pct_any = 100.0 * len(with_think) / n if n else 0.0
        think_counts = [r["think"] for r in with_think]  # * only among traj that used it
        mean_think = statistics.mean(think_counts) if think_counts else 0.0
        med_think = statistics.median(think_counts) if think_counts else 0.0
        tot_turns = sum(r["turns"] for r in rows)
        tot_think = sum(r["think"] for r in rows)
        think_per_turn = tot_think / tot_turns if tot_turns else 0.0
        mean_turns = tot_turns / n if n else 0.0
        chars_per_turn = sum(r["chars"] for r in rows) / tot_turns if tot_turns else 0.0
        native = sum(r["reason"] for r in rows)
        short = model.split("/")[-1]
        print(f"{short:<26}{n:>5}{pct_any:>6.0f}%"
              f"{mean_think:>8.1f}/{med_think:<3.0f}{think_per_turn:>11.2f}"
              f"{mean_turns:>11.1f}{chars_per_turn:>11.0f}{native:>10}")
    print("-" * 93)
    print("* think/traj = mean/median thinking tags (<thinking> OR native <think>) per")
    print("  trajectory, AMONG trajectories that used one at least once.")
    print("  think/turn = thinking tags / target turn (all traj). chars/turn = mean target")
    print("  text length per turn. nativeCoT = ContentReasoning blocks (reasoning OFF => ~0).")

    # also dump contributing dirs per model for sanity
    print("\nContributing dirs per model:")
    for model in sorted(per_model):
        print(f"  {model.split('/')[-1]}: {sorted(contributing_dirs[model])}")


if __name__ == "__main__":
    main()
