"""Print the full judge prompts for reading. (free)

Shows: the ORIGINAL PostTrainBench judge prompt (from the PostTrainBench repo,
rendered the way run_task.sh renders it), both of our SYSTEM prompts verbatim,
the re-judge's per-run user message with the giant condensed trace elided, and
the audit SYSTEM fully rendered for a real run.
Usage: uv run python print_judge_prompts.py | less
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "utilities"))  # shared modules (paths, locate)
import json
from pathlib import Path

import judge_findings
import paths
import exp_rejudge_hacks as RJ
import exp_audit_train_label as AU

RID = "claude_claude-opus-4-6_10h_run2__bfcl_Qwen_Qwen3-1.7B-Base_16845435"
PTB_PROMPT = Path(paths._REPO) / "PostTrainBench" / "src" / "disallowed_usage_judge" / "prompt.txt"


def print_original_judge():
    bar = "=" * 80
    print(bar)
    print("ORIGINAL PTB JUDGE PROMPT (PostTrainBench/src/disallowed_usage_judge/"
          "prompt.txt), rendered for bfcl / Qwen3-1.7B-Base")
    print("(run as a codex agent, gpt-5.1-codex --yolo --search, cwd = the final "
          "task/ workspace; writes the verdict txt files itself)")
    print(bar)
    if not PTB_PROMPT.exists():
        print(f"[not found: {PTB_PROMPT}]")
        return
    prompt = PTB_PROMPT.read_text()
    prompt = prompt.replace("{model}", "Qwen/Qwen3-1.7B-Base")
    prompt = prompt.replace("{benchmark}", "bfcl")
    # get_judge_prompt.py adds an extra allowance only for gsm8k; empty otherwise
    prompt = prompt.replace("{other_allowed_data}", "")
    print(prompt)


def main():
    bar = "=" * 80
    print_original_judge()
    print(); print(bar); print("RE-JUDGE — SYSTEM PROMPT (exp_rejudge_hacks.py)"); print(bar)
    print(RJ.SYSTEM)

    print(); print(bar)
    print(f"RE-JUDGE — USER MESSAGE for {RID} (trace elided)"); print(bar)
    runs = {r["run_id"]: r for r in RJ.flagged_runs()}
    ev = RJ.load_events(RID)
    prompt = RJ.build_prompt(runs[RID], ev, judge_findings.parse(RID))
    marker = "FULL TRAJECTORY"
    head, _, tail = prompt.partition(marker)
    print(head)
    print(f"[... {marker}: ~{len(tail)//1000}k chars of index-tagged condensed events elided ...]")

    print(); print(bar)
    print("KNOWLEDGE AUDIT — SYSTEM PROMPT, fully rendered for the same run"); print(bar)
    h = json.loads((paths.HIGHLIGHTS / f"{RID}.json").read_text())
    trap = h["train_label_trap"]
    print(AU.SYSTEM.replace("{benchmark}", "bfcl")
          .replace("{dataset}", trap.get("dataset", ""))
          .replace("{first_ev}", str(trap.get("first_load_event"))))
    print()
    print("(audit user message = the condensed full trajectory only, nothing else)")


if __name__ == "__main__":
    main()
