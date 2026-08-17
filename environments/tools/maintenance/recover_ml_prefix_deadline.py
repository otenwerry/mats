"""Recover ML prefixes whose final OpenCode response beat the task deadline.

This is a free, local maintenance command. It never calls a model and never edits the
original Inspect logs. It applies the same fail-closed terminal-event rule used by live
runs, rebuilds a validated prefix payload in memory, and stores an explicit recovery
record in that payload. Without ``--apply`` it only reports what would be recovered.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402
from lib.exp_real_continuation import (  # noqa: E402
    build_prefix_spec,
    store_prefix_payload,
    validate_prefix_payload,
)
from lib.interrupted_native_transcript import (  # noqa: E402
    recover_predeadline_opencode_submission,
)
from prefixes.exp_ml_prefix import (  # noqa: E402
    _sample_usage_summary,
    _write_run_payload,
    build_payload,
    payload_name,
)


INDEX_FORMAT = "environments-ml-prefix-deadline-recovery-v1"


def _normalized_user_text(message: Any) -> str:
    text = str(getattr(message, "text", "") or "")
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def _followup_base_messages(sample: Any) -> list[Any]:
    protocol = (sample.metadata or {}).get("protocol") or {}
    followups = protocol.get("follow_up_user_messages") or []
    if len(followups) != 1 or not isinstance(followups[0], str):
        raise ValueError("sample does not contain exactly one stored follow-up prompt")
    followup = followups[0]
    matches = [
        index
        for index, message in enumerate(sample.messages or [])
        if getattr(message, "role", None) == "user"
        and _normalized_user_text(message) == followup
    ]
    if len(matches) != 1:
        raise ValueError(
            "could not identify exactly one follow-up boundary in the transcript"
        )
    boundary = matches[0]
    stored_followup = sample.messages[boundary]
    if not hasattr(stored_followup, "model_copy"):
        raise ValueError("stored follow-up is not an Inspect chat message")
    return [
        *(sample.messages or [])[:boundary],
        # Normalize OpenCode's known literal quote wrapper while retaining the
        # original Inspect message ID. Constructing a fresh ChatMessageUser here
        # would assign a random ID and defeat content-addressed recovery.
        stored_followup.model_copy(update={"content": followup}),
    ]


def _sidecar(run_dir: Path, log: Any, sample: Any) -> Path:
    direct = (
        run_dir
        / "real_artifacts"
        / str(log.eval.task)
        / f"{sample.id}_ep{sample.epoch}"
    )
    if direct.is_dir():
        return direct
    compute = ((sample.metadata or {}).get("real_env") or {}).get("compute") or {}
    cell_id = compute.get("cell_id")
    matches = list(
        (run_dir / "real_artifacts").glob(
            f"*{cell_id or 'no-matching-cell'}*/{sample.id}_ep{sample.epoch}"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"could not identify exactly one native sidecar for {log.eval.task}"
        )
    return matches[0]


def recover_sample_payload(
    run_dir: Path,
    log_path: Path,
    log: Any,
    sample: Any,
) -> tuple[dict | None, dict]:
    real_env = (sample.metadata or {}).get("real_env") or {}
    protocol = real_env.get("protocol") or {}
    harness = real_env.get("harness") or {}
    eval_metadata = log.eval.metadata or {}
    reason = None
    if real_env.get("family") != "ml_prefix_only":
        reason = "not_ml_prefix_only"
    elif eval_metadata.get("experiment") != "ml_prefix_only":
        reason = "missing_ml_prefix_experiment_identity"
    elif harness.get("scaffold") != "opencode":
        reason = "not_opencode"
    elif protocol.get("submissions") != 1:
        reason = "not_one_submission"
    elif protocol.get("follow_up_sent") is not True:
        reason = "follow_up_not_sent"
    elif protocol.get("ended_reason") != "wall_clock_limit":
        reason = "not_wall_clock_limit"
    if reason is not None:
        return None, {"status": "skipped", "reason": reason}

    reset = (real_env.get("clock") or {}).get("second_pass_reset") or {}
    deadline = reset.get("deadline_seconds_from_start")
    base_messages = _followup_base_messages(sample)
    recovered_messages, transcript_record, deadline_record = (
        recover_predeadline_opencode_submission(
            base_messages,
            sample.events or [],
            deadline_seconds_from_start=deadline,
            event_start=0,
            applied_before_judging=False,
            attachments=sample.attachments,
        )
    )
    if deadline_record is None:
        return None, {
            "status": "rejected",
            "reason": "no_proven_predeadline_terminal_response",
        }

    repaired = deepcopy(sample)
    repaired.messages = recovered_messages
    repaired_real_env = repaired.metadata["real_env"]
    original_protocol = deepcopy(repaired_real_env["protocol"])
    repaired_real_env["protocol"].update({
        "submissions": 2,
        "ended_reason": "protocol_end",
    })
    deadline_record = {
        **deadline_record,
        "historical_repair": True,
        "source_run": run_dir.name,
        "source_eval": log_path.name,
        "original_protocol": original_protocol,
    }
    repaired_real_env["native_submission_recovery"] = deadline_record
    repaired_real_env["interrupted_native_transcript"] = transcript_record
    policy = repaired_real_env.get("judge_evidence_policy") or {}
    reasons = policy.setdefault("lossy_reasons", [])
    if "interrupted_native_transcript_reconstructed" not in reasons:
        reasons.append("interrupted_native_transcript_reconstructed")
    policy["lossy"] = bool(reasons)
    submissions = repaired_real_env.get("submission_artifacts") or []
    if not any(item.get("submission") == 2 for item in submissions):
        submissions.append({
            "submission": 2,
            "artifacts": repaired_real_env.get("artifacts") or {},
            "recovered_from_predeadline_terminal_response": True,
        })
    repaired_real_env["submission_artifacts"] = submissions

    target = str(eval_metadata.get("target_name") or "")
    member = str(repaired.id)
    source_epoch = int(
        (repaired_real_env.get("compute") or {}).get("original_epoch")
        or repaired.epoch
    )
    base_name = str(eval_metadata.get("prefix_name_base") or "")
    cfg = {
        "model": target,
        "seed": member,
        "harness": str(eval_metadata.get("harness") or ""),
        "reasoning": bool(eval_metadata.get("reasoning")),
        "run_name": base_name,
        "name": payload_name(base_name, target, member, source_epoch),
    }
    payload = build_payload(
        cfg,
        repaired,
        _sample_usage_summary(cfg, repaired),
        sidecar=_sidecar(run_dir, log, repaired),
    )
    # Content addressing must be idempotent across repeated recovery checks.
    payload["source"]["generated_at"] = deadline_record["terminal_event_timestamp"]
    validate_prefix_payload(payload, origin=str(log_path))
    prefix = build_prefix_spec(payload, harness=cfg["harness"])
    return payload, {
        "status": "recoverable",
        "name": prefix.name,
        "sha256": prefix.sha256,
        "target": target,
        "seed": member,
        "epoch": source_epoch,
        "margin_before_deadline_seconds": deadline_record[
            "margin_before_deadline_seconds"
        ],
    }


def recover_run(run_dir: Path, *, apply: bool) -> dict:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")
    records = []
    recovered_count = 0
    for info in list_eval_logs(str(run_dir)):
        log_path = Path(str(getattr(info, "name", None) or info))
        log = read_eval_log(info, resolve_attachments=True)
        for sample in log.samples or []:
            try:
                payload, record = recover_sample_payload(
                    run_dir, log_path, log, sample
                )
            except Exception as error:
                payload = None
                record = {
                    "status": "error",
                    "reason": f"{type(error).__name__}: {error}",
                }
            record = {
                "task": str(log.eval.task),
                "sample": str(sample.id),
                **record,
            }
            if payload is not None and apply:
                stored = store_prefix_payload(payload)
                run_copy = _write_run_payload(stored, run_dir)
                record.update({
                    "status": "recovered",
                    "payload_file": str(stored),
                    "run_payload_file": str(run_copy),
                })
                recovered_count += 1
            records.append(record)

    result = {
        "format": INDEX_FORMAT,
        "run_dir": str(run_dir),
        "applied": apply,
        "recoverable": sum(
            record["status"] in {"recoverable", "recovered"} for record in records
        ),
        "recovered": recovered_count,
        "records": records,
    }
    if apply:
        index = run_dir / "deadline_recovery_index.json"
        index.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        result["index_file"] = str(index)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="store validated payloads; without this flag the command is read-only",
    )
    args = parser.parse_args()
    print(json.dumps(recover_run(args.run_dir, apply=args.apply), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
