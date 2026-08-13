"""Repair saved rejudge scores rejected only by corrected post-validation.

This is a free, local repair. It never calls a model. The original paid provider
response remains in the Inspect events, and applying a repair first makes a byte-for-
byte backup beside the log.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_ai.log import read_eval_log, resolve_sample_attachments, write_eval_log

from lib.environment_judge.evidence import EvidenceBundle
from lib.environment_judge.exp_runner import validate_judgment_evidence
from lib.environment_judge.schema import result_model_for_family
from lib.environment_judge.score import score_from_judgment


SCORE_KEY = "environment_judge"
REPAIR_VERSION = "prose-validation-guidance-v2"


def _submitted_answer(sample) -> dict:
    calls = [
        call
        for event in sample.events or []
        if getattr(event, "event", None) == "model"
        for choice in (getattr(event, "output", None).choices or [])
        for call in (choice.message.tool_calls or [])
        if call.function == "submit_judgment"
    ]
    if not calls:
        raise ValueError("sample has no stored submit_judgment tool call")
    arguments = calls[-1].arguments
    if not isinstance(arguments, dict):
        raise ValueError("stored submit_judgment arguments are not an object")
    return arguments


def _repaired_score(raw_sample):
    old_score = (raw_sample.scores or {}).get(SCORE_KEY)
    if old_score is None:
        return None
    old_envelope = ((old_score.metadata or {}).get(SCORE_KEY) or {})
    if old_envelope.get("post_validation") != "failed":
        return None
    if "justification must name every active dimension" not in str(
        old_score.explanation or ""
    ):
        return None
    raw_judgment = old_envelope.get("unvalidated_answer")
    if not isinstance(raw_judgment, dict):
        raise ValueError("failed score has no stored unvalidated answer")

    sample = resolve_sample_attachments(raw_sample, "full")
    submitted = _submitted_answer(sample)
    justification = str(submitted.get("explanation") or "").strip()
    family = str(old_envelope.get("family") or "")
    judgment = result_model_for_family(family).model_validate(raw_judgment)
    evidence = EvidenceBundle.model_validate(old_envelope.get("evidence") or {})
    validate_judgment_evidence(
        judgment,
        SimpleNamespace(evidence=evidence),
        justification=justification,
    )

    repaired_envelope = {
        **old_envelope,
        "post_validation": "passed",
        "justification": justification,
        "post_validation_recovery": {
            "version": REPAIR_VERSION,
            "paid_model_call_reused": True,
            "original_score_value": old_score.value,
            "original_explanation": old_score.explanation,
        },
    }
    return score_from_judgment(
        judgment=judgment.model_dump(mode="json"),
        call_metadata={SCORE_KEY: repaired_envelope},
        family=family,
        official_stage="final",
        reused_stage_one=False,
        extra_envelope={
            "judgment_role": old_envelope.get("judgment_role"),
            "retrospective_rejudge": old_envelope.get("retrospective_rejudge"),
        },
    )


def repair_log(
    log_path: Path, *, apply: bool
) -> tuple[int, Path | None, dict[str, str]]:
    log = read_eval_log(str(log_path))
    repaired_scores = {}
    rejected = {}
    repaired_samples = []
    for sample in log.samples or []:
        try:
            repaired = _repaired_score(sample)
        except Exception as error:
            rejected[str(sample.id)] = f"{type(error).__name__}: {error}"
            repaired_samples.append(sample)
            continue
        if repaired is None:
            repaired_samples.append(sample)
            continue
        repaired_scores[str(sample.id)] = repaired
        scores = {**(sample.scores or {}), SCORE_KEY: repaired}
        events = [
            event.model_copy(update={"score": repaired})
            if getattr(event, "event", None) == "score"
            else event
            for event in sample.events or []
        ]
        repaired_samples.append(sample.model_copy(update={
            "scores": scores,
            "events": events,
        }))

    if not repaired_scores or not apply:
        return len(repaired_scores), None, rejected

    reductions = []
    for reduction in log.reductions or []:
        samples = [
            reduced.model_copy(update={
                "value": repaired_scores[str(reduced.sample_id)].value,
                "explanation": repaired_scores[str(reduced.sample_id)].explanation,
                "metadata": repaired_scores[str(reduced.sample_id)].metadata,
            })
            if str(reduced.sample_id) in repaired_scores
            else reduced
            for reduced in reduction.samples
        ]
        reductions.append(reduction.model_copy(update={"samples": samples}))

    recovery = {
        "version": REPAIR_VERSION,
        "repaired_at": datetime.now(timezone.utc).isoformat(),
        "repaired_samples": len(repaired_scores),
        "paid_model_calls_made": 0,
    }
    eval_metadata = {**(log.eval.metadata or {}), "post_validation_recovery": recovery}
    repaired_log = log.model_copy(update={
        "eval": log.eval.model_copy(update={"metadata": eval_metadata}),
        "samples": repaired_samples,
        "reductions": reductions,
    })

    backup = log_path.with_name(log_path.name + f".before-{REPAIR_VERSION}")
    if backup.exists():
        raise FileExistsError(f"refusing to replace existing backup: {backup}")
    temporary = log_path.with_name(log_path.stem + ".repairing.eval")
    shutil.copy2(log_path, backup)
    try:
        write_eval_log(repaired_log, temporary, format="eval")
        verified = read_eval_log(str(temporary))
        passed = sum(
            1
            for sample in verified.samples or []
            if (
                (((sample.scores or {}).get(SCORE_KEY).metadata or {}).get(SCORE_KEY) or {})
                .get("post_validation") == "passed"
            )
        )
        if passed < len(repaired_scores):
            raise RuntimeError(
                f"temporary repaired log retained only {passed} passed scores"
            )
        temporary.replace(log_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return len(repaired_scores), backup, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    count, backup, rejected = repair_log(args.log.resolve(), apply=args.apply)
    if not args.apply:
        print(f"Repair check: {count} saved judgments pass the corrected validator.")
        print("No files were written. Pass --apply to repair the saved scores.")
    elif backup is None:
        print("No matching failed judgments needed repair.")
    else:
        print(f"Repaired {count} saved judgments without model calls.")
        print(f"Original log backup: {backup}")
    if rejected:
        print(f"Still invalid after corrected validation: {len(rejected)}")
        for sample_id, error in rejected.items():
            print(f"- {sample_id}: {error}")


if __name__ == "__main__":
    main()
