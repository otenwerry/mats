"""Canonical real-trajectory artifact and loss adapter for every current judge call.

Production final judging and retrospective rejudging both call this module. Historical
runs can lack fields introduced by the standalone pipeline; those absences become stored
judge caveats rather than reconstructed evidence.
"""

from __future__ import annotations

import json
from typing import Sequence

from inspect_ai.model import ChatMessage, ChatMessageTool

from environment_judge.evidence import ArtifactSnapshot, EvidenceCaveat
from real_target_tools import REAL_MAX_TOOL_OUTPUT


def artifact_snapshots(artifacts: dict, snapshot: str) -> list[ArtifactSnapshot]:
    snapshots: list[ArtifactSnapshot] = []
    for entry in artifacts.get("files") or []:
        content = entry.get("content")
        read_error = entry.get("error")
        if content is None and not read_error:
            read_error = "artifact content was unavailable"
        snapshots.append(ArtifactSnapshot(
            path=str(entry.get("path") or "unknown"),
            snapshot=snapshot,
            content=content,
            sha256=entry.get("sha256"),
            byte_count=entry.get("bytes"),
            truncated=bool(entry.get("truncated")),
            read_error=read_error,
        ))
    return snapshots


def tool_output_truncation_numbers(messages: Sequence[ChatMessage]) -> list[int]:
    marker = "The output of your call to "
    return [
        index
        for index, message in enumerate(messages, start=1)
        if isinstance(message, ChatMessageTool) and marker in message.text
    ]


def upstream_caveats(
    messages: Sequence[ChatMessage],
    artifact_records: Sequence[dict],
) -> list[EvidenceCaveat]:
    caveats: list[EvidenceCaveat] = []
    truncated_tools = tool_output_truncation_numbers(messages)
    if truncated_tools:
        caveats.append(EvidenceCaveat(
            code="target_tool_output_truncated",
            description=(
                f"Inspect capped these target-visible tool results at "
                f"{REAL_MAX_TOOL_OUTPUT} bytes before the judge saw them."
            ),
            source="upstream",
            messages=truncated_tools,
            artifacts=[],
        ))
    skipped = sorted({
        path
        for record in artifact_records
        for path in (record.get("artifacts") or {}).get("skipped_non_text") or []
    })
    if skipped:
        caveats.append(EvidenceCaveat(
            code="changed_non_text_artifacts_not_copied",
            description=(
                "Changed non-text artifact contents were not copied into judge evidence: "
                + ", ".join(skipped)
            ),
            source="upstream",
            messages=[],
            artifacts=[],
        ))
    return caveats


def artifact_records(real_env: dict) -> list[dict]:
    submissions = real_env.get("submission_artifacts")
    records = [record for record in submissions if isinstance(record, dict)] \
        if isinstance(submissions, list) else []
    if isinstance(real_env.get("artifacts"), dict):
        records.append({"artifacts": real_env["artifacts"]})
    return records


def artifacts_from_real_env(real_env: dict) -> list[ArtifactSnapshot]:
    snapshots: list[ArtifactSnapshot] = []
    task_context = real_env.get("task_context")
    if isinstance(task_context, dict) and isinstance(task_context.get("content"), str):
        snapshots.append(ArtifactSnapshot.model_validate(task_context))
    submissions = real_env.get("submission_artifacts")
    records = submissions if isinstance(submissions, list) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        snapshots.extend(artifact_snapshots(
            record.get("artifacts") or {},
            f"submission_{record.get('submission')}",
        ))
    if isinstance(real_env.get("artifacts"), dict):
        snapshots.extend(artifact_snapshots(real_env["artifacts"], "final"))

    gates = real_env.get("gates") or []
    if gates and isinstance(gates[0], dict) and gates[0].get("detectors"):
        detector_text = json.dumps(gates[0]["detectors"], indent=2, sort_keys=True)
        snapshots.append(ArtifactSnapshot(
            path="mechanical_evidence.json",
            snapshot="submission_1",
            content=detector_text,
            byte_count=len(detector_text.encode()),
        ))
    grade = real_env.get("grade")
    if isinstance(grade, dict):
        grade_text = json.dumps(grade, indent=2, sort_keys=True)
        snapshots.append(ArtifactSnapshot(
            path="final_grade.json",
            snapshot="final",
            content=grade_text,
            byte_count=len(grade_text.encode()),
        ))

    unique: dict[tuple[str, str], ArtifactSnapshot] = {}
    for snapshot in snapshots:
        unique[(snapshot.path, snapshot.snapshot)] = snapshot
    return list(unique.values())


def complete_real_trajectory_evidence(
    messages: Sequence[ChatMessage],
    real_env: dict,
) -> tuple[list[ArtifactSnapshot], list[EvidenceCaveat]]:
    """Return the current full-trajectory evidence contract and all source loss."""

    records = artifact_records(real_env)
    caveats = upstream_caveats(messages, records)
    task_context = real_env.get("task_context")
    if not (
        isinstance(task_context, dict)
        and isinstance(task_context.get("content"), str)
    ):
        caveats.append(EvidenceCaveat(
            code="initial_task_snapshot_unavailable_upstream",
            description=(
                "This stored trajectory predates the initial-task artifact record; "
                "the judge received the observable messages but no separate initial "
                "README snapshot."
            ),
            source="upstream",
            messages=[],
            artifacts=[],
        ))
    submissions = real_env.get("submission_artifacts")
    if not isinstance(submissions, list):
        caveats.append(EvidenceCaveat(
            code="submission_snapshots_unavailable_upstream",
            description=(
                "This stored trajectory predates per-submission artifact snapshots; "
                "the judge received the final stored artifacts but cannot treat them as "
                "evidence of every earlier submission state."
            ),
            source="upstream",
            messages=[],
            artifacts=[],
        ))
    if not isinstance(real_env.get("artifacts"), dict):
        caveats.append(EvidenceCaveat(
            code="final_artifacts_unavailable_upstream",
            description="No final artifact record was available for this trajectory.",
            source="upstream",
            messages=[],
            artifacts=[],
        ))
    return artifacts_from_real_env(real_env), caveats
