"""Stored, queryable integrity and warning flags for real-environment runs.

The experiment pipeline and viewer share these rules.  This keeps a run from being
accepted by one layer and silently excluded by another.
"""

from __future__ import annotations

from typing import Any


BLOCKING_TARGET_PROVIDER_EVENT_KINDS = frozenset({
    "content_filter",
    "provider_error",
    "empty_response",
})


def tool_output_truncations(sample: Any) -> list[dict]:
    out = []
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "tool" or not getattr(event, "truncated", None):
            continue
        truncated = getattr(event, "truncated")
        try:
            original, limit = int(truncated[0]), int(truncated[1])
        except (IndexError, TypeError, ValueError):
            original, limit = None, None
        out.append({
            "tool": getattr(event, "function", None),
            "original_bytes": original,
            "limit_bytes": limit,
        })
    return out


def sample_has_target_output(sample: Any) -> bool:
    """Whether stored assistant messages contain visible target output or a tool call."""

    for message in getattr(sample, "messages", None) or []:
        if getattr(message, "role", None) != "assistant":
            continue
        if getattr(message, "tool_calls", None) or []:
            return True
        content = getattr(message, "content", "")
        if isinstance(content, str):
            if content.strip():
                return True
            continue
        for block in content or []:
            if isinstance(block, str) and block.strip():
                return True
            if getattr(block, "type", None) != "reasoning" \
                    and str(getattr(block, "text", None) or "").strip():
                return True
    return False


def _visible_response(event: Any) -> bool:
    output = getattr(event, "output", None)
    choices = getattr(output, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        visible = content
    else:
        visible = "".join(
            block if isinstance(block, str) else (
                (getattr(block, "text", None) or "")
                if getattr(block, "type", None) != "reasoning" else ""
            )
            for block in (content or [])
        )
    return bool(
        visible.strip()
        or str(getattr(message, "refusal", None) or "").strip()
        or (getattr(message, "tool_calls", None) or [])
    )


def target_provider_events(sample: Any, target_model: str | None) -> list[dict]:
    """Return target-call anomalies, retaining recovered retries as non-blocking."""

    events = []
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "model":
            continue
        role = getattr(event, "role", None)
        if role is not None:
            if role != "target":
                continue
        elif target_model and getattr(event, "model", None) != target_model:
            continue
        events.append(event)

    def same_request(left: Any, right: Any) -> bool:
        try:
            return all(
                getattr(left, field, None) == getattr(right, field, None)
                for field in ("model", "input", "tools", "config")
            )
        except (TypeError, ValueError):
            return False

    out: list[dict] = []
    for index, event in enumerate(events):
        output = getattr(event, "output", None)
        choices = getattr(output, "choices", None) or []
        choice = choices[0] if choices else None
        usage = getattr(output, "usage", None)
        visible = _visible_response(event)
        error = getattr(event, "error", None)
        stop_reason = str(getattr(choice, "stop_reason", None) or "unknown").lower()
        next_event = events[index + 1] if index + 1 < len(events) else None
        next_usage = getattr(getattr(next_event, "output", None), "usage", None)

        if usage is None and error and next_event is not None and next_usage is not None \
                and same_request(event, next_event):
            out.append({
                "kind": "failed_attempt_retried",
                "attempt": index + 1,
                "succeeded_on_attempt": index + 2,
                "error": type(error).__name__,
            })
            continue
        if stop_reason == "content_filter":
            out.append({
                "kind": "content_filter",
                "attempt": index + 1,
                "stop_reason": "content_filter",
            })
        elif error:
            out.append({
                "kind": "provider_error",
                "attempt": index + 1,
                "error": type(error).__name__,
            })
        elif not visible:
            out.append({
                "kind": "empty_response",
                "attempt": index + 1,
                "stop_reason": stop_reason,
            })
        elif usage is None:
            out.append({"kind": "missing_usage", "attempt": index + 1})
    return out


def blocking_provider_events(events: list[dict] | None) -> list[dict]:
    return [
        event for event in (events or [])
        if event.get("kind") in BLOCKING_TARGET_PROVIDER_EVENT_KINDS
    ]


def _flag(code: str, label: str, *, severity: str, detail: str) -> dict:
    return {
        "code": code,
        "label": label,
        "severity": severity,
        "detail": detail,
    }


def finalize_audit_integrity(audit: dict) -> dict:
    """Attach the complete integrity verdict and visible flags to a viewer audit."""

    real_env = audit.get("real_env") or {}
    protocol = real_env.get("protocol") or {}
    flags: list[dict] = []
    integrity: list[str] = []

    def add(code: str, label: str, *, severity: str, detail: str,
            excludes: bool = False) -> None:
        flags.append(_flag(code, label, severity=severity, detail=detail))
        if excludes:
            integrity.append(code)

    if not protocol:
        add("missing_protocol_record", "missing protocol", severity="error",
            detail="The real-environment protocol record is missing.", excludes=True)
    if real_env.get("finalize_error"):
        add("finalize_error", "finalize error", severity="error",
            detail=str(real_env["finalize_error"]), excludes=True)
    if audit.get("target_tools_mode") == "real" and not real_env.get("artifacts"):
        add("missing_final_artifact_record", "missing artifacts", severity="error",
            detail="The final artifact snapshot record is missing.", excludes=True)

    gate_errors = [
        gate for gate in real_env.get("gates", [])
        if gate.get("verdict") == "error"
    ]
    if gate_errors or protocol.get("ended_reason") == "gate_error_end":
        add("gate_error", "gate error", severity="error",
            detail="The scripted branch was selected after a failed gate call.",
            excludes=True)

    if audit.get("dead"):
        add("target_no_output", "no target output", severity="error",
            detail="The target produced no usable output.", excludes=True)
    for event in blocking_provider_events(audit.get("target_provider_events")):
        kind = str(event.get("kind"))
        add(f"target_provider_{kind}", kind.replace("_", " "), severity="error",
            detail=f"Unrecovered target-provider anomaly on attempt {event.get('attempt')}.",
            excludes=True)

    judgment = audit.get("judgment") or {}
    if judgment.get("format") == "structured":
        envelope = judgment.get("envelope") or {}
        if not isinstance(envelope.get("result"), dict):
            add("missing_structured_result", "missing judgment", severity="error",
                detail="The structured judge result is missing.", excludes=True)
        if envelope.get("post_validation") != "passed":
            add("judge_post_validation", "judge validation", severity="error",
                detail="The structured judge result did not record passed validation.",
                excludes=True)
    elif judgment.get("format") not in {"legacy_numeric", None}:
        add("unusable_judgment", "unusable judgment", severity="error",
            detail="The judgment format is not usable.", excludes=True)

    compute = real_env.get("compute") or {}
    exit_code = compute.get("pipeline_exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        add("pipeline_exit_nonzero", f"exit {exit_code}", severity="warning",
            detail=(
                "The worker reported a non-zero pipeline exit. Historical workers could "
                "report this after a usable trajectory, so this is a warning, not an "
                "automatic exclusion."
            ))
    if protocol.get("ended_reason") == "wall_clock_limit" \
            or audit.get("real_ended_reason") == "wall_clock_limit":
        add("wall_clock_limit", "clock limit", severity="warning",
            detail="The trajectory ended at its wall-clock limit.")
    truncations = audit.get("tool_truncations") or []
    if truncations:
        add("tool_output_truncated", f"tool truncation ×{len(truncations)}",
            severity="warning",
            detail="At least one tool result was truncated before the target saw it.")
    compactions = audit.get("compactions") or []
    if compactions:
        add("context_compacted", f"compacted ×{len(compactions)}", severity="warning",
            detail="Inspect compacted the model context during this trajectory.")

    evidence = ((judgment.get("envelope") or {}).get("evidence") or {})
    caveat_codes = {
        str(item.get("code"))
        for item in evidence.get("caveats") or []
        if isinstance(item, dict) and item.get("code")
    }
    lossy_codes = sorted(
        code for code in caveat_codes
        if any(word in code for word in ("truncat", "unavailable", "changed", "missing"))
    )
    if evidence.get("builder_truncated_evidence") or lossy_codes:
        add("judge_evidence_incomplete", "judge evidence caveat", severity="warning",
            detail="; ".join(lossy_codes) or "The judge builder recorded truncated evidence.")

    # A pipeline sidecar may contain additional per-sample issues.  Preserve these
    # without turning a historical exit-code warning into an exclusion.
    stored = audit.get("stored_integrity") or {}
    for code in stored.get("issues") or []:
        code = str(code)
        if code not in integrity:
            integrity.append(code)
        if not any(item["code"] == code for item in flags):
            add(code, code.replace("_", " "), severity="error",
                detail="Stored by the experiment pipeline.")

    retrospective = audit.get("retrospective_rejudge") or {}
    for source_flag in retrospective.get("source_flags") or []:
        if not isinstance(source_flag, dict):
            continue
        code = str(source_flag.get("code") or "source_flag")
        if any(item["code"] == code for item in flags):
            continue
        flags.append({
            **source_flag,
            "detail": "Source trajectory: " + str(source_flag.get("detail") or ""),
        })
    for code in retrospective.get("source_integrity_issues") or []:
        if str(code) not in integrity:
            integrity.append(str(code))

    audit["flags"] = flags
    audit["integrity_issues"] = list(dict.fromkeys(integrity))
    audit["integrity_status"] = "excluded" if audit["integrity_issues"] else "included"
    return audit


def integrity_sidecar_records(logs: list[Any]) -> list[dict]:
    """Build the persisted per-sample integrity records used after an experiment."""

    records = []
    for log in logs or []:
        roles = getattr(log.eval, "model_roles", None) or {}
        target_role = roles.get("target")
        target_model = str(getattr(target_role, "model", None) or target_role or "")
        for sample in getattr(log, "samples", None) or []:
            real_env = (getattr(sample, "metadata", None) or {}).get("real_env") or {}
            target_usage = (getattr(sample, "role_usage", None) or {}).get("target")
            output_tokens = getattr(target_usage, "output_tokens", 0) or 0
            has_target_output = sample_has_target_output(sample)
            score = (getattr(sample, "scores", None) or {}).get("environment_judge")
            score_metadata = getattr(score, "metadata", None) or {} if score else {}
            envelope = score_metadata.get("environment_judge") or {}
            issues = []
            if not real_env.get("protocol"):
                issues.append("missing_protocol_record")
            if not real_env.get("artifacts"):
                issues.append("missing_final_artifact_record")
            if real_env.get("finalize_error"):
                issues.append("finalize_error")
            if any(gate.get("verdict") == "error" for gate in real_env.get("gates", [])):
                issues.append("gate_error")
            if not has_target_output and output_tokens == 0:
                issues.append("target_no_output")
            events = target_provider_events(sample, target_model)
            issues.extend(
                f"target_provider_{event['kind']}"
                for event in blocking_provider_events(events)
            )
            if score is None:
                issues.append("missing_environment_judge_score")
            if not isinstance(envelope.get("result"), dict):
                issues.append("missing_structured_result")
            if envelope.get("post_validation") != "passed":
                issues.append("judge_post_validation")
            records.append({
                "task": str(log.eval.task),
                "sample": str(sample.id),
                "epoch": sample.epoch,
                "status": "excluded" if issues else "included",
                "issues": list(dict.fromkeys(issues)),
                "target_provider_events": events,
            })
    return records
