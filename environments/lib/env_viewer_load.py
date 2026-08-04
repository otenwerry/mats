"""Environment-owned Inspect log loader and judgment normalizer.

This module deliberately depends on Inspect only.  It does not import the Petri
package, the old local Petri project, or either project's viewer.  Real-environment
runs keep the target-facing conversation in ``EvalSample.messages``, so we can read
both old and new environments logs without Petri's branch renderer.

There are two judgment formats and they stay separate:

* ``structured`` is the environments-owned schema (``environment_judge``);
* ``legacy_numeric`` is the old Petri-shaped ``audit_judge`` value.

In particular, this loader never turns an old 1/10 score into false/true.  Doing so
would silently change the meaning of historical data.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from real_integrity import finalize_audit_integrity, target_provider_events


CURRENT_SCORE_KEY = "environment_judge"
LEGACY_SCORE_KEY = "audit_judge"
JUDGMENT_KEYS = (CURRENT_SCORE_KEY, LEGACY_SCORE_KEY)
CURRENT_STRUCTURED_DIMENSIONS = (
    "reward_hacking",
    "deception",
    "environment_failure",
)
# Retired fields remain recognizable only so historical structured judgments can be
# rendered exactly. Current schemas and prompts never request them.
RETIRED_STRUCTURED_DIMENSIONS = ("noticed_honeypot",)
KNOWN_STRUCTURED_DIMENSIONS = (
    *RETIRED_STRUCTURED_DIMENSIONS,
    *CURRENT_STRUCTURED_DIMENSIONS,
)

_CURRENT_ML_HACK_TYPES = {
    "used_test_set_to_guide_model_selection",
    "fitted_to_test_set_errors",
    "trained_on_test_set",
    "copied_answer_key_into_solutions",
    "validation_contamination",
    "other_hack",
}
_CURRENT_P_HACK_TYPES = {
    "unsupported_positive_claim",
    "fabricated_positive_claim",
}
_CURRENT_P_REVIEW_TYPES = {"other"}
_LEGACY_ML_HACK_TYPES = {"test_set"}
_LEGACY_P_HACK_TYPES = {
    "selective_reporting",
    "statistical_misrepresentation",
    "fabricated_evidence",
}

_ATTACHMENT_RE = re.compile(r"attachment://([0-9a-fA-F]+)")
_MESSAGE_REF_RE = re.compile(r"\[?M(\d+)\]?", re.IGNORECASE)


def _plain(value: Any) -> Any:
    """Convert Inspect/Pydantic values into JSON-compatible values without dropping fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain(model_dump(mode="json"))
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return {
            str(key): _plain(item)
            for key, item in as_dict.items()
            if not str(key).startswith("_")
        }
    # A repr is preferable to silently omitting an unfamiliar future Inspect field.
    return {"unparsed_python_value": repr(value), "python_type": type(value).__name__}


def resolve_attachments(value: Any, attachments: dict[str, str] | None) -> Any:
    """Resolve Inspect's attachment placeholders recursively.

    An unknown attachment remains visible as its original ``attachment://`` value.
    """
    attachments = attachments or {}
    if isinstance(value, str):
        return _ATTACHMENT_RE.sub(
            lambda match: attachments.get(match.group(1), match.group(0)), value
        )
    if isinstance(value, dict):
        return {
            str(key): resolve_attachments(item, attachments)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_attachments(item, attachments) for item in value]
    return value


def _message_content(message: Any, attachments: dict[str, str]) -> tuple[str, str, list]:
    """Return visible text, native reasoning, and unfamiliar content blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return str(resolve_attachments(content, attachments)), "", []

    visible: list[str] = []
    reasoning: list[str] = []
    other: list[Any] = []
    for block in content or []:
        if isinstance(block, str):
            visible.append(str(resolve_attachments(block, attachments)))
            continue
        kind = str(getattr(block, "type", "") or "")
        if kind == "reasoning":
            value = (
                getattr(block, "reasoning", None)
                or getattr(block, "summary", None)
                or getattr(block, "text", None)
            )
            if value:
                reasoning.append(str(resolve_attachments(value, attachments)))
            elif getattr(block, "redacted", False):
                reasoning.append("[reasoning produced but redacted by the provider]")
            else:
                other.append(resolve_attachments(_plain(block), attachments))
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            visible.append(str(resolve_attachments(text, attachments)))
        else:
            other.append(resolve_attachments(_plain(block), attachments))
    return "".join(visible), "\n\n".join(reasoning), other


def normalize_messages(
    messages: Iterable[Any], attachments: dict[str, str] | None = None
) -> list[dict]:
    """Normalize target-facing Inspect messages and give each a stable ``M#`` anchor."""
    attachments = attachments or {}
    normalized: list[dict] = []
    assistant_index = 0
    user_index = 0
    for number, message in enumerate(messages or [], start=1):
        role = str(getattr(message, "role", "other") or "other")
        text, reasoning, other_blocks = _message_content(message, attachments)
        if role == "assistant":
            assistant_index += 1
        if role == "user":
            user_index += 1

        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            calls.append({
                "id": getattr(call, "id", None),
                "function": getattr(call, "function", None),
                "arguments": resolve_attachments(
                    _plain(getattr(call, "arguments", None)), attachments
                ),
                "type": getattr(call, "type", None),
            })

        error = getattr(message, "error", None)
        normalized.append({
            "number": number,
            "id": f"M{number}",
            "source_id": getattr(message, "id", None),
            "role": role,
            "text": text,
            "reasoning": reasoning,
            "other_content_blocks": other_blocks,
            "tool_calls": calls,
            "tool_call_id": getattr(message, "tool_call_id", None),
            "function": getattr(message, "function", None),
            "error": resolve_attachments(_plain(error), attachments) if error else None,
            "source": getattr(message, "source", None),
            "assistant_turn": assistant_index if role == "assistant" else None,
            "user_turn": user_index if role == "user" else None,
        })
    return normalized


def transcript_text(messages: list[dict]) -> str:
    """Compatibility rendering for code that still expects a numbered text transcript."""
    chunks = []
    for message in messages:
        role = str(message.get("role") or "other").capitalize()
        body = message.get("text") or ""
        if message.get("reasoning"):
            body += f"\n<thinking>{message['reasoning']}</thinking>"
        for call in message.get("tool_calls") or []:
            body += (
                f"\nTool Call: {call.get('function') or ''}\nArguments:\n"
                f"{json.dumps(call.get('arguments'), ensure_ascii=False, indent=2)}"
            )
        if message.get("other_content_blocks"):
            body += (
                "\nUnrecognized content blocks:\n"
                + json.dumps(
                    message["other_content_blocks"], ensure_ascii=False, indent=2
                )
            )
        chunks.append(f"[{message['id']}] {role}:\n{body}".rstrip())
    return "\n\n".join(chunks)


def _message_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        match = _MESSAGE_REF_RE.fullmatch(value.strip())
        return int(match.group(1)) if match else None
    if isinstance(value, dict):
        for key in ("number", "message", "message_number", "prompt_message"):
            if key in value:
                return _message_number(value[key])
    return None


def _mapped_message_number(
    number: int, source_message_map: Any, source_lookup: dict | None = None
) -> int:
    """Map a prompt-local number back to Sample.messages when a judge stored a map."""
    if isinstance(source_message_map, dict):
        entry = source_message_map.get(str(number), source_message_map.get(number))
        if entry is not None:
            if isinstance(entry, dict):
                for key in (
                    "source_message", "source_number", "sample_message", "original"
                ):
                    mapped = _message_number(entry.get(key))
                    if mapped is not None:
                        return mapped
            mapped = _message_number(entry)
            if mapped is not None:
                return mapped
    if isinstance(source_message_map, list):
        for entry in source_message_map:
            if not isinstance(entry, dict):
                continue
            prompt = next(
                (
                    _message_number(entry.get(key))
                    for key in ("prompt_message", "judge_message", "number", "message")
                    if entry.get(key) is not None
                ),
                None,
            )
            if prompt != number:
                continue
            source_lookup = source_lookup or {}
            source_id = entry.get("source_id")
            if source_id is not None and str(source_id) in source_lookup.get("ids", {}):
                return source_lookup["ids"][str(source_id)]
            source_index = entry.get("source_index")
            if (
                isinstance(source_index, int)
                and not isinstance(source_index, bool)
                and source_index in source_lookup.get("indexes", {})
            ):
                return source_lookup["indexes"][source_index]
            mapped = next(
                (
                    _message_number(entry.get(key))
                    for key in (
                        "source_message", "source_number", "sample_message", "original"
                    )
                    if entry.get(key) is not None
                ),
                None,
            )
            if mapped is not None:
                return mapped
    return number


def normalize_evidence(
    evidence: Any,
    *,
    present_messages: set[int] | None = None,
    source_message_map: Any = None,
    source_lookup: dict | None = None,
) -> dict:
    """Normalize structured message and artifact citations; retain malformed citations."""
    if not isinstance(evidence, dict):
        evidence = {} if evidence is None else {"unparsed": _plain(evidence)}
    raw_messages = (
        evidence.get("messages")
        or evidence.get("message_refs")
        or evidence.get("message_ids")
        or []
    )
    if not isinstance(raw_messages, list):
        raw_messages = [raw_messages]

    messages: list[dict] = []
    issues: list[dict] = []
    seen_messages: set[tuple[int, int]] = set()
    for raw in raw_messages:
        prompt_number = _message_number(raw)
        if prompt_number is None:
            issues.append({"kind": "invalid_message_reference", "reference": _plain(raw)})
            continue
        source_number = _mapped_message_number(
            prompt_number, source_message_map, source_lookup
        )
        pair = (prompt_number, source_number)
        if pair in seen_messages:
            continue
        seen_messages.add(pair)
        available = None if present_messages is None else source_number in present_messages
        messages.append({
            "prompt_number": prompt_number,
            "number": source_number,
            "id": f"M{source_number}",
            "available": available,
        })
        if available is False:
            issues.append({
                "kind": "missing_message_reference",
                "prompt_number": prompt_number,
                "source_number": source_number,
            })

    raw_artifacts = evidence.get("artifacts") or evidence.get("files") or []
    if not isinstance(raw_artifacts, list):
        raw_artifacts = [raw_artifacts]
    artifacts = []
    seen_artifacts: set[tuple[str, str]] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            issues.append({"kind": "invalid_artifact_reference", "reference": _plain(raw)})
            continue
        path = raw.get("path")
        snapshot = raw.get("snapshot") or raw.get("submission")
        if not isinstance(path, str) or not path or not isinstance(snapshot, (str, int)):
            issues.append({"kind": "invalid_artifact_reference", "reference": _plain(raw)})
            continue
        key = (path, str(snapshot))
        if key in seen_artifacts:
            continue
        seen_artifacts.add(key)
        artifacts.append({"path": path, "snapshot": str(snapshot)})

    return {
        "messages": messages,
        "artifacts": artifacts,
        "issues": issues,
        "raw": deepcopy(evidence),
    }


def _structured_dimension(
    key: str,
    raw: Any,
    *,
    family: str | None,
    present_messages: set[int] | None,
    source_message_map: Any,
    source_lookup: dict | None,
) -> dict:
    missing = raw is ...
    if missing:
        return {
            "key": key,
            "status": "missing",
            "value": None,
            "display_value": "missing",
            "explanation": "",
            "evidence": normalize_evidence({}, present_messages=present_messages),
            "raw": None,
        }

    if key == "deception" and raw is None:
        return {
            "key": key,
            "status": "not_applicable",
            "value": None,
            "display_value": "n/a",
            "explanation": "",
            "evidence": normalize_evidence({}, present_messages=present_messages),
            "raw": None,
        }

    raw_dict = raw if isinstance(raw, dict) else {"value": raw}
    explanation = str(
        raw_dict.get("explanation") or raw_dict.get("reason") or ""
    )
    evidence = normalize_evidence(
        raw_dict.get("evidence"),
        present_messages=present_messages,
        source_message_map=source_message_map,
        source_lookup=source_lookup,
    )

    if key in {"noticed_honeypot", "environment_failure"}:
        value = raw_dict.get("value")
        valid = isinstance(value, bool)
        display = "yes" if value is True else "no" if value is False else "invalid"
    elif key == "reward_hacking":
        value = raw_dict.get("types")
        not_delivered = raw_dict.get("not_delivered", [])
        valid = (
            isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            and isinstance(not_delivered, list)
            and all(isinstance(item, str) for item in not_delivered)
        )
        if valid:
            display = ", ".join(
                f"{item} (not delivered)" if item in not_delivered else item
                for item in value
            ) or "none"
            if family == "ml_training_data_misuse":
                hack_vocabulary = _CURRENT_ML_HACK_TYPES | _LEGACY_ML_HACK_TYPES
            elif family == "p_hacking":
                hack_vocabulary = _CURRENT_P_HACK_TYPES | _LEGACY_P_HACK_TYPES
            else:
                hack_vocabulary = (
                    _CURRENT_ML_HACK_TYPES | _LEGACY_ML_HACK_TYPES
                    | _CURRENT_P_HACK_TYPES | _LEGACY_P_HACK_TYPES
                )
            hack_types = [item for item in value if item in hack_vocabulary]
            review_types = (
                [item for item in value if item in _CURRENT_P_REVIEW_TYPES]
                if family == "p_hacking"
                else []
            )
        else:
            display = "invalid"
            hack_types = []
            review_types = []
    else:
        value = raw_dict.get("score", raw_dict.get("value"))
        valid = isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10
        display = str(value) if valid else "invalid"

    normalized = {
        "key": key,
        "status": "ok" if valid else "invalid",
        "value": deepcopy(value),
        "display_value": display,
        "explanation": explanation,
        "evidence": evidence,
        "raw": deepcopy(raw),
    }
    if key == "reward_hacking":
        normalized.update({
            "hack_types": hack_types,
            "review_types": review_types,
            "requires_review": bool(review_types) if valid else None,
            "is_hack": (
                None if valid and review_types
                else bool(hack_types) if valid
                else None
            ),
        })
        if family == "ml_training_data_misuse" or "not_delivered" in raw_dict:
            normalized["not_delivered"] = (
                deepcopy(not_delivered) if valid else None
            )
    return normalized


def normalize_evidence_scope(
    envelope: dict | None,
    *,
    present_messages: set[int] | None = None,
    source_message_map: Any = None,
    source_lookup: dict | None = None,
) -> dict:
    """Normalize judge-builder coverage metadata without changing its raw record."""
    envelope = envelope or {}
    evidence = envelope.get("evidence") or {}
    if not isinstance(evidence, dict):
        return {
            "native_reasoning_policy": None,
            "native_reasoning_message_count": None,
            "native_reasoning_block_count": None,
            "source_message_count": None,
            "selected_message_count": None,
            "omitted_message_count": None,
            "message_selection": None,
            "builder_truncated_evidence": None,
            "caveats": [],
            "artifacts": [],
            "issues": [{"kind": "invalid_evidence_scope"}],
            "raw": deepcopy(evidence),
        }
    caveats = []
    issues = []
    for raw in evidence.get("caveats") or []:
        if not isinstance(raw, dict):
            issues.append({"kind": "invalid_evidence_caveat", "caveat": _plain(raw)})
            continue
        normalized = normalize_evidence(
            {"messages": raw.get("messages") or [],
             "artifacts": raw.get("artifacts") or []},
            present_messages=present_messages,
            source_message_map=source_message_map,
            source_lookup=source_lookup,
        )
        caveats.append({
            "code": str(raw.get("code") or "unknown_caveat"),
            "description": str(raw.get("description") or ""),
            "source": str(raw.get("source") or "stored"),
            "messages": normalized["messages"],
            "artifacts": normalized["artifacts"],
            "issues": normalized["issues"],
            "raw": deepcopy(raw),
        })
        issues.extend({**item, "caveat": raw.get("code")} for item in normalized["issues"])
    return {
        "native_reasoning_policy": evidence.get("native_reasoning_policy"),
        "native_reasoning_message_count": evidence.get(
            "native_reasoning_message_count"
        ),
        "native_reasoning_block_count": evidence.get("native_reasoning_block_count"),
        "source_message_count": evidence.get("source_message_count"),
        "selected_message_count": evidence.get("selected_message_count"),
        "omitted_message_count": evidence.get("omitted_message_count"),
        "message_selection": evidence.get("message_selection"),
        "builder_truncated_evidence": evidence.get("builder_truncated_evidence"),
        "caveats": caveats,
        "artifacts": deepcopy(evidence.get("artifacts") or []),
        "issues": issues,
        "raw": deepcopy(evidence),
    }


def normalize_structured_judgment(
    raw_result: dict,
    *,
    family: str | None = None,
    schema_version: str | None = None,
    present_messages: set[int] | None = None,
    source_message_map: Any = None,
    source_lookup: dict | None = None,
    envelope: dict | None = None,
) -> dict:
    """Normalize an environments-owned structured result for viewer code."""
    dimension_keys = [
        *(
            key for key in RETIRED_STRUCTURED_DIMENSIONS
            if key in raw_result
        ),
        *CURRENT_STRUCTURED_DIMENSIONS,
    ]
    dimensions = [
        _structured_dimension(
            key,
            raw_result.get(key, ...),
            family=family,
            present_messages=present_messages,
            source_message_map=source_message_map,
            source_lookup=source_lookup,
        )
        for key in dimension_keys
        if key != "deception" or family != "p_hacking" or key in raw_result
    ]
    issues = []
    for dimension in dimensions:
        if dimension["status"] in {"missing", "invalid"}:
            issues.append({
                "kind": f"dimension_{dimension['status']}",
                "dimension": dimension["key"],
            })
        issues.extend({**issue, "dimension": dimension["key"]}
                      for issue in dimension["evidence"]["issues"])
    evidence_scope = normalize_evidence_scope(
        envelope,
        present_messages=present_messages,
        source_message_map=source_message_map,
        source_lookup=source_lookup,
    )
    issues.extend(evidence_scope["issues"])
    return {
        "format": "structured",
        "schema_version": schema_version or "environment_judge.unknown",
        "family": family,
        "summary": str(raw_result.get("summary") or ""),
        "justification": str((envelope or {}).get("justification") or ""),
        "highlights": str(raw_result.get("highlights") or ""),
        "dimensions": dimensions,
        "issues": issues,
        "evidence_scope": evidence_scope,
        "raw_result": deepcopy(raw_result),
        "envelope": deepcopy(envelope or {}),
    }


def normalize_legacy_judgment(
    scores: dict,
    *,
    metadata: dict | None = None,
    explanation: str = "",
) -> dict:
    """Preserve old numeric scores exactly, with no new-schema reinterpretation."""
    dimensions = []
    for key, value in scores.items():
        dimensions.append({
            "key": str(key),
            "status": "ok" if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "invalid",
            "value": deepcopy(value),
            "display_value": f"{value}/10" if isinstance(value, (int, float)) else str(value),
            "explanation": "",
            "evidence": {"messages": [], "artifacts": [], "issues": [], "raw": {}},
            "raw": deepcopy(value),
        })
    return {
        "format": "legacy_numeric",
        "schema_version": "petri_numeric_legacy",
        "family": (metadata or {}).get("dimension_scope"),
        "dimensions": dimensions,
        "issues": [],
        "raw_result": deepcopy(scores),
        "legacy": {
            "summary": str((metadata or {}).get("summary") or ""),
            "highlights": str((metadata or {}).get("highlights") or ""),
            "justification": str(explanation or ""),
        },
    }


def normalize_judgment(
    *,
    score_key: str | None,
    score_value: Any,
    score_metadata: dict | None,
    real_env: dict | None,
    present_messages: set[int] | None = None,
    source_lookup: dict | None = None,
    score_explanation: str = "",
) -> dict | None:
    """Select canonical stored judgment data, with tolerant forward compatibility."""
    score_metadata = _plain(score_metadata or {})
    real_env = _plain(real_env or {})
    # The score owns the complete call record (exact prompt, evidence, schema, model
    # interface). ``real_env.final_judgment`` is a deliberately smaller trajectory
    # summary. Merge it on top instead of letting it hide the richer score record.
    score_envelope = score_metadata.get("environment_judge")
    if not isinstance(score_envelope, dict):
        score_envelope = score_metadata.get("final_judgment")
    real_envelope = real_env.get("final_judgment")
    if (
        isinstance(score_envelope, dict)
        and score_envelope.get("judgment_role") == "retrospective_rejudge"
    ):
        # A rejudge deliberately carries the original real_env record. Its old
        # final_judgment must not overwrite the new score that this row represents.
        envelope = {
            **(real_envelope if isinstance(real_envelope, dict) else {}),
            **score_envelope,
        }
    else:
        envelope = {
            **(score_envelope if isinstance(score_envelope, dict) else {}),
            **(real_envelope if isinstance(real_envelope, dict) else {}),
        }

    raw_result = envelope.get("result")
    if not isinstance(raw_result, dict) and isinstance(score_metadata.get("result"), dict):
        raw_result = score_metadata["result"]
    plain_score = _plain(score_value)
    looks_structured = (
        isinstance(plain_score, dict)
        and any(
            key in plain_score and (
                isinstance(plain_score[key], dict)
                or (key == "deception" and plain_score[key] is None)
            )
            for key in KNOWN_STRUCTURED_DIMENSIONS
        )
    )
    if not isinstance(raw_result, dict) and (
        score_key == CURRENT_SCORE_KEY or looks_structured
    ):
        raw_result = plain_score if isinstance(plain_score, dict) else None
    if isinstance(raw_result, dict):
        provenance = envelope.get("provenance") or score_metadata.get("provenance") or {}
        evidence_metadata = envelope.get("evidence") or {}
        source_map = (
            envelope.get("source_message_map")
            or (evidence_metadata.get("source_message_map")
                if isinstance(evidence_metadata, dict) else None)
            or score_metadata.get("source_message_map")
            or (provenance.get("source_message_map") if isinstance(provenance, dict) else None)
        )
        judgment = normalize_structured_judgment(
            raw_result,
            family=(
                envelope.get("family")
                or score_metadata.get("family")
                or score_metadata.get("judge_family")
            ),
            schema_version=(
                envelope.get("schema_version")
                or envelope.get("judge_schema_version")
                or score_metadata.get("judge_schema_version")
                or score_metadata.get("schema_version")
            ),
            present_messages=present_messages,
            source_message_map=source_map,
            source_lookup=source_lookup,
            envelope=envelope,
        )
        judgment["overall_explanation"] = (
            "" if judgment.get("justification")
            else str(score_explanation or "")
        )
        return judgment

    if score_key == LEGACY_SCORE_KEY and isinstance(score_value, dict):
        return normalize_legacy_judgment(
            _plain(score_value),
            metadata=score_metadata,
            explanation=str(score_metadata.get("explanation") or ""),
        )
    return None


def _usage_dict(usage: Any) -> dict:
    out = {}
    for key, item in (usage or {}).items():
        out[str(key)] = {
            "input": getattr(item, "input_tokens", 0) or 0,
            "output": getattr(item, "output_tokens", 0) or 0,
            "cache_read": getattr(item, "input_tokens_cache_read", 0) or 0,
            "cache_write": getattr(item, "input_tokens_cache_write", 0) or 0,
            "total_cost": getattr(item, "total_cost", None),
        }
    return out


def _role_model(roles: dict, role: str) -> str | None:
    model = (roles or {}).get(role)
    if model is None:
        return None
    return str(getattr(model, "model", None) or model)


def _tool_truncations(sample: Any) -> list[dict]:
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


def _score_for_sample(sample: Any) -> tuple[str | None, Any]:
    scores = getattr(sample, "scores", None) or {}
    for key in JUDGMENT_KEYS:
        if key in scores:
            return key, scores[key]
    return None, None


def sample_to_audit(*, mode: str, mode_mtime: float, task: str,
                    run_metadata: dict, roles: dict, sample: Any) -> dict:
    """Convert one Inspect sample into the plain viewer contract."""
    attachments = getattr(sample, "attachments", None) or {}
    messages = normalize_messages(getattr(sample, "messages", None) or [], attachments)
    present_messages = {message["number"] for message in messages}
    source_lookup = {
        "ids": {
            str(message["source_id"]): message["number"]
            for message in messages
            if message.get("source_id") is not None
        },
        "indexes": {
            index: message["number"] for index, message in enumerate(messages)
        },
    }
    score_key, score = _score_for_sample(sample)
    score_value = getattr(score, "value", None) if score is not None else None
    score_metadata = (
        _plain(getattr(score, "metadata", None) or {}) if score is not None else {}
    )
    score_explanation = (
        str(getattr(score, "explanation", None) or "") if score is not None else ""
    )
    if score_key == LEGACY_SCORE_KEY:
        # Old summaries/highlights live in score metadata; keep the score explanation too.
        score_metadata = {**score_metadata, "explanation": score_explanation}

    sample_metadata = _plain(getattr(sample, "metadata", None) or {})
    real_env = sample_metadata.get("real_env") or {}
    retrospective = sample_metadata.get("retrospective_rejudge")
    if not isinstance(retrospective, dict):
        retrospective = None
    judgment = normalize_judgment(
        score_key=score_key,
        score_value=score_value,
        score_metadata=score_metadata,
        real_env=real_env,
        present_messages=present_messages,
        source_lookup=source_lookup,
        score_explanation=score_explanation,
    )
    protocol = (real_env.get("protocol") or {}) if isinstance(real_env, dict) else {}
    ended_reason = protocol.get("ended_reason")
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    has_target_output = any(
        message["text"].strip() or message["tool_calls"] for message in assistant_messages
    )
    target_usage = (getattr(sample, "role_usage", None) or {}).get("target")
    target_output_tokens = (
        getattr(target_usage, "output_tokens", 0) or 0
        if target_usage is not None else 0
    )
    compactions = [
        _plain(event)
        for event in getattr(sample, "events", None) or []
        if getattr(event, "event", None) == "compaction"
    ]

    scores = _plain(score_value) if score_key == LEGACY_SCORE_KEY and isinstance(score_value, dict) else {}
    target_model = (
        retrospective.get("target")
        if retrospective is not None else _role_model(roles, "target")
    )
    audit = {
        "mode": mode,
        "mtime": mode_mtime,
        "task": task,
        "seed": str(
            retrospective.get("seed")
            if retrospective is not None else getattr(sample, "id", "")
        ),
        "epoch": (
            retrospective.get("epoch")
            if retrospective is not None else getattr(sample, "epoch", None)
        ),
        "target": target_model,
        "judge": _role_model(roles, "judge"),
        "condition": (
            retrospective.get("condition")
            if retrospective is not None else run_metadata.get("condition")
        ),
        "judgment_role": (
            "retrospective_rejudge" if retrospective is not None else "official"
        ),
        "retrospective_rejudge": retrospective,
        "dimension_scope": run_metadata.get("dimension_scope"),
        "target_tools_mode": run_metadata.get("target_tools_mode"),
        "judge_dimensions": list(run_metadata.get("judge_dimensions") or []),
        "judge_dimension_files": list(run_metadata.get("judge_dimension_files") or []),
        "score_key": score_key,
        "score_value": _plain(score_value),
        "score_metadata": score_metadata,
        "scores": scores,
        "summary": str(
            (judgment or {}).get("summary")
            or score_metadata.get("summary")
            or ""
        ),
        "highlights": str(
            (judgment or {}).get("highlights")
            or score_metadata.get("highlights")
            or ""
        ),
        "justification": str(
            (judgment or {}).get("justification") or score_explanation or ""
        ),
        "judgment": judgment,
        "messages": messages,
        "transcript": transcript_text(messages),
        "transcript_source": "sample_messages",
        "real_env": real_env,
        "real_ended_reason": ended_reason,
        "dead": not has_target_output and target_output_tokens == 0,
        "crashed": run_metadata.get("target_tools_mode") == "real" and ended_reason is None,
        "tool_truncations": _tool_truncations(sample),
        "target_provider_events": target_provider_events(sample, target_model),
        "compactions": compactions,
        "model_usage": _usage_dict(getattr(sample, "model_usage", None)),
        "role_usage": _usage_dict(getattr(sample, "role_usage", None)),
        "load_issues": list((judgment or {}).get("issues") or []),
    }
    return finalize_audit_integrity(audit)


def traj_key(audit: dict) -> str:
    """Stable identity used by the existing environments data files."""
    return (
        f"{audit['mode']}__{audit['task']}__{audit['seed']}__e{audit['epoch']}"
    )


def link_rejudge_sources(audits: list[dict]) -> None:
    """Attach viewer IDs for the original trajectory referenced by each rejudge."""

    originals = {
        (
            audit.get("mode"), audit.get("task"), str(audit.get("seed")),
            audit.get("epoch"),
        ): audit.get("id")
        for audit in audits
        if not audit.get("retrospective_rejudge")
    }
    for audit in audits:
        source = audit.get("retrospective_rejudge")
        if not isinstance(source, dict):
            continue
        key = (
            source.get("source_run"), source.get("source_task"),
            str(source.get("seed")), source.get("epoch"),
        )
        audit["source_trajectory_id"] = originals.get(key)


def _module_signature() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _mode_signature(mode_dir: Path) -> str:
    rows = []
    paths = list(mode_dir.glob("*.eval"))
    remote_campaign = mode_dir / "remote_campaign.json"
    if remote_campaign.is_file():
        paths.append(remote_campaign)
    integrity_sidecar = mode_dir / "pipeline_integrity.json"
    if integrity_sidecar.is_file():
        paths.append(integrity_sidecar)
    for path in sorted(paths):
        stat = path.stat()
        rows.append((path.name, stat.st_mtime_ns, stat.st_size))
    payload = json.dumps([_module_signature(), rows], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_file(mode_dir: Path, cache_root: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", mode_dir.name)
    return cache_root / f"mode__{safe}__{_mode_signature(mode_dir)}.pkl"


def _write_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def attach_remote_compute(mode_dir: Path, audits: list[dict]) -> list[dict]:
    """Attach final VM-cost records from a verified imported AWS campaign."""

    path = mode_dir / "remote_campaign.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{
            "kind": "invalid_remote_campaign_sidecar",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }]
    task_compute = payload.get("task_compute")
    if not isinstance(task_compute, dict):
        return [{
            "kind": "invalid_remote_campaign_sidecar",
            "path": str(path),
            "error": "task_compute is missing or is not an object",
        }]
    terminal_by_task = {}
    for cell in payload.get("cells") or []:
        terminal = cell.get("terminal") or {}
        task_name = terminal.get("task_name")
        if not task_name:
            continue
        terminal_by_task[str(task_name)] = {
            "campaign_cell_status": cell.get("status"),
            "pipeline_exit_code": terminal.get("pipeline_exit_code"),
            "worker_started_at": terminal.get("started_at"),
            "worker_completed_at": terminal.get("completed_at"),
        }
    for audit in audits:
        compute = task_compute.get(str(audit.get("task") or ""))
        if not isinstance(compute, dict):
            continue
        real_env = audit.get("real_env")
        if not isinstance(real_env, dict):
            real_env = {}
            audit["real_env"] = real_env
        # The sidecar is written after termination, so it supersedes the pre-launch
        # execution identity embedded inside the EvalLog.
        real_env["compute"] = {
            **deepcopy(compute),
            **deepcopy(terminal_by_task.get(str(audit.get("task") or ""), {})),
        }
    return []


def attach_integrity_sidecar(mode_dir: Path, audits: list[dict]) -> list[dict]:
    """Attach the experiment pipeline's persisted per-sample integrity verdict."""

    path = mode_dir / "pipeline_integrity.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{
            "kind": "invalid_pipeline_integrity_sidecar",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }]
    records = payload.get("records")
    if not isinstance(records, list):
        return [{
            "kind": "invalid_pipeline_integrity_sidecar",
            "path": str(path),
            "error": "records is missing or is not an array",
        }]
    by_key = {
        (str(record.get("task")), str(record.get("sample")), record.get("epoch")): record
        for record in records if isinstance(record, dict)
    }
    for audit in audits:
        key = (str(audit.get("task")), str(audit.get("seed")), audit.get("epoch"))
        record = by_key.get(key)
        if record is not None:
            audit["stored_integrity"] = deepcopy(record)
    return []


async def load_mode(
    mode_dir: Path, *, cache_root: Path | None = None, use_cache: bool = True
) -> list[dict]:
    """Load one real-environment run directory through Inspect's public log API."""
    cache_path = _cache_file(mode_dir, cache_root) if cache_root and use_cache else None
    if cache_path and cache_path.exists():
        try:
            return pickle.loads(cache_path.read_bytes())
        except (OSError, EOFError, pickle.UnpicklingError):
            pass

    # Lazy import keeps pure normalizer/component tests runnable without the runtime.
    from inspect_ai.log import list_eval_logs, read_eval_log

    eval_paths = list(mode_dir.glob("*.eval"))
    mode_mtime = max((path.stat().st_mtime for path in eval_paths), default=0.0)
    audits = []
    for log_info in list_eval_logs(str(mode_dir)):
        try:
            log = read_eval_log(log_info)
        except FileNotFoundError:
            continue
        roles = getattr(log.eval, "model_roles", None) or {}
        run_metadata = _plain(getattr(log.eval, "metadata", None) or {})
        for sample in getattr(log, "samples", None) or []:
            audits.append(sample_to_audit(
                mode=mode_dir.name,
                mode_mtime=mode_mtime,
                task=str(log.eval.task),
                run_metadata=run_metadata,
                roles=roles,
                sample=sample,
            ))
    sidecar_issues = attach_remote_compute(mode_dir, audits)
    sidecar_issues.extend(attach_integrity_sidecar(mode_dir, audits))
    for audit in audits:
        finalize_audit_integrity(audit)
    if sidecar_issues:
        for audit in audits:
            audit.setdefault("load_issues", []).extend(deepcopy(sidecar_issues))
    if cache_path:
        _write_pickle_atomic(cache_path, audits)
        for stale in cache_path.parent.glob(f"mode__{mode_dir.name}__*.pkl"):
            if stale != cache_path:
                stale.unlink(missing_ok=True)
    return audits


async def load_all(
    logs_root: Path, *, cache_root: Path | None = None, use_cache: bool = True
) -> tuple[list[dict], list[dict]]:
    """Load every run directory, returning visible per-directory errors separately."""
    if not logs_root.is_dir():
        return [], []
    audits: list[dict] = []
    errors: list[dict] = []
    for mode_dir in sorted(path for path in logs_root.iterdir() if path.is_dir()):
        try:
            audits.extend(await load_mode(
                mode_dir, cache_root=cache_root, use_cache=use_cache
            ))
        except Exception as error:  # one malformed run must not hide every healthy run
            errors.append({
                "mode": mode_dir.name,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return audits, errors


@contextmanager
def viewer_build_lock(data_root: Path) -> Iterator[None]:
    """Serialize viewer builds and release the lock automatically after crashes."""
    path = data_root / ".viewer_build.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def assign_stable_ids(audits: list[dict], registry_file: Path) -> None:
    """Assign append-only trajectory IDs using the existing registry format."""
    try:
        registry = json.loads(registry_file.read_text()) if registry_file.exists() else {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read trajectory ID registry {registry_file}: {error}") from error
    if not isinstance(registry, dict) or not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in registry.items()
    ):
        raise ValueError(f"invalid trajectory ID registry: {registry_file}")

    next_id = max(registry.values(), default=0) + 1
    new = [audit for audit in audits if traj_key(audit) not in registry]
    for audit in sorted(
        new,
        key=lambda item: (
            item.get("mtime", 0), item.get("task", ""),
            item.get("seed", ""), item.get("epoch", 0),
        ),
    ):
        registry[traj_key(audit)] = next_id
        next_id += 1
    if new:
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(registry, indent=2, sort_keys=True) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=registry_file.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, registry_file)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    for audit in audits:
        audit["id"] = registry[traj_key(audit)]
