"""Recover observable OpenCode history when its native call is interrupted.

Inspect SWE's OpenCode adapter returns an ``AgentState`` only after the native CLI
process exits.  If the outer sample clock cancels that await, Inspect still has the
individual target ``ModelEvent`` records, but ``TaskState.messages`` remains frozen at
the user message that started the call.  This module conservatively joins the newest
recorded native history back onto that exact user boundary.

Recovery is deliberately partial and fail-closed.  It never invents a tool result or
claims to cover activity after the last recorded target event.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


RECOVERY_FORMAT = "interrupted-opencode-transcript-v1"
_ATTACHMENT_RE = re.compile(r"attachment://([0-9a-fA-F]+)")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _resolve_attachments(text: str, attachments: dict[str, str] | None) -> str:
    attachments = attachments or {}
    return _ATTACHMENT_RE.sub(
        lambda match: attachments.get(match.group(1), match.group(0)), text
    )


def _message_text(
    message: Any, attachments: dict[str, str] | None = None
) -> str:
    text = _get(message, "text")
    if isinstance(text, str):
        return _resolve_attachments(text, attachments)
    content = _get(message, "content", "")
    if isinstance(content, str):
        return _resolve_attachments(content, attachments)
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(_resolve_attachments(block, attachments))
            continue
        for field in ("text", "reasoning", "summary"):
            value = _get(block, field)
            if isinstance(value, str) and value:
                parts.append(_resolve_attachments(value, attachments))
                break
    return "".join(parts)


def _native_user_text(
    message: Any, attachments: dict[str, str] | None = None
) -> tuple[str, str]:
    """Remove only OpenCode's known literal outer-double-quote wrapper."""

    text = _message_text(message, attachments)
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1], "outer_double_quotes"
    return text, "none"


def _observable_message(message: Any) -> bool:
    """Whether a target output contains text, reasoning, refusal, or a tool call."""

    if message is None:
        return False
    if _get(message, "tool_calls", []) or []:
        return True
    if str(_get(message, "refusal", "") or "").strip():
        return True
    content = _get(message, "content", "")
    if isinstance(content, str):
        return bool(content.strip())
    for block in content or []:
        if isinstance(block, str) and block.strip():
            return True
        for field in ("text", "reasoning", "summary"):
            if str(_get(block, field, "") or "").strip():
                return True
    return False


def _event_output_message(event: Any) -> Any | None:
    output = _get(event, "output")
    message = _get(output, "message")
    if message is not None:
        return message
    choices = _get(output, "choices", []) or []
    return _get(choices[0], "message") if choices else None


def _target_events(
    events: Sequence[Any], target_model: str | None
) -> list[tuple[int, Any]]:
    selected: list[tuple[int, Any]] = []
    for index, event in enumerate(events):
        if _get(event, "event") != "model":
            continue
        role = _get(event, "role")
        if role is not None:
            if role != "target":
                continue
        elif target_model and _get(event, "model") != target_model:
            continue
        selected.append((index, event))
    return selected


def recover_interrupted_opencode_messages(
    base_messages: Sequence[Any],
    events: Sequence[Any],
    *,
    target_model: str | None = None,
    event_start: int = 0,
    applied_before_judging: bool,
    attachments: dict[str, str] | None = None,
) -> tuple[list[Any], dict | None]:
    """Return ``base_messages`` plus recorded native history after its final user.

    The join is accepted only when the official transcript ends in one user message
    and a target event input contains exactly one text-identical copy of that user
    message (allowing OpenCode's known literal quote wrapper).  The newest matching
    event has the fullest history.  A truly empty terminal model output is omitted;
    observable reasoning is retained.
    """

    base = list(base_messages)
    all_events = list(events)
    if not base or _get(base[-1], "role") != "user":
        return base, None
    if not isinstance(event_start, int) or event_start < 0:
        return base, None

    boundary_text = _message_text(base[-1], attachments)
    selected = _target_events(all_events[event_start:], target_model)
    if not selected:
        return base, None

    chosen: tuple[int, Any, list[Any], int, str] | None = None
    for relative_index, event in reversed(selected):
        native_input = list(_get(event, "input", []) or [])
        matches: list[tuple[int, str]] = []
        for message_index, message in enumerate(native_input):
            if _get(message, "role") != "user":
                continue
            candidate_text, wrapper = _native_user_text(message, attachments)
            if candidate_text == boundary_text:
                matches.append((message_index, wrapper))
        if len(matches) == 1:
            match_index, wrapper = matches[0]
            chosen = (
                event_start + relative_index,
                event,
                native_input,
                match_index,
                wrapper,
            )
            break
    if chosen is None:
        return base, None

    event_index, event, native_input, match_index, wrapper = chosen
    recovered = list(native_input[match_index + 1:])
    output_message = _event_output_message(event)
    terminal_output_included = False
    if _observable_message(output_message):
        output_id = _get(output_message, "id")
        known_ids = {
            str(_get(message, "id"))
            for message in [*base, *recovered]
            if _get(message, "id") is not None
        }
        if output_id is None or str(output_id) not in known_ids:
            recovered.append(output_message)
            terminal_output_included = True

    if not recovered:
        return base, None

    merged = [*base, *recovered]
    record = {
        "format": RECOVERY_FORMAT,
        "scaffold": "opencode",
        "source": "target_model_event_inputs_and_terminal_output",
        "reconstructed": True,
        "applied_before_judging": bool(applied_before_judging),
        "original_message_count": len(base),
        "reconstructed_message_count": len(merged),
        "added_message_count": len(recovered),
        "matched_user_wrapper": wrapper,
        "terminal_output_included": terminal_output_included,
        "terminal_event_index": event_index,
        "terminal_event_id": (
            str(_get(event, "uuid") or _get(event, "id"))
            if _get(event, "uuid") is not None or _get(event, "id") is not None
            else None
        ),
        "terminal_event_timestamp": (
            str(_get(event, "timestamp"))
            if _get(event, "timestamp") is not None else None
        ),
        "coverage": "through_last_matching_recorded_target_model_event",
        "complete": False,
        "limitation": (
            "The native OpenCode call did not return an AgentState. Recorded model "
            "history was recovered through the newest target event matching the "
            "interrupted user turn; activity after that event and an interrupted tool "
            "result may be unavailable."
        ),
    }
    return merged, record
