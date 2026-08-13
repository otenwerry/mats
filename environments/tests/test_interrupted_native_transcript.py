"""Free tests for conservative interrupted OpenCode transcript recovery."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from interrupted_native_transcript import (  # noqa: E402
    RECOVERY_FORMAT,
    recover_interrupted_opencode_messages,
)
from env_viewer_load import sample_to_audit  # noqa: E402


def ns(**values):
    return SimpleNamespace(**values)


def message(role: str, content: object, *, message_id: str) -> SimpleNamespace:
    return ns(role=role, content=content, id=message_id, tool_calls=[])


def model_event(input_messages: list, output_message: object) -> SimpleNamespace:
    return ns(
        event="model",
        role="target",
        model="mock/glm",
        input=input_messages,
        output=ns(message=output_message, choices=[]),
        id="event-final",
        timestamp="2026-08-13T12:00:00Z",
    )


def test_recovers_newest_native_history_after_exact_quoted_user_boundary() -> None:
    system = message("system", "system", message_id="m0")
    first_user = message("user", "first task", message_id="m1")
    first_answer = message("assistant", "first answer", message_id="m2")
    followup = message("user", "Please finish.", message_id="m3")
    native_followup = message("user", '"Please finish."', message_id="native-u2")
    reasoning = message(
        "assistant",
        [ns(type="reasoning", reasoning="I should inspect the file.")],
        message_id="m4",
    )
    tool_result = message("tool", "file contents", message_id="m5")
    final = message("assistant", "Finished.", message_id="m6")
    event = model_event([native_followup, reasoning, tool_result], final)

    recovered, record = recover_interrupted_opencode_messages(
        [system, first_user, first_answer, followup],
        [event],
        target_model="mock/glm",
        applied_before_judging=True,
    )

    assert recovered == [
        system, first_user, first_answer, followup, reasoning, tool_result, final,
    ]
    assert record is not None
    assert record["format"] == RECOVERY_FORMAT
    assert record["matched_user_wrapper"] == "outer_double_quotes"
    assert record["added_message_count"] == 3
    assert record["terminal_output_included"] is True
    assert record["applied_before_judging"] is True
    assert record["complete"] is False


def test_omits_truly_empty_terminal_output_but_keeps_recorded_history() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    action = message("assistant", "working", message_id="a2")
    empty = message("assistant", "", message_id="empty")

    recovered, record = recover_interrupted_opencode_messages(
        [followup],
        [model_event([native_followup, action], empty)],
        applied_before_judging=False,
    )

    assert recovered == [followup, action]
    assert record is not None
    assert record["added_message_count"] == 1
    assert record["terminal_output_included"] is False


def test_fails_closed_when_boundary_is_ambiguous_or_not_final_user() -> None:
    followup = message("user", "continue", message_id="u2")
    duplicate_a = message("user", '"continue"', message_id="native-a")
    duplicate_b = message("user", "continue", message_id="native-b")
    output = message("assistant", "answer", message_id="a2")

    ambiguous, ambiguous_record = recover_interrupted_opencode_messages(
        [followup],
        [model_event([duplicate_a, duplicate_b], output)],
        applied_before_judging=False,
    )
    complete, complete_record = recover_interrupted_opencode_messages(
        [followup, output],
        [model_event([duplicate_a], output)],
        applied_before_judging=False,
    )

    assert ambiguous == [followup]
    assert ambiguous_record is None
    assert complete == [followup, output]
    assert complete_record is None


def test_viewer_marks_historical_judgment_that_missed_recovered_messages() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    action = message("assistant", "second-turn work", message_id="a2")
    final = message("assistant", "second-turn answer", message_id="a3")
    event = model_event([native_followup, action], final)
    event.working_start = 10.0
    event.output.usage = ns(
        input_tokens=100,
        input_tokens_cache_read=0,
        input_tokens_cache_write=0,
    )
    event.error = None
    event.tools = []
    event.config = {}
    sample = ns(
        id="seed", epoch=1, messages=[followup], events=[event], attachments={},
        metadata={"real_env": {
            "harness": {"mode": "production", "scaffold": "opencode"},
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": {"files": []},
        }},
        scores={
            "audit_judge": ns(value={"behavior": 4}, metadata={}, explanation="")
        },
        role_usage={"target": ns(output_tokens=20)}, model_usage={}, started_at=None,
    )

    audit = sample_to_audit(
        mode="run", mode_mtime=0.0, task="task", run_metadata={
            "target_tools_mode": "real",
            "harness": "production",
        },
        roles={"target": ns(model="mock/glm")}, sample=sample,
    )

    assert audit["transcript_source"] == "reconstructed_interrupted_opencode_events"
    assert [item["text"] for item in audit["messages"]] == [
        "continue", "second-turn work", "second-turn answer",
    ]
    assert audit["judgment_transcript_coverage"] == {
        "complete": False,
        "stored_judgment_predates_reconstruction": True,
        "messages_seen_by_stored_judgment": 1,
        "messages_shown_after_reconstruction": 3,
        "recovered_messages_not_seen_by_stored_judgment": 2,
    }
    assert "judge_missing_recovered_messages" in audit["integrity_issues"]
