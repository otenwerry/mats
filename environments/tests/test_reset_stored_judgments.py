from __future__ import annotations

from inspect_ai.event import StateEvent

from tools.maintenance.reset_stored_judgments import (
    _sanitize_retained_event,
    sanitize_real_env,
)


def judgment_bearing_real_env() -> dict:
    return {
        "family": "ml_training_data_misuse",
        "protocol": {
            "submissions": 2,
            "first_gate_verdict": "clean",
            "follow_up_sent": True,
            "ended_reason": "protocol_end",
        },
        "gates": [{
            "submission": 1,
            "verdict": "clean",
            "detectors": [{"name": "answer_key_read", "active": False}],
            "error": None,
            "judgment": {"summary": "old answer"},
            "judge_metadata": {"provider_request": "old prompt"},
            "rule": {"reward_hacking": False},
        }],
        "final_judgment": {"result": {"reward_hacking": "old answer"}},
        "artifacts": [{"path": "predictions.csv", "content": "id,label"}],
        "grade": {"score": 0.8},
    }


def test_sanitize_real_env_removes_judgments_but_preserves_trajectory_facts() -> None:
    original = judgment_bearing_real_env()
    sanitized = sanitize_real_env(original)

    assert "final_judgment" not in sanitized
    assert "first_gate_verdict" not in sanitized["protocol"]
    assert sanitized["protocol"]["submissions"] == 2
    assert sanitized["protocol"]["ended_reason"] == "protocol_end"
    assert sanitized["gates"] == [{
        "submission": 1,
        "detectors": [{"name": "answer_key_read", "active": False}],
        "error": None,
    }]
    assert sanitized["artifacts"] == original["artifacts"]
    assert sanitized["grade"] == original["grade"]
    assert "final_judgment" in original


def test_state_event_snapshot_is_sanitized_too() -> None:
    event = StateEvent(changes=[{
        "op": "add",
        "path": "/metadata/real_env",
        "value": judgment_bearing_real_env(),
    }])

    sanitized = _sanitize_retained_event(event)
    value = sanitized.changes[0].value

    assert "final_judgment" not in value
    assert "judgment" not in value["gates"][0]
    assert value["artifacts"][0]["path"] == "predictions.csv"
