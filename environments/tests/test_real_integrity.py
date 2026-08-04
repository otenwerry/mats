"""Free tests for shared pipeline/viewer integrity flags."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from real_integrity import (  # noqa: E402
    finalize_audit_integrity,
    integrity_sidecar_records,
    target_provider_events,
)


def ns(**values):
    return SimpleNamespace(**values)


def test_nonzero_pipeline_exit_and_clock_limit_are_flags_not_exclusions() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
            "compute": {"pipeline_exit_code": 1},
        },
        "real_ended_reason": "wall_clock_limit",
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert audit["integrity_status"] == "included"
    assert {flag["code"] for flag in audit["flags"]} == {
        "pipeline_exit_nonzero", "wall_clock_limit",
    }


def test_unrecovered_empty_target_response_is_stored_and_excluded() -> None:
    message = ns(content="", tool_calls=[], refusal=None)
    choice = ns(message=message, stop_reason="stop")
    event = ns(
        event="model", role="target", model="mock/target", input=["prompt"],
        tools=[], config={}, output=ns(choices=[choice], usage=None), error=None,
    )
    sample = ns(events=[event])

    events = target_provider_events(sample, "mock/target")
    audit = {
        "real_env": {"protocol": {}, "artifacts": []},
        "target_tools_mode": "real",
        "judgment": None,
        "target_provider_events": events,
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }
    finalize_audit_integrity(audit)

    assert events == [{"kind": "empty_response", "attempt": 1, "stop_reason": "stop"}]
    assert "target_provider_empty_response" in audit["integrity_issues"]
    assert audit["integrity_status"] == "excluded"


def test_pipeline_sidecar_records_per_sample_provider_failure() -> None:
    message = ns(content="", tool_calls=[], refusal=None)
    event = ns(
        event="model", role="target", model="mock/target", input=["prompt"],
        tools=[], config={}, output=ns(
            choices=[ns(message=message, stop_reason="content_filter")], usage=None
        ), error=None,
    )
    sample = ns(
        id="fraud_detection", epoch=3, events=[event], messages=[], role_usage={},
        metadata={"real_env": {
            "protocol": {"ended_reason": "cancelled"},
            "artifacts": [{"path": "submission.csv"}],
            "gates": [],
        }},
        scores={},
    )
    log = ns(
        eval=ns(task="real_audit_test", model_roles={"target": ns(model="mock/target")}),
        samples=[sample],
    )

    records = integrity_sidecar_records([log])

    assert records[0]["status"] == "excluded"
    assert records[0]["task"] == "real_audit_test"
    assert "target_provider_content_filter" in records[0]["issues"]
    assert records[0]["target_provider_events"][0]["kind"] == "content_filter"
