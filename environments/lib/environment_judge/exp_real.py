"""Paid current judge call for one stored or live real trajectory."""

from __future__ import annotations

from inspect_ai.model import ChatMessage, Model

from real_judge_evidence import complete_real_trajectory_evidence

from .exp_runner import judge_trajectory


async def judge_complete_real_trajectory(
    *,
    family,
    messages: list[ChatMessage],
    real_env: dict,
    model: str | Model | None,
):
    """Apply the current final-judgment prompt/schema/evidence policy."""

    artifacts, caveats = complete_real_trajectory_evidence(messages, real_env)
    return await judge_trajectory(
        family=family,
        stage="final",
        messages=messages,
        artifacts=artifacts,
        upstream_caveats=caveats,
        model=model,
    )
