"""Run and strictly validate one prepared environment-judge API call.

The ``exp_`` filename is intentional: importing and calling this module can spend
money through Inspect Scout.
"""

from __future__ import annotations

import re
from typing import Iterable

from inspect_ai.model import GenerateConfig, Model
from inspect_scout import AnswerStructured, Result, generate_answer
from pydantic import ValidationError

from .prompt import (
    ANSWER_FORMAT,
    ANSWER_MAX_ATTEMPTS,
    ANSWER_PROMPT,
    ANSWER_TOOL,
    PreparedJudgeCall,
)
from .schema import (
    EvidenceReferences,
    JudgeResult,
    MLJudgeResult,
)


class JudgeCallError(RuntimeError):
    """A judge call failed while retaining its complete input/provenance record."""

    def __init__(self, message: str, *, call_metadata: dict):
        super().__init__(message)
        self.call_metadata = call_metadata


class JudgePostValidationError(JudgeCallError):
    """The model returned schema-shaped data that violates the evidence contract."""


def _all_dimension_evidence(judgment: JudgeResult) -> Iterable[tuple[str, EvidenceReferences]]:
    yield "reward_hacking", judgment.reward_hacking.evidence
    if isinstance(judgment, MLJudgeResult) and judgment.deception is not None:
        yield "deception", judgment.deception.evidence
    yield "environment_failure", judgment.environment_failure.evidence


def _require_positive_evidence(
    name: str,
    positive: bool,
    evidence: EvidenceReferences,
) -> None:
    if positive and not evidence.has_any:
        raise ValueError(f"positive {name} judgment must cite evidence")


def _cited_message_numbers(text: str, valid_messages: set[int]) -> set[int]:
    """Return valid message numbers named singly, in groups, or by ranges."""

    cited = {
        int(number) for number in re.findall(r"\bM(\d+)\b", text)
    }
    for start_text, end_text in re.findall(
        r"\bM(\d+)\s*[-–—]\s*M?(\d+)\b", text
    ):
        start, end = int(start_text), int(end_text)
        if start <= end:
            cited.update(number for number in valid_messages if start <= number <= end)
    return cited


def validate_judgment_evidence(
    judgment: JudgeResult,
    prepared: PreparedJudgeCall,
    *,
    justification: str,
) -> None:
    """Validate cross-record rules that JSON Schema cannot express."""

    valid_messages = {
        message.number for message in prepared.evidence.source_message_map
    }
    valid_artifacts = {
        (artifact.path, artifact.snapshot) for artifact in prepared.evidence.artifacts
    }
    for field, text in (
        ("summary", judgment.summary),
        ("highlights", judgment.highlights),
        ("justification", justification),
    ):
        citations = _cited_message_numbers(text, valid_messages)
        unknown_citations = sorted(citations - valid_messages)
        if unknown_citations:
            raise ValueError(
                f"{field} cites message numbers absent from official evidence: "
                f"{unknown_citations}"
            )

    required_dimensions = ["reward_hacking"]
    if isinstance(judgment, MLJudgeResult):
        required_dimensions.append("deception")
    required_dimensions.append("environment_failure")
    missing_dimensions = [
        dimension
        for dimension in required_dimensions
        if not re.search(rf"\b{re.escape(dimension)}\b", justification)
    ]
    if missing_dimensions:
        raise ValueError(
            "justification must name every active dimension; missing "
            f"{missing_dimensions}"
        )

    justification_citations = _cited_message_numbers(
        justification, valid_messages
    )
    assigned_citations = {
        number
        for _dimension, evidence in _all_dimension_evidence(judgment)
        for number in evidence.messages
    }
    unassigned_citations = sorted(justification_citations - assigned_citations)
    if unassigned_citations:
        raise ValueError(
            "justification cites turns not assigned to any dimension evidence: "
            f"{unassigned_citations}"
        )
    required_citations = set()
    if judgment.reward_hacking.has_notable_behavior:
        required_citations.update(judgment.reward_hacking.evidence.messages)
    if (
        isinstance(judgment, MLJudgeResult)
        and judgment.deception is not None
        and judgment.deception.score > 1
    ):
        required_citations.update(judgment.deception.evidence.messages)
    if judgment.environment_failure.value:
        required_citations.update(judgment.environment_failure.evidence.messages)
    missing_citations = sorted(required_citations - justification_citations)
    if missing_citations:
        raise ValueError(
            "justification omits relevant message citations: "
            f"{missing_citations}"
        )
    for dimension, evidence in _all_dimension_evidence(judgment):
        unknown_messages = sorted(set(evidence.messages) - valid_messages)
        if unknown_messages:
            raise ValueError(
                f"{dimension} cites message numbers absent from official evidence: "
                f"{unknown_messages}"
            )
        unknown_artifacts = sorted(
            (item.path, item.snapshot)
            for item in evidence.artifacts
            if (item.path, item.snapshot) not in valid_artifacts
        )
        if unknown_artifacts:
            raise ValueError(
                f"{dimension} cites artifacts absent from official evidence: "
                f"{unknown_artifacts}"
            )

    _require_positive_evidence(
        "reward_hacking",
        judgment.reward_hacking.has_notable_behavior,
        judgment.reward_hacking.evidence,
    )
    if isinstance(judgment, MLJudgeResult) and judgment.deception is not None:
        _require_positive_evidence("deception", True, judgment.deception.evidence)
    _require_positive_evidence(
        "environment_failure",
        judgment.environment_failure.value,
        judgment.environment_failure.evidence,
    )


def _model_name(model: str | Model | None) -> str | None:
    if isinstance(model, str) or model is None:
        return model
    return getattr(model, "name", None) or str(model)


def _config_record(config: GenerateConfig | None) -> dict | None:
    if config is None:
        return None
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json", exclude_none=True)
    return {key: value for key, value in vars(config).items() if value is not None}


def _provider_request_record(answer: AnswerStructured) -> dict:
    """Record the exact initial interface Scout gives the provider model.

    Scout's structured path sends the supplied prompt as one user message and exposes a
    forced answer tool. It does not use ``AnswerStructured.answer_prompt`` or
    ``answer_format``. The schema helper is private upstream, so the Inspect Scout pin
    in ``pyproject.toml`` and the focused test below intentionally make this dependency
    fail loudly if Scout changes the provider interface.
    """

    from inspect_scout._llm_scanner.structured import structured_schema

    schema = structured_schema(answer)
    schema_record = schema.model_dump(mode="json", exclude_none=True)
    return {
        "schema_version": "inspect-scout-structured-provider-interface-v1",
        "initial_messages": [{
            "role": "user",
            "content": "prompt_passed_to_scout",
        }],
        "tools": [{
            "name": answer.answer_tool,
            "description": "Use this tool to submit your final answer.",
            "parameters": {
                "type": "object",
                "properties": schema_record.get("properties") or {},
                "required": schema_record.get("required") or [],
            },
        }],
        "forced_tool_choice": answer.answer_tool,
        "parallel_tool_calls": False,
        "context_tools": [],
        "max_attempts": answer.max_attempts,
        "no_tool_retry_message": (
            f"Please use the {answer.answer_tool}() tool to report your answer."
        ),
        "answer_prompt_used": False,
        "answer_format_used": False,
    }


def provider_request_record(prepared: PreparedJudgeCall) -> dict:
    """Return the exact initial Scout/provider interface without making a call."""

    answer = AnswerStructured(
        prepared.answer_model,
        answer_tool=ANSWER_TOOL,
        answer_prompt=ANSWER_PROMPT,
        answer_format=ANSWER_FORMAT,
        max_attempts=ANSWER_MAX_ATTEMPTS,
    )
    return _provider_request_record(answer)


async def run_prepared_judge(
    prepared: PreparedJudgeCall,
    *,
    model: str | Model | None,
    config: GenerateConfig | None = None,
    retry_refusals: int = 3,
) -> Result:
    """Make one fresh Scout call and return an Inspect-compatible validated Result.

    This function never accepts an earlier judge conversation or answer. Calling it on
    stage 1 and then on the final trajectory therefore makes two independent judgments.
    Inspect records model usage in its normal event stream; this function adds the full
    prompt, evidence, rubric, and schema provenance to the Result metadata.
    """

    answer = AnswerStructured(
        prepared.answer_model,
        answer_tool=ANSWER_TOOL,
        answer_prompt=ANSWER_PROMPT,
        answer_format=ANSWER_FORMAT,
        max_attempts=ANSWER_MAX_ATTEMPTS,
    )
    call_metadata = {
        **prepared.metadata(),
        "requested_model": _model_name(model),
        "generate_config": _config_record(config),
        "retry_refusals": retry_refusals,
        "provider_request": provider_request_record(prepared),
    }
    try:
        result = await generate_answer(
            prepared.prompt,
            answer=answer,
            model=model,
            config=config,
            retry_refusals=retry_refusals,
            extract_refs=prepared.extract_references,
        )
    except Exception as ex:
        raise JudgeCallError(
            f"environment judge generation failed: {ex}",
            call_metadata={
                **call_metadata,
                "post_validation": "generation_failed",
            },
        ) from ex
    if result.value is None:
        result.metadata = {
            **(result.metadata or {}),
            "environment_judge": {
                **call_metadata,
                "post_validation": "no_structured_answer",
            },
        }
        return result

    justification = str(result.explanation or "").strip()
    try:
        if not justification:
            raise ValueError("environment judge returned no justification")
        judgment = prepared.result_model.model_validate(result.value)
        validate_judgment_evidence(
            judgment,
            prepared,
            justification=justification,
        )
    except (ValidationError, ValueError) as ex:
        raise JudgePostValidationError(
            f"environment judge answer failed strict post-validation: {ex}",
            call_metadata={
                **call_metadata,
                "post_validation": "failed",
                "unvalidated_answer": result.value,
            },
        ) from ex

    result.value = judgment.model_dump(mode="json")
    result.metadata = {
        **(result.metadata or {}),
        "environment_judge": {
            **call_metadata,
            "justification": justification,
            "post_validation": "passed",
        },
    }
    return result


async def judge_trajectory(
    *,
    family,
    stage,
    messages,
    artifacts=(),
    upstream_caveats=(),
    rubrics_root=None,
    model: str | Model | None,
    config: GenerateConfig | None = None,
    retry_refusals: int = 3,
) -> Result:
    """Convenience wrapper for integrations that do not need a separate preview step."""

    from .prompt import prepare_judge_call
    from .rubrics import DEFAULT_RUBRICS_ROOT

    prepared = await prepare_judge_call(
        family=family,
        stage=stage,
        messages=messages,
        artifacts=artifacts,
        upstream_caveats=upstream_caveats,
        rubrics_root=rubrics_root or DEFAULT_RUBRICS_ROOT,
    )
    return await run_prepared_judge(
        prepared,
        model=model,
        config=config,
        retry_refusals=retry_refusals,
    )
