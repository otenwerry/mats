"""One provider-agnostic structured-output call, for the secondary judge roles.

The secondary judges (auditor deviation, auditor faithfulness, the rollback follow-up
annotation, mechanism similarity) were written against the Anthropic SDK
(`client.messages.parse(..., output_format=SomeModel)`), which locked them to Anthropic
models. This helper does the same job through inspect's model layer, so they can run on
ANY registered model -- including the default judge (Owen 2026-07-30: use Luna for these
too).

Same contract as the call it replaces: give it a system prompt, a user prompt, and a
pydantic model; get back a validated instance plus a usage dict. Structured output is
requested natively (GenerateConfig.response_schema), and `strict` is enabled where the
provider supports it, so a well-formed reply is the provider's job rather than ours. One
repair retry covers providers that only best-effort the schema.

Named exp_* because it makes paid model calls. Importing is free.
"""

from __future__ import annotations

import json
from typing import TypeVar

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
)
from inspect_ai.util import json_schema
from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

DEFAULT_MAX_TOKENS = 8000
# Providers that honour OpenAI-style strict schema adherence (per ResponseSchema.strict).
_STRICT_PREFIXES = ("openai/", "mistral/")

_REPAIR = (
    "Your previous reply did not parse as the required JSON object: {error}\n"
    "Reply again with ONLY the JSON object, matching the schema exactly. No prose, no "
    "code fences."
)


def usage_dict(model_name: str, usage) -> dict:
    """ModelUsage -> the shape the rest of the cost system stores
    (viewer_load.usage_to_dict / annotations.json 'usage')."""
    if usage is None:
        return {"model": model_name, "input": 0, "output": 0,
                "cache_read": 0, "cache_write": 0, "total_cost": None}
    return {
        "model": model_name,
        "input": usage.input_tokens or 0,
        "output": usage.output_tokens or 0,
        "cache_read": getattr(usage, "input_tokens_cache_read", None) or 0,
        "cache_write": getattr(usage, "input_tokens_cache_write", None) or 0,
        "total_cost": getattr(usage, "total_cost", None),
    }


def _accumulate(into: dict, extra: dict) -> dict:
    """Sum two usage dicts (a repair retry must not hide its own cost)."""
    if not into:
        return dict(extra)
    out = dict(into)
    for key in ("input", "output", "cache_read", "cache_write"):
        out[key] = (out.get(key) or 0) + (extra.get(key) or 0)
    a, b = out.get("total_cost"), extra.get("total_cost")
    out["total_cost"] = None if a is None and b is None else (a or 0) + (b or 0)
    return out


def _strip_fences(text: str) -> str:
    """Tolerate a ```json fenced reply from a provider that ignores response_schema."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


async def structured_judge(
    model,
    *,
    system: str,
    user: str,
    output_format: type[M],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    schema_name: str | None = None,
    strict: bool | None = None,
) -> tuple[M, dict]:
    """One structured judge call. Returns (validated instance, usage dict).

    Raises ValueError if the model cannot produce a parseable object after one repair
    attempt -- every caller already treats an exception as "this trajectory failed", which
    keeps a malformed reply from being recorded as an empty judgement.
    """
    name = str(model)
    if strict is None:
        strict = name.startswith(_STRICT_PREFIXES)
    config = GenerateConfig(
        max_tokens=max_tokens,
        response_schema=ResponseSchema(
            name=schema_name or output_format.__name__,
            json_schema=json_schema(output_format),
            strict=strict or None,
        ),
    )
    messages = [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
    usage: dict = {}
    last_error = ""
    for attempt in range(2):
        output = await model.generate(input=messages, config=config)
        usage = _accumulate(usage, usage_dict(name, getattr(output, "usage", None)))
        completion = output.completion or ""
        try:
            return output_format.model_validate_json(_strip_fences(completion)), usage
        except (ValidationError, json.JSONDecodeError, ValueError) as ex:
            last_error = str(ex)[:400]
            if attempt == 0:
                messages = messages + [
                    output.message,
                    ChatMessageUser(content=_REPAIR.format(error=last_error)),
                ]
    raise ValueError(
        f"{name} did not return a valid {output_format.__name__} after a repair "
        f"attempt: {last_error}"
    )
