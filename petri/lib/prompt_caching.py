"""Reliable provider prompt caching for Petri's paid model calls.

There are two different kinds of caching in this stack:

* Inspect's ``cache=`` / ``CachePolicy`` replays a previously generated RESPONSE.
  Petri deliberately leaves that off because epochs must be independent samples.
* Provider prompt caching reuses only the prompt's KV prefix.  Every request still
  generates a fresh response.  Inspect enables this by default for supported model
  providers, and direct Anthropic SDK calls opt in with ``cache_control`` below.

The remaining concurrency trap is that a provider cache entry is unavailable until
the first matching request starts returning.  Petri normally launches all N epochs at
once, so N identical initial calls can otherwise all miss. ``install_inspect_warmup``
places a one-time barrier around identical, cache-sized first requests: one request
finishes, then the waiters make their own normal paid calls. No model output is shared.

Every generation pipeline also calls ``write_report``. The resulting
``prompt_cache_report.json`` keeps cache reads/writes and barrier evidence queryable in
the run directory; a zero is never described as a verified hit.

OpenRouter calls with cache-sized histories also receive a stable ``session_id`` based
on the conversation opening. This pins subsequent calls to the provider endpoint that
handled the first successful request, improving implicit-cache reliability without
reusing any response.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from inspect_ai.model import ChatMessageBase, GenerateConfig


logger = logging.getLogger(__name__)
T = TypeVar("T")

# A conservative character proxy for the lowest common 1,024-token provider floor.
# It can occasionally wait on a prompt that tokenizes below the threshold; that costs
# one response of latency, never money or sample independence. Larger Opus thresholds
# are handled by Anthropic itself and surfaced honestly in the post-run report.
_MIN_CACHE_PROMPT_CHARS = 3_000
_CACHE_CONTROL = {"type": "ephemeral"}  # five minutes; batch calls happen immediately


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_key(namespace: str, *parts: Any) -> str:
    """Hash exact reusable-prefix inputs without retaining their potentially huge text."""
    raw = json.dumps(
        [namespace, *(_jsonable(part) for part in parts)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


class PromptCacheWarmupBarrier:
    """Let exactly one matching request finish before its parallel peers start.

    The barrier stores only hashes. A successful warm-up opens the key permanently for
    this process; a failed warm-up leaves it cold so the next waiter gets a chance.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._warmed: set[str] = set()
        self.waited_requests = 0
        self.warmed_prefixes = 0
        self.failed_warmups = 0

    async def run(
        self,
        key: str,
        call: Callable[[], Awaitable[T]],
        *,
        succeeded: Callable[[T], bool] | None = None,
    ) -> T:
        if key in self._warmed:
            return await call()
        lock = self._locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            self.waited_requests += 1
        async with lock:
            if key in self._warmed:
                # Release the per-prefix lock before making this fresh call. All waiters
                # should fan out after the one warm-up, not run serially one by one.
                pass
            else:
                try:
                    result = await call()
                except BaseException:
                    self.failed_warmups += 1
                    raise
                ok = succeeded(result) if succeeded is not None else True
                if ok:
                    self._warmed.add(key)
                    self.warmed_prefixes += 1
                else:
                    self.failed_warmups += 1
                return result
        return await call()

    def stats(self) -> dict[str, int]:
        return {
            "warmed_prefixes": self.warmed_prefixes,
            "requests_held_for_warmup": self.waited_requests,
            "failed_warmups": self.failed_warmups,
        }


_inspect_barrier = PromptCacheWarmupBarrier()
_direct_barrier = PromptCacheWarmupBarrier()
_installed = False


def cached_system(text: str) -> list[dict[str, Any]]:
    """Anthropic system content with a five-minute explicit cache breakpoint."""
    return [{"type": "text", "text": text, "cache_control": dict(_CACHE_CONTROL)}]


def cached_user_prefix(prefix: str, suffix: str) -> list[dict[str, Any]]:
    """Anthropic user content split after a repeated, cacheable prefix."""
    return [
        {"type": "text", "text": prefix, "cache_control": dict(_CACHE_CONTROL)},
        {"type": "text", "text": suffix},
    ]


async def run_direct_cached(prefix_key: str, call: Callable[[], Awaitable[T]]) -> T:
    """Warm an explicitly marked direct-Anthropic prefix before parallel reuse."""
    return await _direct_barrier.run(prefix_key, call)


def _inspect_succeeded(result: Any) -> bool:
    output = result[0] if isinstance(result, tuple) and result else result
    return not isinstance(output, BaseException)


def _cache_key_message(message: Any) -> Any:
    """Serialize one message without Inspect's provider-invisible bookkeeping ID.

    Inspect gives each ChatMessage a fresh top-level `id`, including messages it creates
    independently for otherwise identical epochs. Anthropic, OpenAI, and OpenRouter do not
    send that field as prompt content, so including it prevents identical provider requests
    from meeting at the warm-up barrier. Strip only this top-level ChatMessage field. Nested
    tool-call IDs remain intact because providers do send them and tool results refer to them.
    """
    if not isinstance(message, ChatMessageBase):
        return _jsonable(message)
    payload = message.model_dump(mode="json", exclude_none=False)
    payload.pop("id", None)
    return _jsonable(payload)


def _openrouter_session_id(model: Any, input_messages: Any) -> str | None:
    """Stable provider-routing session for one OpenRouter conversation.

    OpenRouter's implicit sticky key is based on the first system/developer message and
    first non-system message. Mirror that boundary explicitly so routing becomes sticky
    after the first successful request, rather than only after OpenRouter observes a cache
    hit. The ID controls provider routing only; provider caches still compare exact prompts.
    """
    if not isinstance(input_messages, (list, tuple)) or not input_messages:
        return None
    first_system = next(
        (message for message in input_messages
         if getattr(message, "role", None) == "system"),
        None,
    )
    first_non_system = next(
        (message for message in input_messages
         if getattr(message, "role", None) != "system"),
        None,
    )
    opening = [
        _cache_key_message(message)
        for message in (first_system, first_non_system)
        if message is not None
    ]
    if not opening:
        return None
    digest = stable_key(
        "openrouter-sticky-session-v1",
        getattr(model, "model_name", None),
        getattr(model, "base_url", None),
        opening,
    )
    return f"petri-prefix-{digest[:48]}"


def _with_openrouter_session(
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Copy one provider call with a derived session_id; never mutate caller config."""
    input_messages = kwargs.get("input", args[0] if len(args) > 0 else None)
    session_id = _openrouter_session_id(model, input_messages)
    if session_id is None:
        return args, kwargs

    config = kwargs.get("config", args[3] if len(args) > 3 else None)
    if config is None:
        config = GenerateConfig()
    if not isinstance(config, GenerateConfig):
        return args, kwargs
    extra_body = dict(config.extra_body or {})
    # A future pipeline may deliberately select its own routing boundary. Respect it.
    if "session_id" in extra_body:
        return args, kwargs
    extra_body["session_id"] = session_id
    routed_config = config.model_copy(update={"extra_body": extra_body}, deep=True)

    routed_args = list(args)
    routed_kwargs = dict(kwargs)
    if "config" in routed_kwargs:
        routed_kwargs["config"] = routed_config
    elif len(routed_args) > 3:
        routed_args[3] = routed_config
    else:
        routed_kwargs["config"] = routed_config
    return tuple(routed_args), routed_kwargs


def _inspect_request_key(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    # Inspect's provider generate signature is (input, tools, tool_choice, config), but
    # callers may use positional or keyword arguments.
    input_messages = kwargs.get("input", args[0] if len(args) > 0 else None)
    tools = kwargs.get("tools", args[1] if len(args) > 1 else None)
    tool_choice = kwargs.get("tool_choice", args[2] if len(args) > 2 else None)
    config = kwargs.get("config", args[3] if len(args) > 3 else None)
    if input_messages is None:
        return None
    # The varying question/new task is normally the final user message (or folded into
    # it). Provider caches can reuse the stable history before that message, so warm on
    # that prefix rather than requiring the request suffix to be identical. This is what
    # lets a resumed context asked N different questions benefit from one warm-up.
    message_list = list(input_messages) if isinstance(input_messages, (list, tuple)) else None
    stable_messages = message_list[:-1] if message_list else input_messages
    stable_message_payload = (
        [_cache_key_message(message) for message in stable_messages]
        if isinstance(stable_messages, list)
        else _cache_key_message(stable_messages)
    )
    payload = _jsonable([stable_message_payload, tools, tool_choice, config])
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(serialized) < _MIN_CACHE_PROMPT_CHARS:
        return None
    return stable_key(
        "inspect-provider-request-v2",
        getattr(model, "model_name", None),
        getattr(model, "base_url", None),
        payload,
    )


def _patch_provider(cls: type, *, openrouter: bool = False) -> None:
    original = cls.generate
    if getattr(original, "_petri_prompt_cache_warmup", False):
        return

    async def generate(self, *args, **kwargs):
        key = _inspect_request_key(self, args, kwargs)
        if key is None:
            return await original(self, *args, **kwargs)
        call_args, call_kwargs = (
            _with_openrouter_session(self, args, kwargs)
            if openrouter else (args, kwargs)
        )
        # Include a caller-supplied or derived session_id in the local grouping key too.
        key = _inspect_request_key(self, call_args, call_kwargs)
        assert key is not None
        return await _inspect_barrier.run(
            key,
            lambda: original(self, *call_args, **call_kwargs),
            succeeded=_inspect_succeeded,
        )

    generate._petri_prompt_cache_warmup = True
    cls.generate = generate


def install_inspect_warmup() -> bool:
    """Install the fresh-response warm-up barrier on Inspect's paid API providers."""
    global _installed
    if _installed:
        return True
    try:
        from inspect_ai.model._providers.anthropic import AnthropicAPI
        from inspect_ai.model._providers.openai import OpenAIAPI
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
    except Exception as exc:
        logger.warning(
            "prompt_caching: Inspect provider imports moved; warm-up barrier DISABLED (%s)",
            exc,
        )
        return False
    _patch_provider(AnthropicAPI)
    _patch_provider(OpenAIAPI)
    _patch_provider(OpenRouterAPI, openrouter=True)
    _installed = True
    logger.info("prompt_caching: provider prompt-cache warm-up barrier installed")
    return True


def _model_status(model: str, cache_read: int, cache_write: int) -> str:
    if cache_read > 0:
        return "verified_cache_read"
    if cache_write > 0:
        return "cache_write_only_no_reuse_observed"
    if model.startswith(("anthropic/", "openai/", "openrouter/")):
        return "no_cache_activity_observed"
    return "provider_cache_support_unknown"


def build_report(run_dir: Path) -> dict[str, Any]:
    """Read Inspect headers and build a provider-neutral, non-lossy cache report."""
    from inspect_ai.log import read_eval_log

    totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_write_input_tokens": 0, "total_cost_usd": 0.0}
    )
    n_logs = 0
    for path in sorted(Path(run_dir).glob("*.eval")):
        log = read_eval_log(str(path), header_only=True)
        n_logs += 1
        for model, usage in (log.stats.model_usage or {}).items():
            row = totals[model]
            row["input_tokens"] += usage.input_tokens or 0
            row["output_tokens"] += usage.output_tokens or 0
            row["cache_read_input_tokens"] += usage.input_tokens_cache_read or 0
            row["cache_write_input_tokens"] += usage.input_tokens_cache_write or 0
            row["total_cost_usd"] += usage.total_cost or 0.0

    models = {}
    for model, raw in sorted(totals.items()):
        row = dict(raw)
        row["total_cost_usd"] = round(float(row["total_cost_usd"]), 8)
        row["status"] = _model_status(
            model,
            int(row["cache_read_input_tokens"]),
            int(row["cache_write_input_tokens"]),
        )
        denom = int(row["input_tokens"]) + int(row["cache_read_input_tokens"]) + int(
            row["cache_write_input_tokens"]
        )
        row["cache_read_fraction_of_prompt_tokens"] = (
            round(int(row["cache_read_input_tokens"]) / denom, 6) if denom else None
        )
        models[model] = row

    return {
        "schema_version": 3,
        "semantics": "provider prompt-prefix caching; every response remains freshly generated",
        "inspect_response_cache_enabled": False,
        "attribution": (
            "Inspect header totals are grouped by model slug; distinct roles that use "
            "the same slug are combined"
        ),
        "n_eval_logs": n_logs,
        "warmup_barrier": {
            "installed": _installed,
            "key_version": "v2-ignore-provider-invisible-chat-message-id",
            "openrouter_session_routing": "v1-stable-from-conversation-opening",
            "inspect": _inspect_barrier.stats(),
            "direct_anthropic": _direct_barrier.stats(),
        },
        "models": models,
        "verified_models": [m for m, row in models.items()
                            if row["status"] == "verified_cache_read"],
        "unverified_models": [m for m, row in models.items()
                              if row["status"] != "verified_cache_read"],
    }


def write_report(run_dir: Path) -> dict[str, Any]:
    report = build_report(Path(run_dir))
    path = Path(run_dir) / "prompt_cache_report.json"
    path.write_text(json.dumps(report, indent=2))
    read = sum(int(row["cache_read_input_tokens"]) for row in report["models"].values())
    write = sum(int(row["cache_write_input_tokens"]) for row in report["models"].values())
    print(
        f"[prompt cache] {read:,} tokens read, {write:,} written; "
        f"verified models={len(report['verified_models'])}, "
        f"unverified={len(report['unverified_models'])}; evidence -> {path}",
        flush=True,
    )
    return report
