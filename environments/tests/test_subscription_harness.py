"""Free tests for direct subscription transcript and accounting adapters."""

from __future__ import annotations

import asyncio
import base64
import gzip
import importlib.util
import inspect
import pathlib
import sys
import tomllib
from types import SimpleNamespace

from inspect_ai.model import ChatMessageUser


ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_subscription_harness as subscription  # noqa: E402
from cost_tracking import estimate_usage_cost  # noqa: E402
from exp_target_harness import subscription_harness_metadata  # noqa: E402


def _fake_emit(**kwargs):
    return SimpleNamespace(output=SimpleNamespace(message=kwargs["assistant"]))


def test_claude_stream_records_hidden_system_prompt_and_subscription_usage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_claude_stream(
        [
            {
                "type": "system",
                "subtype": "init",
                "claude_code_version": "2.1.220",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 50,
                    },
                },
            },
            {
                "type": "result",
                "session_id": "session",
                "total_cost_usd": 0.123,
                "num_turns": 1,
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="anthropic/claude-opus-4-6",
        reasoning=True,
        include_system_marker=True,
    )

    assert [message.role for message in parsed.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "unavailable" in parsed.messages[0].text.lower()
    assert parsed.usage == [{
        "reported": True,
        "input": 100,
        "output": 20,
        "cache_read": 50,
        "cache_write": 0,
        "reasoning": None,
        "total": 170,
    }]
    assert parsed.rate_limits[0]["api_list_equivalent_usd"] == 0.123
    assert parsed.system_prompt_observed is False
    assert parsed.native_session_id == "session"
    assert parsed.unmetered_model_calls == 0


def test_codex_rollout_records_native_system_usage_and_quota(monkeypatch) -> None:
    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_codex_rollout(
        [
            {
                "type": "session_meta",
                "payload": {
                    "cli_version": "0.146.1",
                    "base_instructions": {"text": "native base"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"text": "reasoning summary"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 25,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 5,
                            "total_tokens": 120,
                        }
                    },
                    "rate_limits": {
                        "primary": {"used_percent": 7.0, "window_minutes": 300}
                    },
                },
            },
        ],
        prior_messages=[],
        routed_slug="openai/gpt-5.6-sol",
        reasoning=True,
        initial_session=True,
    )

    assert [message.role for message in parsed.messages] == [
        "system",
        "system",
        "user",
        "assistant",
    ]
    assert parsed.system_prompt_observed is True
    assert parsed.usage[0]["input"] == 75
    assert parsed.usage[0]["cache_read"] == 25
    assert parsed.usage[0]["reasoning"] == 5
    assert parsed.rate_limits[0]["primary"]["used_percent"] == 7.0


def test_codex_rollout_merges_one_turn_into_one_assistant_message(monkeypatch) -> None:
    """A reasoning + message + tool-call turn is one response, not three."""

    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_codex_rollout(
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"text": "thinking"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "running the check"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call-1",
                    "arguments": '{"command": "ls"}',
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 40,
                            "output_tokens": 10,
                        }
                    },
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "file.txt",
                },
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="openai/gpt-5.6-sol",
        reasoning=True,
        initial_session=False,
    )

    assert [message.role for message in parsed.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assistant = parsed.messages[1]
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].id == "call-1"
    assert "running the check" in assistant.text
    assert len(parsed.usage) == 1
    assert parsed.unmetered_model_calls == 0
    assert parsed.messages[2].tool_call_id == "call-1"


def test_codex_rollout_counts_a_response_missing_its_token_count(monkeypatch) -> None:
    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_codex_rollout(
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "final answer"}],
                },
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="openai/gpt-5.6-sol",
        reasoning=True,
        initial_session=False,
    )

    assert [message.role for message in parsed.messages] == ["user", "assistant"]
    assert parsed.usage == []
    assert parsed.unmetered_model_calls == 1


def test_subscription_metadata_and_cost_are_explicitly_not_metered() -> None:
    metadata = subscription_harness_metadata(
        "opus-4.6", "anthropic/claude-opus-4-6"
    )
    assert metadata["mode"] == "subscription"
    assert metadata["agent_billing"] == "subscription_included_usage"
    assert metadata["per_run_agent_cost_available"] is False
    assert metadata["native_system_prompt_observability"]["complete"] is False
    assert metadata["network_policy"]["container"] == "internal_only"

    priced = estimate_usage_cost(
        "subscription/anthropic/claude-opus-4-6",
        {"input": 100, "output": 10},
    )
    assert priced == {
        "cost_usd": None,
        "exact": False,
        "source": "subscription_not_metered",
    }


def test_subscription_native_reported_version_must_match_pin() -> None:
    harness = {
        "scaffold": "codex",
        "scaffold_version_selector": "0.146.1",
    }
    subscription.record_subscription_native_version(harness, "0.146.1")
    assert harness["native_reported_scaffold_version_matches_pin"] is True

    try:
        subscription.record_subscription_native_version(harness, "0.999.0")
    except RuntimeError as error:
        assert "does not match pin" in str(error)
    else:
        raise AssertionError("a mismatched subscription CLI version was accepted")


def test_missing_native_version_is_flagged_not_silent() -> None:
    harness = {"scaffold": "codex", "scaffold_version_selector": "0.146.1"}
    subscription.record_subscription_native_version(harness, None)
    assert harness["native_reported_scaffold_version"] is None
    assert harness["native_reported_scaffold_version_matches_pin"] is None

    subscription.record_subscription_native_version(harness, "0.146.1")
    assert harness["native_reported_scaffold_version"] == "0.146.1"
    assert harness["native_reported_scaffold_version_matches_pin"] is True

    # A later version-less invocation (codex resume) must not clobber it.
    subscription.record_subscription_native_version(harness, None)
    assert harness["native_reported_scaffold_version"] == "0.146.1"
    assert harness["native_reported_scaffold_version_matches_pin"] is True


def test_subscription_agent_record_aggregates_dollars_and_unmetered_calls() -> None:
    session_ref = subscription.NativeSessionRef(value="session")
    session_ref.invocations = [
        {
            "usage": [{"input": 100, "output": 10, "cache_read": 0,
                       "cache_write": 0, "reasoning": 0, "total": 110}],
            "unmetered_model_calls": 0,
            "rate_limits": [{
                "source": "claude_code_result",
                "api_list_equivalent_usd": 0.10,
            }],
        },
        {
            "usage": [],
            "unmetered_model_calls": 1,
            "rate_limits": [{
                "source": "claude_code_result",
                "api_list_equivalent_usd": 0.05,
            }],
        },
    ]
    session_ref.last_invocation = session_ref.invocations[-1]

    record = subscription.subscription_agent_record(session_ref)

    assert record["api_list_equivalent_usd_per_invocation"] == [0.10, 0.05]
    assert abs(record["api_list_equivalent_usd_total"] - 0.15) < 1e-9
    assert record["unmetered_model_call_count"] == 1
    assert record["model_call_count"] == 1
    assert record["usage_totals"]["input"] == 100


def test_subscription_resume_requires_a_native_session_id() -> None:
    try:
        subscription.build_subscription_agent(
            "opus-4.6",
            "anthropic/claude-opus-4-6",
            reasoning=True,
            native_resume={"native_session_id": None},
        )
    except RuntimeError as error:
        assert "native_session_id" in str(error)
    else:
        raise AssertionError(
            "a subscription resume without a native session id was accepted"
        )


def test_recorded_subscription_usage_reaches_inspect_tallies_unpriced(
    monkeypatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_record(model, usage, role=None):
        recorded.update(model=model, usage=usage, role=role)

    monkeypatch.setattr(
        "inspect_ai.model._model.record_and_check_model_usage", fake_record
    )
    usage = subscription._usage_record(
        {"input_tokens": 100, "output_tokens": 20}, openai_total_input=False
    )

    subscription._record_subscription_usage("anthropic/claude-opus-4-6", usage)

    assert recorded["model"] == "subscription/anthropic/claude-opus-4-6"
    assert recorded["role"] == "target"
    assert recorded["usage"].input_tokens == 100
    assert recorded["usage"].output_tokens == 20
    assert recorded["usage"].total_cost is None

    from inspect_ai.model import get_model_info

    info = get_model_info("subscription/anthropic/claude-opus-4-6")
    assert info is not None and info.cost is None


def test_unreported_usage_is_never_recorded_to_inspect(monkeypatch) -> None:
    def fail_record(model, usage, role=None):
        raise AssertionError("zero usage must not be recorded")

    monkeypatch.setattr(
        "inspect_ai.model._model.record_and_check_model_usage", fail_record
    )
    usage = subscription._usage_record(None, openai_total_input=False)
    subscription._record_subscription_usage("anthropic/claude-opus-4-6", usage)


def test_claude_b64_env_seeds_credentials_into_the_sandbox(monkeypatch) -> None:
    credentials = b'{"claudeAiOauth":{"accessToken":"placeholder"}}'
    encoded = base64.b64encode(credentials).decode()
    written: dict[str, object] = {}

    async def fake_write_private_file(sbox, path, data):
        written.update(sbox=sbox, path=path, data=data)

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv(subscription.CLAUDE_AUTH_B64_ENV, encoded)
    monkeypatch.setattr(subscription, "_write_private_file", fake_write_private_file)
    marker = object()

    env, source = asyncio.run(subscription._seed_claude_auth(marker))

    assert env == {}
    assert source == subscription.CLAUDE_AUTH_B64_ENV
    assert written == {
        "sbox": marker,
        "path": "/root/.claude/.credentials.json",
        "data": credentials,
    }
    assert (
        subscription.subscription_auth_source("claude_code")
        == subscription.CLAUDE_AUTH_B64_ENV
    )


def test_subscription_proxy_only_allows_provider_domain_suffixes() -> None:
    proxy_path = ENVIRONMENTS / "sandbox" / "subscription_proxy" / "proxy.py"
    spec = importlib.util.spec_from_file_location("subscription_proxy", proxy_path)
    assert spec is not None and spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)

    assert proxy.allowed_host("api.anthropic.com")
    assert proxy.allowed_host("chatgpt.com")
    assert not proxy.allowed_host("anthropic.com.example.org")
    assert not proxy.allowed_host("example.org")


def test_subscription_auth_preflight_accepts_alternatives_and_fails_loudly(
    monkeypatch,
) -> None:
    available = {"claude_code": "macos_keychain", "codex": None}
    monkeypatch.setattr(
        subscription,
        "subscription_auth_source",
        lambda scaffold: available.get(scaffold, "api_fallback"),
    )
    assert subscription.require_subscription_auth(["claude_code"]) == {
        "claude_code": "macos_keychain"
    }
    try:
        subscription.require_subscription_auth(["codex"])
    except SystemExit as error:
        assert "CODEX_ACCESS_TOKEN" in str(error)
    else:
        raise AssertionError("missing Codex subscription login was accepted")


def test_gzip_base64_codex_auth_is_restored_only_inside_sandbox(monkeypatch) -> None:
    auth_json = b'{"auth_mode":"chatgpt","tokens":{"access_token":"placeholder"}}'
    encoded = base64.b64encode(gzip.compress(auth_json)).decode()
    written: dict[str, object] = {}

    async def fake_write_private_file(sbox, path, data):
        written.update(sbox=sbox, path=path, data=data)

    monkeypatch.setenv(subscription.CODEX_AUTH_GZIP_B64_ENV, encoded)
    monkeypatch.setattr(subscription, "_write_private_file", fake_write_private_file)
    marker = object()

    source = asyncio.run(subscription._seed_codex_auth(marker, "/usr/bin/codex"))

    assert source == subscription.CODEX_AUTH_GZIP_B64_ENV
    assert written == {
        "sbox": marker,
        "path": "/root/.codex/auth.json",
        "data": auth_json,
    }


def test_subscription_native_tools_cannot_read_cli_credentials() -> None:
    claude = subscription._claude_security_settings()
    assert claude["sandbox"]["enabled"] is True
    assert claude["sandbox"]["failIfUnavailable"] is True
    assert claude["sandbox"]["allowUnsandboxedCommands"] is False
    assert "/root/.claude" in claude["sandbox"]["filesystem"]["denyRead"]
    assert "Read(//root/.claude/**)" in claude["permissions"]["deny"]
    assert claude["sandbox"]["credentials"]["envVars"] == [
        {"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"}
    ]

    codex = subscription._codex_security_config(
        model="gpt-5.6-sol", reasoning=True
    )
    assert codex["default_permissions"] == "subscription-workspace"
    assert codex["permissions.subscription-workspace"]["extends"] == ":workspace"
    assert codex["shell_environment_policy"]["inherit"] == "core"
    assert codex["allow_login_shell"] is False
    assert codex["cli_auth_credentials_store"] == "file"
    assert codex["features.apps"] is False
    assert codex["features.multi_agent"] is False
    filesystem = tomllib.loads(subscription._codex_filesystem_deny_toml())
    rules = filesystem["permissions"]["subscription-workspace"]["filesystem"]
    assert rules["/root/.codex/auth.json"] == "deny"
    assert rules["/proc"] == "deny"

    source = inspect.getsource(subscription.build_subscription_agent)
    assert "dangerously-skip-permissions" not in source
    assert "dangerously-bypass-approvals-and-sandbox" not in source


def test_subscription_images_install_required_nested_sandbox_helpers() -> None:
    for relative in (
        "sandbox/ml/Dockerfile.subscription",
        "sandbox/p_hacking/Dockerfile.subscription",
    ):
        dockerfile = (ENVIRONMENTS / relative).read_text()
        assert "bubblewrap" in dockerfile
        assert "ripgrep" in dockerfile
        assert "socat" in dockerfile
    for relative in (
        "sandbox/ml/compose.subscription.yaml",
        "sandbox/p_hacking/compose.subscription.yaml",
    ):
        compose = (ENVIRONMENTS / relative).read_text()
        assert "seccomp:unconfined" in compose
        assert "mats-environments-" in compose
