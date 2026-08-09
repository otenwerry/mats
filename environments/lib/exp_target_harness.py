"""Agent-harness selection and production-scaffold adapters.

This module can launch paid agent calls through Inspect agents, hence the ``exp_``
prefix. Importing it, validating a resume bundle, and constructing an agent are free;
calling the returned agent is not.

Simple mode remains implemented in :mod:`exp_real_audit`. Production mode uses the
official API-backed Inspect SWE bridges for Claude Code, Codex CLI, and OpenCode.
Subscription mode runs Claude Code and Codex through native subscription login while
retaining API-backed OpenCode. Native continuations restore the scaffold's own session
files into a fresh workspace; an Inspect transcript alone is not resumable state.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import re
import tarfile
import uuid
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Sequence
from unittest.mock import patch


HARNESS_CHOICES = ("simple", "production", "subscription")
NATIVE_HARNESS_MODES = frozenset({"production", "subscription"})
INSPECT_SWE_VERSION = "0.2.63"
PRODUCTION_SCAFFOLD_VERSIONS = {
    "claude_code": "2.1.220",
    "codex": "0.146.1",
    "opencode": "1.18.14",
}
NATIVE_RESUME_FORMAT = "environments-production-native-resume-v1"

_NATIVE_SESSION_PATHS = {
    "claude_code": (
        "root/.claude/projects",
        "root/.claude/tasks",
        "root/.claude/todos",
        "root/.claude/session-env",
        "root/.claude/file-history",
    ),
    "codex": ("root/.codex/sessions",),
    "opencode": (
        "root/.local/share/opencode",
        "root/.local/state/opencode",
    ),
}


def resolve_harness(value: str | None) -> str:
    """Resolve the required agent-harness CLI choice."""

    if value is None:
        raise SystemExit(
            "--harness is required (no default); choose --harness=simple, "
            "--harness=production, or --harness=subscription"
        )
    harness = value.strip().lower()
    if harness not in HARNESS_CHOICES:
        raise SystemExit(
            f"--harness must be one of {list(HARNESS_CHOICES)}, got {value!r}"
        )
    return harness


def production_scaffold_for_target(target_name: str, routed_slug: str) -> str:
    """Map agent identity to the production scaffold requested for this project."""

    routed = routed_slug.lower()
    name = target_name.lower()
    if routed.startswith("anthropic/claude-") or name.startswith(("opus-", "sonnet-")):
        return "claude_code"
    if routed.startswith("openai/gpt-") or name.startswith("gpt-"):
        return "codex"
    return "opencode"


_DATED_MODEL_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def codex_subscription_model(routed_slug: str) -> str:
    """The model name Codex accepts on a ChatGPT account.

    Subscription Codex rejects dated API snapshot names ("'gpt-5.5-2026-04-23'
    model is not supported when using Codex with a ChatGPT account"), so the
    undated family name is requested and the served snapshot is not pinnable.
    """

    return _DATED_MODEL_SUFFIX_RE.sub("", routed_slug.split("/", 1)[-1])


def _inspect_swe_version() -> str:
    try:
        return package_version("inspect-swe")
    except PackageNotFoundError:
        return "unavailable"


def assert_inspect_swe_pin() -> None:
    installed = _inspect_swe_version()
    if installed != INSPECT_SWE_VERSION:
        raise RuntimeError(
            "production harness requires the pinned inspect-swe version "
            f"{INSPECT_SWE_VERSION}, but {installed} is installed"
        )


def production_harness_metadata(target_name: str, routed_slug: str) -> dict:
    """Stable run metadata for the selected, exactly pinned production scaffold."""

    scaffold = production_scaffold_for_target(target_name, routed_slug)
    return {
        "mode": "production",
        "scaffold": scaffold,
        "scaffold_version_selector": PRODUCTION_SCAFFOLD_VERSIONS[scaffold],
        "scaffold_model_config": (
            routed_slug if scaffold == "opencode" else "derived_from_served_target"
        ),
        "inspect_swe_version": _inspect_swe_version(),
        "working_directory": "/workspace",
        "native_tools": True,
        "environment_system_prompt_applied": False,
        "network_policy": {
            "container": "none",
            "native_network_tools": "explicitly_disabled",
        },
        # This is monitoring coverage, not evidence that loss occurred. Confirmed
        # loss is recorded separately in context_compactions/native_loss_events.
        "native_loss_monitoring": {
            "complete": False,
            "limitation": (
                "The scaffold can change native behavior in ways that are not "
                "represented by an Inspect event. This flag does not imply that "
                "any data was lost in this run."
            ),
        },
        "context_compactions": [],
        "native_loss_events": [],
    }


def subscription_harness_metadata(target_name: str, routed_slug: str) -> dict:
    """Stable metadata for native subscription calls and OpenCode's API fallback."""

    metadata = production_harness_metadata(target_name, routed_slug)
    scaffold = metadata["scaffold"]
    metadata.update({
        "mode": "subscription",
        "agent_billing": (
            "api_fallback" if scaffold == "opencode" else "subscription_included_usage"
        ),
        "per_run_agent_cost_available": scaffold == "opencode",
        "subscription_usage_tracking": (
            "not_applicable_api_fallback"
            if scaffold == "opencode"
            else "native_cli_provider_tokens"
        ),
        "subscription_quota_tracking": (
            "native_rate_limit_snapshots_when_reported_otherwise_unavailable"
            if scaffold in {"codex", "claude_code"}
            else "not_applicable_api_fallback"
        ),
        "native_system_prompt_observability": {
            "complete": scaffold != "claude_code",
            "loss_reason": (
                "claude_code_subscription_cli_does_not_expose_complete_native_system_prompt"
                if scaffold == "claude_code" else None
            ),
        },
        **(
            {
                "subscription_model_requested": codex_subscription_model(routed_slug),
                "subscription_snapshot_pinning": (
                    "chatgpt_account_serves_undated_model_snapshot_not_pinnable"
                ),
            }
            if scaffold == "codex"
            else {}
        ),
        "credential_isolation": (
            {
                "credential_copy": "disposable_container_private_file_or_process_env",
                "native_file_tools": "credential_paths_denied",
                "native_shell": "provider_sandbox_denies_credential_paths",
                "subprocess_credentials": "scrubbed",
                "sandbox_failure_policy": "fail_closed",
            }
            if scaffold in {"claude_code", "codex"}
            else {"mode": "api_fallback_no_subscription_credential"}
        ),
        "network_policy": (
            {
                "container": "internal_only",
                "provider_egress": "allowlisted_connect_proxy",
                "outer_seccomp": "unconfined_for_nested_user_namespace",
                "allowed_domain_families": [
                    "anthropic.com", "claude.ai", "openai.com", "chatgpt.com"
                ],
                "native_network_tools": "explicitly_disabled",
            }
            if scaffold != "opencode"
            else metadata["network_policy"]
        ),
    })
    return metadata


def _new_claude_session_id(native_resume: dict | None) -> str:
    if native_resume is not None:
        session_id = native_resume.get("native_session_id")
        try:
            return str(uuid.UUID(str(session_id)))
        except (TypeError, ValueError, AttributeError) as error:
            raise SystemExit(
                "production Claude Code resume bundle has an invalid native_session_id"
            ) from error
    return str(uuid.uuid4())


def build_production_agent(
    target_name: str,
    routed_slug: str,
    *,
    time_tool: Any | None = None,
    native_resume: dict | None = None,
):
    """Construct one exactly pinned production scaffold agent for one sample.

    The Inspect SWE 0.2.63 Claude bridge allocates its resume UUID internally and
    exposes no parameter for it. For a native continuation, this function replaces
    that one synchronous UUID allocation with the UUID stored in our bundle. The
    patch is active only while the agent factory runs (there is no ``await`` inside
    the patch), and the exact Inspect SWE version is asserted above.
    """

    from inspect_ai.agent import BridgedToolsSpec
    from inspect_swe import claude_code, codex_cli, opencode

    assert_inspect_swe_pin()
    bridged_tools = (
        [BridgedToolsSpec(name="environment", tools=[time_tool])]
        if time_tool is not None
        else None
    )
    scaffold = production_scaffold_for_target(target_name, routed_slug)
    common = {
        "model": None,
        "cwd": "/workspace",
        "version": PRODUCTION_SCAFFOLD_VERSIONS[scaffold],
        "attempts": 1,
        "bridged_tools": bridged_tools,
    }
    if scaffold == "claude_code":
        session_id = _new_claude_session_id(native_resume)
        module = importlib.import_module(
            "inspect_swe._claude_code.claude_code"
        )
        with patch.object(module.uuid, "uuid4", return_value=uuid.UUID(session_id)):
            agent = claude_code(
                disallowed_tools=["WebSearch", "WebFetch"],
                **common,
            )
        return agent, session_id
    if scaffold == "codex":
        return codex_cli(
            web_search="disabled",
            home_dir="/root/.codex",
            **common,
        ), None

    # Inline config has higher precedence than the bridge-generated config file and
    # merges with it. Explicit denies remain denies even though Inspect SWE invokes
    # OpenCode with its non-interactive permission flag.
    opencode_config = json.dumps(
        {
            "permission": {"webfetch": "deny", "websearch": "deny"},
            "autoupdate": False,
            "share": "disabled",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    opencode_env = {"OPENCODE_CONFIG_CONTENT": opencode_config}
    if routed_slug.startswith("openrouter/"):
        opencode_env["OPENROUTER_API_KEY"] = "sk-none"
    return (
        opencode(opencode_model=routed_slug, env=opencode_env, **common),
        None,
    )


def production_agent_input_messages(messages: Sequence[Any]) -> list[Any]:
    """Avoid re-injecting a native scaffold system prompt on native resume.

    Inspect SWE uses the presence of an assistant message to select the CLI's native
    resume command. Codex and OpenCode also reinterpret inbound Inspect system
    messages as new AGENTS/user instructions, even though the native session already
    owns those messages. Once a session exists, retain every non-system record for
    resume detection and the new user prompt, while letting the CLI reload its own
    system state from disk.
    """

    if any(getattr(message, "role", None) == "assistant" for message in messages):
        return [
            message for message in messages
            if getattr(message, "role", None) != "system"
        ]
    return list(messages)


def _validate_archive_members(data: bytes, scaffold: str) -> list[str]:
    allowed = _NATIVE_SESSION_PATHS[scaffold]
    names: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = member.name.removeprefix("./").rstrip("/")
                if (
                    not name
                    or name.startswith("/")
                    or ".." in name.split("/")
                    or member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                    or not any(name == root or name.startswith(root + "/") for root in allowed)
                ):
                    raise SystemExit(
                        f"native resume bundle has unsafe or unexpected member {member.name!r}"
                    )
                names.append(name)
    except (tarfile.TarError, OSError) as error:
        raise SystemExit(f"native resume bundle is not a valid tar.gz archive: {error}") from error
    if not names:
        raise SystemExit("native resume bundle archive is empty")
    return names


def validate_native_resume_bundle(
    bundle: dict,
    *,
    target_name: str,
    routed_slug: str,
    reasoning: bool,
) -> bytes:
    """Validate compatibility and integrity before any paid continuation call."""

    if not isinstance(bundle, dict) or bundle.get("format") != NATIVE_RESUME_FORMAT:
        raise SystemExit(
            f"production continuation requires a {NATIVE_RESUME_FORMAT!r} native_resume bundle"
        )
    scaffold = production_scaffold_for_target(target_name, routed_slug)
    expected = {
        "scaffold": scaffold,
        "scaffold_version": PRODUCTION_SCAFFOLD_VERSIONS[scaffold],
        "inspect_swe_version": INSPECT_SWE_VERSION,
        "target_name": target_name,
        "routed_slug": routed_slug,
        "reasoning": reasoning,
    }
    mismatches = {
        key: {"bundle": bundle.get(key), "required": value}
        for key, value in expected.items()
        if bundle.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            "production native resume bundle is incompatible with this continuation: "
            + json.dumps(mismatches, sort_keys=True)
        )
    encoded = bundle.get("archive_base64")
    if not isinstance(encoded, str):
        raise SystemExit("production native resume bundle has no archive_base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise SystemExit("production native resume archive_base64 is invalid") from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != bundle.get("archive_sha256") or len(data) != bundle.get("archive_bytes"):
        raise SystemExit("production native resume archive checksum or byte count does not match")
    names = _validate_archive_members(data, scaffold)
    declared_members = bundle.get("archive_members")
    if declared_members != names:
        raise SystemExit("production native resume archive member manifest does not match")
    if scaffold == "claude_code":
        _new_claude_session_id(bundle)
    return data


async def capture_native_resume_bundle(
    *,
    target_name: str,
    routed_slug: str,
    reasoning: bool,
    native_session_id: str | None,
) -> dict:
    """Capture the production scaffold's native conversational state."""

    from inspect_ai.util import sandbox

    scaffold = production_scaffold_for_target(target_name, routed_slug)
    sbox = sandbox()
    available: list[str] = []
    for relative in _NATIVE_SESSION_PATHS[scaffold]:
        result = await sbox.exec(["test", "-e", f"/{relative}"])
        if result.success:
            available.append(relative)
    if not available:
        raise RuntimeError(
            f"{scaffold} produced no native session files at the pinned paths"
        )
    archive_path = f"/tmp/environments-native-resume-{uuid.uuid4().hex}.tar.gz"
    result = await sbox.exec(["tar", "-czf", archive_path, "-C", "/", *available])
    if not result.success:
        raise RuntimeError(
            f"failed to archive {scaffold} native state: {result.stderr.strip()}"
        )
    data = await sbox.read_file(archive_path, text=False)
    members = _validate_archive_members(data, scaffold)
    bundle = {
        "format": NATIVE_RESUME_FORMAT,
        "scaffold": scaffold,
        "scaffold_version": PRODUCTION_SCAFFOLD_VERSIONS[scaffold],
        "inspect_swe_version": _inspect_swe_version(),
        "target_name": target_name,
        "routed_slug": routed_slug,
        "reasoning": reasoning,
        "working_directory": "/workspace",
        "workspace_restored": False,
        "native_session_id": native_session_id,
        "archive_format": "tar.gz",
        "archive_sha256": hashlib.sha256(data).hexdigest(),
        "archive_bytes": len(data),
        "archive_members": members,
        "archive_base64": base64.b64encode(data).decode("ascii"),
    }
    validate_native_resume_bundle(
        bundle,
        target_name=target_name,
        routed_slug=routed_slug,
        reasoning=reasoning,
    )
    return bundle


async def restore_native_resume_bundle(
    bundle: dict,
    *,
    target_name: str,
    routed_slug: str,
    reasoning: bool,
) -> dict:
    """Restore native state into the fresh sandbox, never the old workspace."""

    from inspect_ai.util import sandbox

    data = validate_native_resume_bundle(
        bundle,
        target_name=target_name,
        routed_slug=routed_slug,
        reasoning=reasoning,
    )
    archive_path = f"/tmp/environments-native-restore-{uuid.uuid4().hex}.tar.gz"
    sbox = sandbox()
    await sbox.write_file(archive_path, data)
    result = await sbox.exec(["tar", "-xzf", archive_path, "-C", "/"])
    if not result.success:
        raise RuntimeError(
            "failed to restore production native session: " + result.stderr.strip()
        )
    return {
        "format": bundle["format"],
        "archive_sha256": bundle["archive_sha256"],
        "archive_bytes": bundle["archive_bytes"],
        "scaffold": bundle["scaffold"],
        "scaffold_version": bundle["scaffold_version"],
        "inspect_swe_version": bundle["inspect_swe_version"],
        "workspace_restored": False,
        "restored": True,
    }


def native_resume_record(bundle: dict) -> dict:
    """A small queryable record suitable for logs (the archive stays in a sidecar)."""

    return {
        key: value
        for key, value in bundle.items()
        if key not in {"archive_base64", "archive_members"}
    } | {"archive_member_count": len(bundle.get("archive_members") or [])}


async def detect_opencode_native_loss_events() -> list[dict]:
    """Read confirmed OpenCode compactions/pruning from its native SQLite state."""

    from inspect_ai.util import sandbox

    code = r'''
import json, pathlib, sqlite3
p = pathlib.Path('/root/.local/share/opencode/opencode.db')
if not p.is_file():
    print('[]')
    raise SystemExit
db = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
messages = {row[0]: json.loads(row[1]) for row in db.execute('select id, data from message')}
parts = [(row[0], row[1], json.loads(row[2])) for row in db.execute('select id, message_id, data from part')]
compaction_users = {message_id for _, message_id, data in parts if data.get('type') == 'compaction'}
events = []
for message_id, data in messages.items():
    if (data.get('role') == 'assistant' and data.get('summary') and data.get('finish')
            and not data.get('error') and data.get('parentID') in compaction_users):
        events.append({'kind': 'context_compaction', 'event_uuid': message_id,
                       'source': 'opencode', 'session_id': data.get('sessionID')})
for part_id, _, data in parts:
    compacted = (((data.get('state') or {}).get('time') or {}).get('compacted'))
    if data.get('type') == 'tool' and compacted is not None:
        events.append({'kind': 'tool_output_pruned', 'event_uuid': part_id,
                       'source': 'opencode', 'compacted_at': compacted})
print(json.dumps(events, sort_keys=True))
'''
    result = await sandbox().exec(["python", "-c", code])
    if not result.success:
        return []
    try:
        value = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


async def update_scaffold_loss_events(real_env: dict, events: Sequence[Any]) -> None:
    """Stamp only confirmed native compaction/shortening events into output."""

    harness = real_env.get("harness") or {}
    known = {
        str(item.get("event_uuid"))
        for item in harness.get("native_loss_events") or []
        if isinstance(item, dict)
    }
    for event in events:
        if getattr(event, "event", None) != "compaction":
            continue
        source = str(getattr(event, "source", None) or "")
        if source not in {"claude_code", "codex_cli"}:
            continue
        event_uuid = str(getattr(event, "uuid", None) or "")
        if event_uuid in known:
            continue
        record = {
            "kind": "context_compaction",
            "event_uuid": event_uuid or None,
            "source": source,
            "type": getattr(event, "type", None),
            "tokens_before": getattr(event, "tokens_before", None),
            "tokens_after": getattr(event, "tokens_after", None),
        }
        harness.setdefault("context_compactions", []).append(record)
        harness.setdefault("native_loss_events", []).append(record)
        known.add(event_uuid)

    if harness.get("scaffold") == "opencode":
        for record in await detect_opencode_native_loss_events():
            event_uuid = str(record.get("event_uuid") or "")
            if event_uuid in known:
                continue
            harness.setdefault("native_loss_events", []).append(record)
            if record.get("kind") == "context_compaction":
                harness.setdefault("context_compactions", []).append(record)
            known.add(event_uuid)
    real_env["harness"] = harness


async def update_resolved_scaffold_version(real_env: dict) -> None:
    """Record and enforce the concrete CLI version installed in this sample."""

    harness = real_env.get("harness") or {}
    if (
        harness.get("mode") not in NATIVE_HARNESS_MODES
        or harness.get("scaffold_version_resolved")
    ):
        return
    from inspect_ai.util import sandbox
    from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR

    scaffold = harness.get("scaffold")
    try:
        if scaffold == "opencode":
            package_json = (
                f"{SANDBOX_INSTALL_DIR}/opencode/node_modules/opencode-ai/package.json"
            )
            package = json.loads(await sandbox().read_file(package_json, text=True))
            resolved = package.get("version")
        else:
            binary = "claude" if scaffold == "claude_code" else "codex"
            result = await sandbox().exec(
                ["find", SANDBOX_INSTALL_DIR, "-maxdepth", "1", "-type", "f", "-name", f"{binary}-*", "-print"]
            )
            versions = []
            for path in result.stdout.splitlines():
                match = re.search(
                    rf"/{binary}-(.+)-linux-(?:x64|x64-musl|arm64|arm64-musl)$",
                    path.strip(),
                )
                if match:
                    versions.append(match.group(1))
            resolved = versions[-1] if versions else None
    except Exception:
        resolved = None
    expected = PRODUCTION_SCAFFOLD_VERSIONS.get(str(scaffold))
    harness["scaffold_version_resolved"] = resolved
    harness["scaffold_version_resolution_failed"] = resolved is None
    harness["scaffold_version_matches_pin"] = resolved == expected
    real_env["harness"] = harness
    if resolved is not None and resolved != expected:
        raise RuntimeError(
            f"installed {scaffold} version {resolved} does not match pin {expected}"
        )
