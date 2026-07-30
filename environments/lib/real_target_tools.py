"""Real executors for the fixed target-tool interface (real-environment mode).

The tool NAMES, DESCRIPTIONS, and JSON SCHEMAS are byte-identical to the simulated
audits' interface — `fixed_target_tools.FIXED_TARGET_TOOL_DEFINITIONS` stays the single
source of truth, and tests/test_real_target_tools.py asserts the fingerprint matches.
Only the bodies differ: instead of an auditor fabricating results, each call executes
against the sample's Docker sandbox `/workspace`.

Result semantics are pinned (REAL_TOOL_SEMANTICS_VERSION) because there is no auditor
to improvise them:
  bash        run via `bash -lc` in /workspace; returns stdout+stderr; a nonzero exit
              appends "(exit code N)"; timeouts raise ToolError.
  read_file   UTF-8 text read, relative paths resolved against /workspace.
  write_file  parent dirs created, whole-file UTF-8 write, returns "Wrote <path>".

Tool output shown to the target is capped at REAL_MAX_TOOL_OUTPUT bytes by
execute_tools (a visible truncation envelope in-transcript). This is a deliberate,
metadata-stamped difference from the simulated auditor channel, which was uncapped.
"""

from __future__ import annotations

import posixpath

from inspect_ai.tool import Tool, ToolDef, ToolError, ToolInfo, ToolParams
from inspect_ai.util import sandbox

from fixed_target_tools import FIXED_TARGET_TOOL_DEFINITIONS

REAL_TOOL_SEMANTICS_VERSION = "petri-real-tools-v1"
# Visible per-call cap on tool output shown to the target (bytes). Inspect's default.
REAL_MAX_TOOL_OUTPUT = 16384
# One command may be a real model training run; generous, but not unbounded.
REAL_BASH_TIMEOUT = 900

WORKSPACE_DIR = "/workspace"

_DEFS = {d["name"]: d for d in FIXED_TARGET_TOOL_DEFINITIONS}


def _resolve(path: str) -> str:
    """Workspace-relative path resolution (the tool descriptions promise it)."""
    return path if posixpath.isabs(path) else posixpath.join(WORKSPACE_DIR, path)


async def _bash(command: str) -> str:
    try:
        result = await sandbox().exec(
            ["bash", "-lc", command],
            cwd=WORKSPACE_DIR,
            timeout=REAL_BASH_TIMEOUT,
            timeout_retry=False,
        )
    except TimeoutError:
        raise ToolError(f"command timed out after {REAL_BASH_TIMEOUT}s")
    except UnicodeDecodeError:
        raise ToolError("command produced output that is not valid UTF-8")
    except PermissionError:
        raise ToolError("permission denied executing command")
    output = result.stdout
    if result.stderr:
        output = f"{output}{result.stderr}" if output else result.stderr
    if not result.success:
        output = f"{output}\n(exit code {result.returncode})" if output else (
            f"(exit code {result.returncode})"
        )
    return output


async def _read_file(path: str) -> str:
    resolved = _resolve(path)
    try:
        return await sandbox().read_file(resolved, text=True)
    except FileNotFoundError:
        raise ToolError(f"{path}: no such file")
    except IsADirectoryError:
        raise ToolError(f"{path} is a directory, not a file")
    except UnicodeDecodeError:
        raise ToolError(f"{path} is not valid UTF-8 text")
    except PermissionError:
        raise ToolError(f"{path}: permission denied")


async def _write_file(path: str, content: str) -> str:
    resolved = _resolve(path)
    parent = posixpath.dirname(resolved)
    if parent and parent != "/":
        await sandbox().exec(["mkdir", "-p", parent])
    try:
        await sandbox().write_file(resolved, content)
    except PermissionError:
        raise ToolError(f"{path}: permission denied")
    except IsADirectoryError:
        raise ToolError(f"{path} is a directory")
    return f"Wrote {path}"


_EXECUTORS = {"bash": _bash, "read_file": _read_file, "write_file": _write_file}


def real_tool_defs() -> list[ToolDef]:
    """One ToolDef per fixed definition, schema taken verbatim from the shared source."""
    return [
        ToolDef(
            _EXECUTORS[d["name"]],
            name=d["name"],
            description=d["description"],
            parameters=ToolParams.model_validate(d["parameters"]),
        )
        for d in FIXED_TARGET_TOOL_DEFINITIONS
    ]


def real_target_tools() -> list[Tool]:
    """The three real tools, ready for generate()/execute_tools."""
    return [td.as_tool() for td in real_tool_defs()]


def real_target_tools_infos() -> list[ToolInfo]:
    """ToolInfo view of the REAL tools (what the wire schema will be), for the
    fingerprint test and metadata stamping."""
    return [
        ToolInfo(name=td.name, description=td.description, parameters=td.parameters)
        for td in real_tool_defs()
    ]
