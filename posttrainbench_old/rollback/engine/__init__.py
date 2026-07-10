"""Continuation engines. The faithful path runs the original agent CLI inside
the apptainer container: native_cli (Claude Code) or opencode_cli (OpenCode),
selected by the trajectory's scaffold. An api_replay engine — which drives the
model directly and intercepts tool calls — is the future home of the
LLM-as-environment variant (fake training/eval outputs); it shares the workspace
and session reconstruction and slots in behind the same interface.
"""
from __future__ import annotations

from .. import config


def for_trajectory(traj: config.Trajectory, **kwargs):
    """The native engine matching the trajectory's original scaffold."""
    if traj.scaffold == "opencode":
        from .opencode_cli import OpenCodeEngine
        return OpenCodeEngine(bash_mode=kwargs.get("bash_mode", "execute"),
                              container_sif=kwargs.get("container_sif"))
    if traj.scaffold == "claude":
        from .native_cli import NativeClaudeCodeEngine
        return NativeClaudeCodeEngine(
            bash_mode=kwargs.get("bash_mode", "execute"),
            resume_mode=kwargs.get("resume_mode"),
            container_sif=kwargs.get("container_sif"))
    if traj.scaffold == "codex":
        from .codex_cli import CodexEngine
        return CodexEngine(
            bash_mode=kwargs.get("bash_mode", "execute"),
            resume_mode=kwargs.get("resume_mode"),
            container_sif=kwargs.get("container_sif"))
    raise ValueError(f"no engine for scaffold {traj.scaffold!r}")
