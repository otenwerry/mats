"""Engine interface: given a fully-prepared job home (task/ workspace + .claude/
session + timer), continue the trajectory to termination and capture the new
trace. Concrete engines: native_cli (real Claude Code in apptainer) and, later,
api_replay (direct model loop with optional LLM-as-environment tool faking).
"""
from __future__ import annotations

import abc
from pathlib import Path

from .. import config


class Engine(abc.ABC):
    @abc.abstractmethod
    def prepare(self, spec: config.ExperimentSpec, job_home: Path,
                elapsed_seconds: int) -> dict:
        """Lay down task/, the reconstructed session, timer, and solve script
        into job_home. Returns a manifest. No model calls yet (cheap/offline)."""

    @abc.abstractmethod
    def launch_command(self, spec: config.ExperimentSpec, job_home: Path) -> list[str]:
        """The shell command that runs the continuation (e.g. apptainer exec ...).
        Returned (not executed) so the orchestrator can dry-run or schedule it."""
