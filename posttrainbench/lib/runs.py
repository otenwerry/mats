"""Registry of PTB reward-hack trajectories and their on-disk locations.

A trajectory's run_id is "<run_dir>__<task_dir>", but run_dir names can
themselves contain "__" (e.g. claude_non_api_claude-opus-4-6_1m__10h_run1),
so run_id is resolved against the actual directories on disk, never by
naive splitting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

def _find_root() -> Path:
    """Nearest ancestor holding the sibling `mats-local` data tree.

    Robust to being run from a git worktree (`.claude/worktrees/<name>/...`),
    where the old fixed `parents[3]` pointed at the worktrees dir instead of
    `.../supermats`."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "mats-local").is_dir():
            return p
    return here.parents[3]                          # legacy fallback


ROOT = _find_root()                                 # .../supermats
RAW = ROOT / "mats-local" / "posttrainbench"        # read-only HF mirror
VIEWER = RAW / "viewer_data"
HIGHLIGHTS = Path(__file__).resolve().parents[1] / "highlights"  # in-repo, beside this package
PTB_REPO = ROOT / "references" / "PostTrainBench"   # read-only harness clone
OUT_ROOT = ROOT / "mats-local" / "posttrainbench_outputs"  # our outputs (probes/reconstructions)

# Context-reconstruction campaign policy shared by the batch runner and viewer.
# The original systematic campaign is retained as an early-test source; the
# fresh campaign writes to a new directory so same-event reruns cannot overwrite
# those old outputs.
EARLY_PROBE_CAMPAIGNS = ("probes", "probes_context_recon")
CONTEXT_RECON_CAMPAIGN = "probes_context_recon_current"

# Keep individually deferred trajectories here so they cannot accidentally be
# re-run while still appearing in the viewer's deferred section.
CONTEXT_RECON_DEFERRED_RUN_IDS = frozenset({
    "claude_non_api_claude-opus-4-6_1m__10h_run1__bfcl_google_gemma-3-4b-pt_16955282",
})


def _scaffold(run_dir_name: str) -> str:
    for prefix, name in (("claude", "claude"), ("codex", "codex"),
                         ("opencode", "opencode"), ("qwen3max", "qwen3max")):
        if run_dir_name.startswith(prefix):
            return name
    raise ValueError(f"unknown scaffold for run dir {run_dir_name}")


def _agent(run_dir_name: str) -> str:
    """The PTB agents/<agent>/ dir this run used (longest matching prefix)."""
    agents = sorted((p.name for p in (PTB_REPO / "agents").iterdir() if p.is_dir()),
                    key=len, reverse=True)
    for a in agents:
        if run_dir_name.startswith(a + "_"):
            return a
    raise ValueError(f"no agents/ dir matches run dir {run_dir_name}")


@dataclass
class Trajectory:
    run_id: str
    run_dir: Path
    task_dir: Path
    scaffold: str            # claude | opencode | codex | qwen3max
    agent: str               # PTB agents/<agent> name, e.g. claude_non_api
    benchmark_id: str
    model_to_train: str      # org/name
    num_hours: int
    is_reward_hack: bool
    first_hack_event: int | None
    marked_turns: list[int] = field(default_factory=list)

    @property
    def traj_file(self) -> Path:
        for name in ("solve_out.txt", "trace.txt"):
            p = self.task_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(f"no trajectory file in {self.task_dir}")

    @property
    def viewer_path(self) -> Path:
        return VIEWER / f"{self.run_id}.json"

    def viewer_events(self) -> list[dict]:
        return json.loads(self.viewer_path.read_text())["events"]


def _split_run_id(run_id: str) -> tuple[Path, Path]:
    candidates = [d for d in RAW.iterdir()
                  if d.is_dir() and run_id.startswith(d.name + "__")]
    if not candidates:
        raise FileNotFoundError(f"no run dir on disk matches run_id {run_id}")
    run_dir = max(candidates, key=lambda d: len(d.name))
    task_dir = run_dir / run_id[len(run_dir.name) + 2:]
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task dir missing: {task_dir}")
    return run_dir, task_dir


def _task_meta(task_dir_name: str) -> tuple[str, str]:
    parts = task_dir_name.split("_")
    benchmark, middle = parts[0], parts[1:-1]
    if len(middle) != 2:
        raise ValueError(f"cannot parse model from task dir {task_dir_name}")
    return benchmark, f"{middle[0]}/{middle[1]}"


def load(run_id: str) -> Trajectory:
    run_dir, task_dir = _split_run_id(run_id)
    benchmark, model = _task_meta(task_dir.name)
    hl_path = HIGHLIGHTS / f"{run_id}.json"
    final = {}
    marked: list[int] = []
    if hl_path.exists():
        hl = json.loads(hl_path.read_text())
        final = hl.get("final") or {}
        marked = hl.get("marked_turns") or []
    num_hours = 10 if "_10h" in run_dir.name else 10
    return Trajectory(
        run_id=run_id, run_dir=run_dir, task_dir=task_dir,
        scaffold=_scaffold(run_dir.name), agent=_agent(run_dir.name),
        benchmark_id=benchmark, model_to_train=model, num_hours=num_hours,
        is_reward_hack=bool(final.get("is_reward_hack")),
        first_hack_event=final.get("first_hack_event"), marked_turns=marked,
    )


OPENCODE_BIN = (ROOT / "mats-local" / "tools" / "opencode-1.1.59" /
                "node_modules" / ".bin" / "opencode")


def opencode_cmd() -> str | None:
    """Path to the opencode CLI: the pinned run-era install (opencode-ai@1.1.59,
    the version in PTB's containers) if present, else whatever is on PATH."""
    import shutil
    if OPENCODE_BIN.exists():
        return str(OPENCODE_BIN)
    return shutil.which("opencode")


def env_has(var: str) -> bool:
    """Is the env var available to experiment scripts — set in the process env
    or defined in mats/.env (which the exp_ scripts load via dotenv)?"""
    import os
    if os.environ.get(var):
        return True
    dotenv = HIGHLIGHTS.parents[1] / ".env"
    try:
        return any(line.startswith(f"{var}=") and len(line.strip()) > len(var) + 1
                   for line in dotenv.read_text().splitlines())
    except OSError:
        return False


def reward_hacks() -> list[Trajectory]:
    """All confirmed reward-hack trajectories, from the highlights labels."""
    out = []
    for p in sorted(HIGHLIGHTS.glob("*__*.json")):
        rid = p.stem
        if rid.startswith("rollback"):
            continue
        try:
            hl = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not (hl.get("final") or {}).get("is_reward_hack"):
            continue
        out.append(load(rid))
    return out
