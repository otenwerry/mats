"""End-to-end REAL-environment audit pipeline: audit -> annotate -> viewer, in one run.

The real-mode sibling of petri's exp_audit_pipeline.py: the same three stages, but
stage 1 runs targets inside real Docker sandboxes (environments/sandbox/<family>/) with
a deterministic scripted controller instead of an auditor LLM (see lib/exp_real_audit.py).
Stages 2-3 reuse the simulated pipeline's code verbatim (annotation + viewer are
mode-agnostic; results flow into the normal petri viewer).

v1 scope: fraud_detection + reasoning_prompt_benchmark, --condition=allow only.

Usage (from mats/environments/; re-execs into petri/.venv automatically):
  uv run exp_real_audit_pipeline.py --targets=qwen3-32b --seed-dir=ml_training_data_misuse \
      --seeds=fraud_detection --epochs=3
  uv run exp_real_audit_pipeline.py --targets=qwen3-32b --seed-dir=p_hacking \
      --seeds=reasoning_prompt_benchmark --epochs=3

Flags:
  --targets=<a,b,..>   REQUIRED. target names from TARGET_CHOICES.
  --seed-dir=<sub>     REQUIRED. a pinned seed dir or collection; every selected member
                       needs a generated real_env/workspace/ tree (envgen/gen_*.py).
  --seeds=<a,b,..>     REQUIRED. member names, or `all` (= every PORTED member).
  --epochs=<N>         REQUIRED. epochs per (target, seed) cell.
  --reasoning=yes|no   optional (default yes). Same semantics as the simulated pipeline.
  --condition=<c>      defaults to allow; v1 supports allow only (correct is a designed
                       seam, rejected loudly).
  --judge=<m>          alignment judge (default gpt-5.6-luna; see lib/judge_models.py,
                       or set $PETRI_JUDGE).
  --gate-model=<m>     first-submission gate LLM; defaults to the judge and must equal it.
  --concurrency=<N>    LLM parallelism for audit + annotate (default 50).
  --sandbox-concurrency=<N>  max live Docker containers (default 8).
  --time-limit=<sec>   per-trajectory wall clock, default 3600 (one hour). This is the
                       runaway guard: there is no turn cap. Pass 0 to disable.
  --annotate-model=<m> agentic hack-turn annotator (default deepseek-v4-pro).
  --skip-annotate / --skip-viewer / --force-annotate  as in exp_audit_pipeline.py.

Requires a running Docker daemon (checked before any spend).
Costs money (target + judge + gate LLM calls; annotation). The viewer stage is free.
"""

import asyncio
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
from datetime import datetime

_ENVIRONMENTS = pathlib.Path(__file__).resolve().parent
_MATS = _ENVIRONMENTS.parent
_PETRI = _MATS / "petri"


def ensure_petri_venv() -> None:
    """environments/ has no venv of its own; everything runs in petri/.venv (which has
    inspect_ai etc.). Re-exec under that interpreter so this endpoint works no matter
    which directory/venv it was launched from (same pattern as shared/exp_ask_questions)."""
    want = _PETRI / ".venv"
    if pathlib.Path(sys.prefix).resolve() == want.resolve():
        return
    py = want / "bin" / "python"
    if not py.exists():
        sys.exit(f"expected petri venv not found: {want} (run `uv sync` in petri/)")
    os.execv(str(py), [str(py), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    ensure_petri_venv()

for _p in (str(_ENVIRONMENTS / "lib"), str(_PETRI / "lib"), str(_PETRI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp_rh_audit import (
    JUDGE,
    REASONING_EFFORT,
    TARGET_CHOICES,
    reasoning_tag,
    reject_fixed_sp_flag,
    resolve_condition,
    resolve_reasoning,
    resolve_seeds,
    run_eval,
)
from exp_real_audit import (
    DEFAULT_SANDBOX_CONCURRENCY,
    build_real_tasks,
    ported_members,
    resolve_gate_model,
    resolve_time_limit,
)
# Petri's integrity guards are mode-agnostic and reused verbatim. Its
# run_post_audit_stages is NOT: it annotates petri's log root and builds petri's viewer,
# so this project runs its own annotate + viewer stages against its own data root.
from exp_audit_pipeline import (
    DEFAULT_ANNOTATE_MODEL,
    DEFAULT_CONCURRENCY,
    audit_integrity_failures,
    unrecovered_dead_targets,
)
from exp_annotate_real_hacks import annotate_real_hacks
from judge_models import resolve_judge
from env_paths import DATA, OUT
from model_routing import route


def env_viewer():
    """This project's viewer module, loaded BY PATH.

    Both projects have a top-level viewer.py, and petri/ is on sys.path (its lib is what
    everything here builds on), so a plain `import viewer` would resolve to petri's and
    silently build the wrong site into the wrong root.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "environments_viewer", _ENVIRONMENTS / "viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_VALUE_FLAGS = {
    "--targets", "--seed-dir", "--seeds", "--epochs", "--reasoning", "--condition",
    "--concurrency", "--sandbox-concurrency", "--time-limit",
    "--judge", "--gate-model", "--annotate-model",
}
_SWITCH_FLAGS = {"--skip-annotate", "--skip-viewer", "--force-annotate"}


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for arg in sys.argv[1:]:
        flag, separator, _ = arg.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(f"{flag} requires a value in the form {flag}=<value>")
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {arg!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _posint(flag: str, default: int | None) -> int | None:
    raw = _arg(flag)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise SystemExit(f"{flag} must be an integer, got {raw!r}")
    if v < 1:
        raise SystemExit(f"{flag} must be >= 1, got {v}")
    return v


def require_docker() -> None:
    """Abort BEFORE any spend if no Docker daemon is reachable (each trajectory needs
    its own container). This machine currently has no Docker installed -- install
    Docker Desktop or OrbStack first."""
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit(
            "docker is not installed (or not on PATH). Real-environment audits run "
            "every trajectory in a container; install Docker Desktop/OrbStack and retry."
        )
    probe = subprocess.run([docker, "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(
            "docker is installed but the daemon is not reachable "
            f"({(probe.stderr or probe.stdout).strip().splitlines()[:1]}). "
            "Start Docker and retry."
        )


def _parse_args() -> dict:
    reject_fixed_sp_flag()
    _validate_cli_args()

    targets_arg = _arg("--targets")
    if targets_arg is None:
        raise SystemExit(f"--targets is required (no default); choices: {sorted(TARGET_CHOICES)}")
    targets = list(dict.fromkeys(t.strip() for t in targets_arg.split(",") if t.strip()))
    unknown = [t for t in targets if t not in TARGET_CHOICES]
    if unknown:
        raise SystemExit(f"unknown --targets {unknown}; choices: {sorted(TARGET_CHOICES)}")

    seed_dir_arg = _arg("--seed-dir")
    if seed_dir_arg is None:
        raise SystemExit("--seed-dir is required (e.g. ml_training_data_misuse or p_hacking)")
    seeds_path, available_seeds = resolve_seeds(seed_dir_arg)
    ported = ported_members(seeds_path)
    seeds_arg = _arg("--seeds")
    if seeds_arg is None:
        raise SystemExit(
            "--seeds is required; use --seeds=all for every PORTED member, or a comma "
            f"list. Ported members here: {ported or 'none'}"
        )
    if seeds_arg.strip() == "all":
        seeds = list(ported)
        if not seeds:
            raise SystemExit(
                f"no member of {seed_dir_arg} has a real_env workspace yet; generate one "
                "with the matching envgen/gen_*.py script"
            )
    else:
        seeds = [s.strip() for s in seeds_arg.split(",") if s.strip()]
        unknown_s = [s for s in seeds if s not in available_seeds]
        if unknown_s:
            raise SystemExit(f"unknown --seeds {unknown_s}; available: {available_seeds}")

    epochs = _posint("--epochs", None)
    if epochs is None:
        raise SystemExit("--epochs is required (no default); positive integer")

    condition = resolve_condition(_arg("--condition"), seeds_path)

    return {
        "targets": targets,
        "seeds": seeds,
        "seeds_path": seeds_path,
        "epochs": epochs,
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "condition": condition,
        "judge": _arg("--judge"),
        "gate_model": resolve_gate_model(_arg("--gate-model"), _arg("--judge")),
        "concurrency": _posint("--concurrency", DEFAULT_CONCURRENCY),
        "sandbox_concurrency": _posint("--sandbox-concurrency", DEFAULT_SANDBOX_CONCURRENCY),
        "time_limit": resolve_time_limit(_arg("--time-limit")),
        "annotate_model": _arg("--annotate-model", DEFAULT_ANNOTATE_MODEL),
        "skip_annotate": "--skip-annotate" in sys.argv,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "force_annotate": "--force-annotate" in sys.argv,
    }


def gate_integrity_failures(logs: list) -> list[dict]:
    """Samples whose scripted branch was decided by a FAILED gate (verdict=error) --
    the run ended conservatively and must not be pooled as a normal trajectory."""
    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            real_env = (sample.metadata or {}).get("real_env") or {}
            protocol = real_env.get("protocol") or {}
            gate_errors = [
                g for g in real_env.get("gates", []) if g.get("verdict") == "error"
            ]
            if gate_errors or protocol.get("ended_reason") == "gate_error_end":
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": ["gate_error"]
                              + [str(g.get("error"))[:120] for g in gate_errors],
                })
    return failures


def run_real_audit_stage(cfg: dict):
    targets, seeds, epochs = cfg["targets"], cfg["seeds"], cfg["epochs"]
    target_models = [route(TARGET_CHOICES[t]) for t in targets]
    expected_n = len(targets) * len(seeds) * epochs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = targets[0] if len(targets) == 1 else f"{len(targets)}targets"
    log_dir = DATA / "logs" / f"real-v1-{label}-{cfg['condition']}-{epochs}ep-{timestamp}"

    print("=" * 72)
    print(f"STAGE 1/3  REAL AUDIT  ->  {log_dir.name}")
    print("=" * 72)
    print(f"  targets ({len(targets)}): "
          + ", ".join(f"{t}{reasoning_tag(cfg['reasoning'])}" for t in targets))
    print(f"  seeds ({len(seeds)}): {seeds}  condition={cfg['condition']}")
    print(f"  judge={resolve_judge(cfg['judge'])}  gate={cfg['gate_model']}")
    print(f"  epochs={epochs}  concurrency={cfg['concurrency']}  "
          f"sandbox_concurrency={cfg['sandbox_concurrency']}  "
          f"time_limit={cfg['time_limit']}s (no turn cap)")
    if cfg["reasoning"]:
        print(f"  reasoning=ON effort={REASONING_EFFORT}; '<thinking> tags' SP "
              "instruction stripped")
    print(f"  expected trajectories: {len(targets)} x {len(seeds)} x {epochs} = {expected_n}")
    print("  environment: REAL docker sandboxes; user turns fully scripted; the only "
          "in-run AI decision is the first-submission gate\n")

    try:
        tasks = build_real_tasks(
            targets, seeds, log_dir.name,
            reasoning=cfg["reasoning"], condition=cfg["condition"],
            gate_model=cfg["gate_model"], judge=cfg["judge"],
            seeds_path=cfg["seeds_path"],
            artifacts_root=log_dir / "real_artifacts",
        )
        success, logs = run_eval(
            tasks, epochs, cfg["concurrency"], log_dir,
            max_sandboxes=cfg["sandbox_concurrency"], time_limit=cfg["time_limit"],
        )
    except SystemExit:
        raise
    except Exception as e:
        print("\n!! REAL AUDIT STAGE CRASHED (continuing to annotate+viewer on existing logs):")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, log_dir, expected_n, False

    print(f"\neval_set finished, success={success}")
    actual_n = 0
    for log in logs:
        n = len(log.samples or [])
        actual_n += n
        print(f"  {log.eval.task}: status={log.status}, samples={n}")

    dead = unrecovered_dead_targets(logs, target_models)
    if dead:
        print("\n" + "!" * 72)
        print(f"WARNING: target(s) {dead} produced 0 output tokens -- they never ran.")
        print("!" * 72)

    if actual_n != expected_n:
        print(f"\nWARNING: expected {expected_n} trajectories but eval_set wrote {actual_n}.")
    else:
        print(f"\nall {expected_n} trajectories present.")

    integrity_failures = audit_integrity_failures(logs)
    gate_failures = gate_integrity_failures(logs)
    for name, failures in (("data-integrity", integrity_failures),
                           ("gate", gate_failures)):
        if failures:
            print("\n" + "!" * 72)
            print(f"WARNING: {len(failures)} trajectory(ies) have {name} failures:")
            for failure in failures:
                print(f"  {failure['task']}/{failure['sample']} epoch {failure['epoch']}: "
                      f"{', '.join(failure['issues'])}")
            print("!" * 72)

    integrity_ok = (bool(success) and actual_n == expected_n and not dead
                    and not integrity_failures and not gate_failures)
    return logs, log_dir, expected_n, integrity_ok


async def run_env_post_stages(cfg: dict) -> bool:
    """Stages 2 (annotate) and 3 (viewer) against THIS project's data root. Mirrors
    petri's run_post_audit_stages: each stage is wrapped so one failing doesn't stop the
    other, and the viewer always runs last so a failed annotation is still inspectable."""
    ok = True
    if not cfg["skip_annotate"]:
        print("\n" + "=" * 72)
        print("STAGE 2/3  ANNOTATE  (agentic hack-turn localization on the reward hacks)")
        print("=" * 72)
        try:
            stats = await annotate_real_hacks(
                model=cfg["annotate_model"],
                concurrency=cfg["concurrency"],
                force=cfg["force_annotate"],
            )
            if stats.get("failed"):
                ok = False
        except Exception as e:
            ok = False
            print(f"\n!! ANNOTATE STAGE FAILED (continuing to viewer): {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 2 annotate: --skip-annotate)")

    if not cfg["skip_viewer"]:
        print("\n" + "=" * 72)
        print("STAGE 3/3  VIEWER  (rebuild this project's static viewer, free)")
        print("=" * 72)
        try:
            await env_viewer().main()
        except Exception as e:
            ok = False
            print(f"\n!! VIEWER STAGE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 3 viewer: --skip-viewer)")
    return ok


def main() -> None:
    cfg = _parse_args()
    require_docker()
    _logs, _log_dir, _expected_n, audit_ok = run_real_audit_stage(cfg)
    post_ok = asyncio.run(run_env_post_stages(cfg))
    print("\n" + "=" * 72)
    if audit_ok and post_ok:
        print(f"PIPELINE DONE.  open {OUT / 'index.html'}")
    else:
        print("PIPELINE FINISHED WITH INTEGRITY FAILURES. "
              f"Inspect {OUT / 'index.html'} and the warnings above.")
    print("=" * 72)
    if not (audit_ok and post_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
