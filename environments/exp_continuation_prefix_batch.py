"""Run the full purpose-built-prefix continuation matrix as one unattended batch.

This is a paid orchestration endpoint. It selects the newest valid Natural Questions
(2k-token target, seed 1234), science-ethics, general-ethics, and move-fast payload for
each of the five core agents under both of that agent's required harness routes. It
then runs three AWS campaigns sequentially:

1. all simple-harness continuations;
2. production-harness continuations for the three open models;
3. subscription-harness continuations for Opus and GPT-5.5.

The first two stages share ``--vm-concurrency``. Subscription has an independent
``--subscription-vm-concurrency`` override, with the same 75-VM default, and runs last.
Each campaign is imported before the next begins. Infrastructure failures are retried
once by default; ordinary trajectory failures are recorded and do not block later
stages. The viewer is rebuilt once after every stage has finished.

Usage (from mats/environments/):
  uv run exp_continuation_prefix_batch.py
  uv run exp_continuation_prefix_batch.py --native-only
  uv run exp_continuation_prefix_batch.py --dry-run
  uv run exp_continuation_prefix_batch.py --resume-batch=<batch-id-or-path>

The default is 1,600 paid/included-quota trajectories: 4 prefixes x 5 agents x 2
harness routes x 40 epochs. ``--native-only`` skips the already-completed simple
stage and runs production followed by subscription (800 trajectories at the default
epochs). Batch state is stored under
``mats-local/environments/continuation_batches/``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ENVIRONMENTS = Path(__file__).resolve().parent
LIB = ENVIRONMENTS / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import exp_real_audit_pipeline as base_pipeline  # noqa: E402
from exp_aws_trajectory import (  # noqa: E402
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_REGION,
    DEFAULT_VM_CONCURRENCY,
    AwsTrajectoryError,
    resume_campaign,
    retry_failed,
    run_campaign,
    sha256_file,
)
from exp_real_audit import resolve_gate_model  # noqa: E402
from exp_real_continuation import load_prefix_specs  # noqa: E402
from judge_selection import resolve_judge  # noqa: E402
from model_catalog import resolve_target  # noqa: E402
from pipeline_cli import positive_int_arg, validate_cli_args, value_arg  # noqa: E402
from project_paths import (  # noqa: E402
    CONTINUATION_PREFIXES_ROOT,
    DATA_ROOT,
    VIEWER_ROOT,
)


PREFIX_TYPES = (
    "natural_questions",
    "science_ethics",
    "general_ethics",
    "move_fast",
)
MODEL_HARNESSES = (
    ("opus-4.6", "subscription"),
    ("gpt-5.5", "subscription"),
    ("deepseek-v4-pro", "production"),
    ("glm-5.1", "production"),
    ("kimi-k2.6", "production"),
)
DEFAULT_EPOCHS = 40
DEFAULT_SUBSCRIPTION_VM_CONCURRENCY = DEFAULT_VM_CONCURRENCY
DEFAULT_INFRASTRUCTURE_RETRIES = 1
DEFAULT_TREATMENT = "purpose-built-prefix"
STAGE_ORDER = ("simple", "production", "subscription")
SEEDS_PATH = ENVIRONMENTS / "seeds" / "p_hacking"
BATCH_ROOT = DATA_ROOT / "continuation_batches"

_VALUE_FLAGS = {
    "--epochs",
    "--vm-concurrency",
    "--subscription-vm-concurrency",
    "--infrastructure-retries",
    "--treatment",
    "--aws-region",
    "--aws-instance-type",
    "--aws-bucket",
    "--aws-secret-env",
    "--resume-batch",
}
_SWITCH_FLAGS = {"--skip-viewer", "--dry-run", "--native-only"}


@dataclass(frozen=True)
class Stage:
    key: str
    harness: str
    prefix_files: tuple[Path, ...]
    vm_concurrency: int
    epochs: int

    @property
    def trajectories(self) -> int:
        return len(self.prefix_files) * self.epochs


def _arg(flag: str, default: str | None = None) -> str | None:
    return value_arg(sys.argv, flag, default)


def _posint(flag: str, default: int) -> int:
    value = positive_int_arg(sys.argv, flag, default)
    assert value is not None
    return value


def parse_args() -> dict:
    validate_cli_args(
        sys.argv,
        value_flags=_VALUE_FLAGS,
        switch_flags=_SWITCH_FLAGS,
    )
    treatment = str(_arg("--treatment", DEFAULT_TREATMENT))
    if not treatment or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in treatment
    ):
        raise SystemExit(
            "--treatment must contain only lowercase letters, digits, and hyphens"
        )
    aws_region_arg = _arg("--aws-region")
    aws_instance_type_arg = _arg("--aws-instance-type")
    aws_bucket_arg = _arg("--aws-bucket")
    return {
        "epochs": _posint("--epochs", DEFAULT_EPOCHS),
        "vm_concurrency": _posint(
            "--vm-concurrency", DEFAULT_VM_CONCURRENCY
        ),
        "subscription_vm_concurrency": _posint(
            "--subscription-vm-concurrency",
            DEFAULT_SUBSCRIPTION_VM_CONCURRENCY,
        ),
        "infrastructure_retries": _posint(
            "--infrastructure-retries", DEFAULT_INFRASTRUCTURE_RETRIES
        ),
        "treatment": treatment,
        "aws_region": str(aws_region_arg or DEFAULT_REGION),
        "aws_region_explicit": aws_region_arg is not None,
        "aws_instance_type": str(
            aws_instance_type_arg or DEFAULT_INSTANCE_TYPE
        ),
        "aws_instance_type_explicit": aws_instance_type_arg is not None,
        "aws_bucket": aws_bucket_arg,
        "aws_bucket_explicit": aws_bucket_arg is not None,
        "aws_secret_env": [
            name.strip()
            for name in str(_arg("--aws-secret-env", "") or "").split(",")
            if name.strip()
        ],
        "resume_batch": _arg("--resume-batch"),
        "native_only": "--native-only" in sys.argv,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "dry_run": "--dry-run" in sys.argv,
    }


def _prefix_type(payload: dict) -> str | None:
    source = payload.get("source") or {}
    explicit = source.get("prefix_type")
    if explicit in PREFIX_TYPES:
        return str(explicit)
    if source.get("generator") == "exp_nq_prefix.py":
        if (
            source.get("target_context_tokens") == 2_000
            and source.get("rng_seed") == 1234
            and source.get("reached_target_tokens") is True
        ):
            return "natural_questions"
    return None


def discover_prefixes(root: Path = CONTINUATION_PREFIXES_ROOT) -> dict:
    """Select the newest payload for every exact type/model/harness key."""

    candidates: dict[tuple[str, str, str], tuple[str, Path, dict]] = {}
    errors: list[str] = []
    if not root.is_dir():
        raise SystemExit(f"continuation prefix directory does not exist: {root}")
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: {error}")
            continue
        prefix_type = _prefix_type(payload)
        target = payload.get("target")
        harness = (payload.get("source") or {}).get("harness") or "simple"
        if (
            prefix_type not in PREFIX_TYPES
            or target not in dict(MODEL_HARNESSES)
            or harness not in {"simple", dict(MODEL_HARNESSES)[target]}
            or payload.get("reasoning") is not True
        ):
            continue
        generated = str((payload.get("source") or {}).get("generated_at") or "")
        key = (prefix_type, str(target), str(harness))
        ordering = (generated, path.name)
        prior = candidates.get(key)
        if prior is None or ordering > (prior[0], prior[1].name):
            candidates[key] = (generated, path, payload)

    required = {
        (prefix_type, model, harness)
        for prefix_type in PREFIX_TYPES
        for model, native_harness in MODEL_HARNESSES
        for harness in ("simple", native_harness)
    }
    missing = sorted(required - set(candidates))
    if missing:
        rendered = ", ".join("/".join(key) for key in missing)
        detail = f"; unreadable files: {errors}" if errors else ""
        raise SystemExit(f"missing required purpose-built prefixes: {rendered}{detail}")
    return {
        "files": {
            "/".join(key): str(candidates[key][1])
            for key in sorted(required)
        },
        "selected_at": _now(),
    }


def build_stages(cfg: dict, selection: dict) -> list[Stage]:
    files = selection["files"]

    def selected(harness: str, models: tuple[str, ...]) -> tuple[Path, ...]:
        return tuple(
            Path(files[f"{prefix_type}/{model}/{harness}"])
            for model in models
            for prefix_type in PREFIX_TYPES
        )

    simple_models = tuple(model for model, _harness in MODEL_HARNESSES)
    production_models = tuple(
        model for model, harness in MODEL_HARNESSES if harness == "production"
    )
    subscription_models = tuple(
        model for model, harness in MODEL_HARNESSES if harness == "subscription"
    )
    stages = [
        Stage(
            key="simple",
            harness="simple",
            prefix_files=selected("simple", simple_models),
            vm_concurrency=cfg["vm_concurrency"],
            epochs=cfg["epochs"],
        ),
        Stage(
            key="production",
            harness="production",
            prefix_files=selected("production", production_models),
            vm_concurrency=cfg["vm_concurrency"],
            epochs=cfg["epochs"],
        ),
        Stage(
            key="subscription",
            harness="subscription",
            prefix_files=selected("subscription", subscription_models),
            vm_concurrency=cfg["subscription_vm_concurrency"],
            epochs=cfg["epochs"],
        ),
    ]
    if cfg.get("native_only"):
        return [stage for stage in stages if stage.key != "simple"]
    return stages


def validate_stages(stages: list[Stage]) -> dict[str, list]:
    """Run every local prefix/native-resume check before an AWS call or spend."""

    specs_by_stage = {}
    seen_paths: set[Path] = set()
    for stage in stages:
        if len(stage.prefix_files) != len(set(stage.prefix_files)):
            raise SystemExit(f"stage {stage.key} contains duplicate prefix files")
        specs = load_prefix_specs(
            [], [str(path) for path in stage.prefix_files], harness=stage.harness
        )
        if len(specs) != len(stage.prefix_files):
            raise SystemExit(f"stage {stage.key} did not load every selected prefix")
        specs_by_stage[stage.key] = specs
        seen_paths.update(stage.prefix_files)
    expected_paths = sum(len(stage.prefix_files) for stage in stages)
    if len(seen_paths) != expected_paths:
        raise SystemExit(
            f"expected {expected_paths} distinct prefix payloads, "
            f"found {len(seen_paths)}"
        )
    return specs_by_stage


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _batch_id() -> str:
    return f"continuation-prefix-batch-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def _manifest_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        candidate = BATCH_ROOT / candidate
    if candidate.is_dir():
        candidate = candidate / "batch_manifest.json"
    if not candidate.is_file():
        raise SystemExit(f"continuation batch manifest does not exist: {candidate}")
    return candidate.resolve()


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _new_manifest(
    batch_id: str,
    batch_dir: Path,
    cfg: dict,
    selection: dict,
    stages: list[Stage],
) -> dict:
    return {
        "format": "environments-continuation-prefix-batch-v1",
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "created_at": _now(),
        "completed_at": None,
        "configuration": {
            key: value for key, value in cfg.items()
            if key not in {"dry_run", "resume_batch"}
        },
        "prefix_selection": selection,
        "stages": {
            stage.key: {
                "harness": stage.harness,
                "prefix_files": [str(path) for path in stage.prefix_files],
                "vm_concurrency": stage.vm_concurrency,
                "epochs": stage.epochs,
                "expected_trajectories": stage.trajectories,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "campaign_ids": [],
                "summary": None,
                "error": None,
            }
            for stage in stages
        },
        "viewer_build": None,
        "summary": None,
    }


def _load_resume(cfg: dict) -> tuple[dict, Path, list[Stage]]:
    path = _manifest_path(str(cfg["resume_batch"]))
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "environments-continuation-prefix-batch-v1":
        raise SystemExit(f"not a continuation-prefix batch manifest: {path}")
    stored_cfg = manifest.get("configuration") or {}
    if cfg["dry_run"]:
        raise SystemExit("--dry-run is not meaningful with --resume-batch")
    stored_stages = manifest.get("stages") or {}
    unknown_stages = set(stored_stages) - set(STAGE_ORDER)
    if unknown_stages:
        raise SystemExit(
            f"continuation batch manifest has unknown stages: {sorted(unknown_stages)}"
        )
    ordered_keys = [key for key in STAGE_ORDER if key in stored_stages]
    if not ordered_keys:
        raise SystemExit(f"continuation batch manifest has no stages: {path}")
    stages = [
        Stage(
            key=key,
            harness=str(record["harness"]),
            prefix_files=tuple(Path(item) for item in record["prefix_files"]),
            vm_concurrency=int(record["vm_concurrency"]),
            epochs=int(record["epochs"]),
        )
        for key in ordered_keys
        for record in (stored_stages[key],)
    ]
    validate_stages(stages)
    manifest["configuration"] = stored_cfg
    return manifest, path, stages


def _campaign_cfg(
    cfg: dict,
    stage: Stage,
    specs: list,
    *,
    campaign_id: str,
) -> dict:
    destination = "checkout_redesign"
    descriptors = []
    for spec in specs:
        if spec.payload_path is None:
            raise SystemExit(f"prefix {spec.name!r} has no payload path")
        descriptors.append({
            "name": spec.name,
            "sha256": spec.sha256,
            "file_sha256": sha256_file(spec.payload_path),
            "local_path": str(spec.payload_path),
            "target": spec.target_name,
            "target_model": resolve_target(spec.target_name),
            "reasoning": spec.reasoning,
        })
    targets = list(dict.fromkeys(spec.target_name for spec in specs))
    judge = resolve_judge(None)
    return {
        "campaign_id": campaign_id,
        "treatment": cfg["treatment"],
        "continuation": {
            "treatment": cfg["treatment"],
            "payloads": descriptors,
        },
        "targets": targets,
        "target_models": [resolve_target(target) for target in targets],
        "seeds": [destination],
        # Rotate through every prefix before starting the next epoch. Besides making
        # progress representative throughout a long run, this prevents a 75-VM
        # subscription wave from concentrating all slots on one native account.
        "_cell_selections": [
            (spec.name, destination, epoch)
            for epoch in range(1, stage.epochs + 1)
            for spec in specs
        ],
        "seeds_path": str(SEEDS_PATH),
        "seed_dir": "p_hacking",
        "family": "p_hacking",
        "epochs": stage.epochs,
        "reasoning": None,
        "harness": stage.harness,
        "condition": "allow",
        "pressure": "low",
        "judge": None,
        "judge_resolved": judge,
        "gate_model": resolve_gate_model(None, None),
        "concurrency": 1,
        "sandbox_concurrency": 1,
        "time_limit": 1_800,
        "skip_viewer": True,
        "compute": "aws",
        "vm_concurrency": stage.vm_concurrency,
        "aws_region": cfg["aws_region"],
        "aws_instance_type": cfg["aws_instance_type"],
        "aws_bucket": cfg["aws_bucket"],
        "aws_secret_env": cfg["aws_secret_env"],
    }


def _control_cfg(cfg: dict, stage: Stage) -> dict:
    return {
        "harness": stage.harness,
        "harness_explicit": True,
        "aws_region": cfg["aws_region"],
        "aws_region_explicit": bool(cfg.get("aws_region_explicit")),
        "aws_instance_type": cfg["aws_instance_type"],
        "aws_instance_type_explicit": bool(cfg.get("aws_instance_type_explicit")),
        "aws_bucket": cfg["aws_bucket"],
        "aws_bucket_explicit": bool(cfg.get("aws_bucket_explicit")),
        "aws_secret_env": cfg["aws_secret_env"],
        "vm_concurrency": stage.vm_concurrency,
        "vm_concurrency_explicit": True,
        "dry_run": False,
        "concurrency": 1,
        "skip_viewer": True,
    }


def _selection_key(cell: dict) -> tuple:
    return (
        cell.get("prefix_name"),
        cell.get("seed"),
        cell.get("original_epoch"),
    )


def _stage_summary(states: list[dict], expected: int) -> dict:
    final_cells: dict[tuple, dict] = {}
    for state in states:
        for cell in state.get("cells") or []:
            final_cells[_selection_key(cell)] = cell
    succeeded = sum(
        cell.get("status") == "completed"
        and int((cell.get("terminal") or {}).get("pipeline_exit_code", 1)) == 0
        for cell in final_cells.values()
    )
    infrastructure_failures = sum(
        cell.get("status") in {"infrastructure_failure", "not_launched"}
        for cell in final_cells.values()
    )
    trajectory_failures = len(final_cells) - succeeded - infrastructure_failures
    return {
        "expected": expected,
        "accounted_for": len(final_cells),
        "succeeded": succeeded,
        "trajectory_failures": trajectory_failures,
        "infrastructure_failures": infrastructure_failures,
        "campaigns": len(states),
    }


def _has_infrastructure_failures(state: dict) -> bool:
    return any(
        cell.get("status") in {"infrastructure_failure", "not_launched"}
        for cell in state.get("cells") or []
    )


def _campaign_state_path(campaign_id: str) -> Path:
    return DATA_ROOT / "remote_campaigns" / f"{campaign_id}.json"


def _prior_campaign_states(campaign_ids: list[str]) -> list[dict]:
    """Load prior attempts so a resumed stage keeps its full-cell accounting."""

    states = []
    for campaign_id in campaign_ids:
        path = _campaign_state_path(campaign_id)
        if not path.is_file():
            raise AwsTrajectoryError(
                f"batch manifest references missing local campaign state: {path}"
            )
        states.append(json.loads(path.read_text()))
    return states


def _run_stage(
    cfg: dict,
    stage: Stage,
    specs: list,
    record: dict,
    manifest: dict,
    manifest_path: Path,
) -> None:
    campaign_ids = record.setdefault("campaign_ids", [])
    record.update({
        "status": "running",
        "started_at": record.get("started_at") or _now(),
        "error": None,
    })
    _write_manifest(manifest_path, manifest)

    existing_id = campaign_ids[-1] if campaign_ids else None
    states = _prior_campaign_states(campaign_ids[:-1])
    if existing_id and _campaign_state_path(existing_id).is_file():
        print(f"[resume] {stage.key}: {existing_id}")
        state = resume_campaign(
            _control_cfg(cfg, stage), ENVIRONMENTS, DATA_ROOT, existing_id
        )
    else:
        campaign_id = existing_id or f"{manifest['batch_id']}-{stage.key}"
        if not campaign_ids:
            campaign_ids.append(campaign_id)
            _write_manifest(manifest_path, manifest)
        print(f"[start] {stage.key}: {stage.trajectories} trajectories")
        state = run_campaign(
            _campaign_cfg(cfg, stage, specs, campaign_id=campaign_id),
            ENVIRONMENTS,
            DATA_ROOT,
        )
    states.append(state)

    retries = 0
    while (
        retries < cfg["infrastructure_retries"]
        and _has_infrastructure_failures(state)
    ):
        retries += 1
        print(
            f"[retry {retries}/{cfg['infrastructure_retries']}] "
            f"{stage.key}: infrastructure failures only"
        )
        state = retry_failed(
            _control_cfg(cfg, stage),
            ENVIRONMENTS,
            DATA_ROOT,
            state["campaign_id"],
        )
        states.append(state)
        campaign_ids.append(state["campaign_id"])
        _write_manifest(manifest_path, manifest)

    summary = _stage_summary(states, stage.trajectories)
    record.update({
        "status": "succeeded" if summary["succeeded"] == stage.trajectories else "partial",
        "completed_at": _now(),
        "summary": summary,
        "error": None,
    })
    _write_manifest(manifest_path, manifest)
    print(f"[done] {stage.key}: {summary}")


def _batch_summary(manifest: dict) -> dict:
    summaries = [
        record.get("summary") or {}
        for record in manifest["stages"].values()
    ]
    return {
        "expected": sum(int(item.get("expected") or 0) for item in summaries),
        "succeeded": sum(int(item.get("succeeded") or 0) for item in summaries),
        "trajectory_failures": sum(
            int(item.get("trajectory_failures") or 0) for item in summaries
        ),
        "infrastructure_failures": sum(
            int(item.get("infrastructure_failures") or 0) for item in summaries
        ),
        "stage_statuses": {
            key: record["status"] for key, record in manifest["stages"].items()
        },
    }


def _rebuild_viewer(manifest: dict, manifest_path: Path) -> bool:
    started = _now()
    print("\n[viewer] rebuilding the free static viewer ...")
    ok = asyncio.run(base_pipeline.run_env_post_stages({"skip_viewer": False}))
    manifest["viewer_build"] = {
        "started_at": started,
        "completed_at": _now(),
        "status": "succeeded" if ok else "failed",
    }
    _write_manifest(manifest_path, manifest)
    return ok


def _print_plan(cfg: dict, stages: list[Stage]) -> None:
    print("=" * 78)
    print("PURPOSE-BUILT PREFIX CONTINUATION BATCH")
    print("=" * 78)
    print(
        f"  treatment={cfg['treatment']}  epochs={cfg['epochs']}  "
        "destination=checkout_redesign  pressure=low"
    )
    print(
        f"  normal AWS concurrency={cfg['vm_concurrency']}  "
        f"subscription concurrency={cfg['subscription_vm_concurrency']}"
    )
    print("  order: " + " -> ".join(stage.key for stage in stages))
    for index, stage in enumerate(stages, start=1):
        print(
            f"\n  {index}. {stage.key}: {len(stage.prefix_files)} prefixes x "
            f"{stage.epochs} = {stage.trajectories} trajectories; "
            f"max {stage.vm_concurrency} VMs"
        )
        for path in stage.prefix_files:
            print(f"       {path.name}")
    print(f"\n  total: {sum(stage.trajectories for stage in stages)} trajectories")


def main() -> None:
    cfg = parse_args()
    if cfg["resume_batch"]:
        manifest, manifest_path, stages = _load_resume(cfg)
        stored_cfg = manifest["configuration"]
        cfg = {
            **stored_cfg,
            "resume_batch": str(manifest_path),
            "dry_run": False,
        }
    else:
        selection = discover_prefixes()
        stages = build_stages(cfg, selection)
        validate_stages(stages)
        _print_plan(cfg, stages)
        if cfg["dry_run"]:
            print("\n[dry-run] locally validated; no AWS calls, files, or spend.")
            return
        batch_id = _batch_id()
        batch_dir = BATCH_ROOT / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = batch_dir / "batch_manifest.json"
        manifest = _new_manifest(batch_id, batch_dir, cfg, selection, stages)
        _write_manifest(manifest_path, manifest)

    specs_by_stage = validate_stages(stages)
    print(f"[batch] manifest: {manifest_path}")
    try:
        for stage in stages:
            record = manifest["stages"][stage.key]
            if record["status"] in {"succeeded", "partial"}:
                print(f"[skip] {stage.key}: already {record['status']}")
                continue
            try:
                _run_stage(
                    cfg,
                    stage,
                    specs_by_stage[stage.key],
                    record,
                    manifest,
                    manifest_path,
                )
            except Exception as error:
                record.update({
                    "status": "failed",
                    "completed_at": _now(),
                    "error": f"{type(error).__name__}: {error}",
                })
                _write_manifest(manifest_path, manifest)
                raise
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except AwsTrajectoryError as error:
        print(f"\nAWS trajectory error: {error}")
        raise SystemExit(1) from error
    except Exception as error:
        print(f"\nbatch controller error: {type(error).__name__}: {error}")
        raise

    viewer_ok = True if cfg["skip_viewer"] else _rebuild_viewer(manifest, manifest_path)
    manifest["completed_at"] = _now()
    manifest["summary"] = _batch_summary(manifest)
    _write_manifest(manifest_path, manifest)

    print("\n" + "=" * 78)
    print("CONTINUATION PREFIX BATCH COMPLETE")
    print("=" * 78)
    print(f"  summary: {manifest['summary']}")
    print(f"  manifest: {manifest_path}")
    print(f"  viewer: {VIEWER_ROOT / 'index.html'}")
    if not viewer_ok or manifest["summary"]["succeeded"] != manifest["summary"]["expected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
