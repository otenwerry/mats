"""One-off: re-judge every original audit on the training-data-misuse page.

This is deliberately self-contained and may be deleted after the run. It:

* selects the exact original-audit rows displayed in viewer sweep ``more_exploring``;
* reports raw routed transcripts omitted from the viewer because they lack a usable row;
* requires every selected seed to belong to ``ml_training_data_misuse``;
* uses the same global + seed-scoped dimension routing as the live audit pipeline;
* replaces each successful audit's full entry in ``rejudge_scores.json``;
* removes stale single-dimension overlays that would otherwise override a fresh score;
* checkpoints after every audit, records judge usage/cost in mats-local, and is resumable;
* prints score averages, viewer-outcome transitions, and large per-audit changes; and
* rebuilds the viewer after judging, unless ``--skip-viewer`` is passed.

No trajectories are generated. This costs money only for the judge calls.

Run from ``petri/``:

    uv run tools/exp_rejudge_training_data_misuse_originals.py
    uv run tools/exp_rejudge_training_data_misuse_originals.py --ids=1491,1509

Re-running after an interruption skips entries already completed with the same model and
exact rubric contents. Pass ``--force`` only if you intentionally want to spend again.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PETRI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PETRI_ROOT / "lib"))
sys.path.insert(0, str(PETRI_ROOT))

from dotenv import load_dotenv  # noqa: E402
from inspect_ai.model._model import init_model_usage, model_usage  # noqa: E402
from inspect_petri import audit_judge  # noqa: E402
from inspect_scout import TranscriptContent, transcripts_from  # noqa: E402

import direct_cost  # noqa: E402
from dimension_routing import SEEDS_ROOT, audit_dimension_set  # noqa: E402
import model_prices  # noqa: E402
import openrouter_cost  # noqa: E402
from petri_paths import DATA, ENV_FILE  # noqa: E402
from viewer import (  # noqa: E402
    CONTINUATION_PREFIX,
    CURRENT_SWEEP,
    LOGS,
    REJUDGE_FILE,
    RESAMPLE_PREFIX,
    ROLLBACK_PREFIX,
    SEED_FAMILY,
    sweep_key,
    traj_key,
    v7_outcome_category,
)
from viewer_load import (  # noqa: E402
    DIM_SCORES_DIR,
    load_mode,
    parse_rh_category,
    usage_to_dict,
)


SCOPE = "training_data_misuse_original_audits"
EXPECTED_FAMILY = "ml_training_data_misuse"
BIG_CHANGE = 4
RUN_RECORD_DIR = DATA / "rejudge_runs"
REGISTRY_FILE = DATA / "trajectory_ids.json"


def parse_requested_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token.isdigit() or int(token) < 1 for token in tokens):
        raise ValueError("--ids must be a comma-separated list of positive viewer IDs")
    return {int(token) for token in tokens}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="anthropic/claude-opus-4-8", help="judge model"
    )
    parser.add_argument(
        "--concurrency", type=int, default=50, help="parallel judge calls (default: 50)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-spend on matching completed entries"
    )
    parser.add_argument(
        "--skip-viewer", action="store_true", help="do not rebuild the viewer afterward"
    )
    parser.add_argument(
        "--ids",
        help="only re-judge these comma-separated viewer IDs (default: every page audit)",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    try:
        args.ids = parse_requested_ids(args.ids)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def original_mode_dirs() -> list[Path]:
    excluded = (ROLLBACK_PREFIX, RESAMPLE_PREFIX, CONTINUATION_PREFIX)
    return sorted(
        path
        for path in LOGS.iterdir()
        if path.is_dir() and not path.name.startswith(excluded)
    )


def dimension_digest(selected) -> str:
    digest = hashlib.sha256()
    for path in selected.files:
        digest.update(str(path.relative_to(PETRI_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dimension_provenance(selected) -> dict[str, Any]:
    return {
        "judge_dimensions": selected.names,
        "judge_dimension_files": [
            str(path.relative_to(PETRI_ROOT)) for path in selected.files
        ],
        "dimension_scope": (
            str(selected.scope_dir.relative_to(PETRI_ROOT / "dimensions"))
            if selected.scope_dir is not None
            else None
        ),
        "dimension_digest": dimension_digest(selected),
    }


def judge_cost_from_audit(audit: dict) -> dict | None:
    usage = (audit.get("role_usage") or {}).get("judge")
    if not usage:
        return None
    role_cost = model_prices.cost_by_role(
        {"judge": usage}, {"judge": audit.get("judge")}
    ).get("judge")
    return role_cost


def output_category(scores: dict, justification: str) -> tuple[str, str | None]:
    rh_category, _rh_types, _parse_error = parse_rh_category(
        justification or "", scores.get("reward_hacking")
    )
    category = v7_outcome_category({"scores": scores, "rh_category": rh_category})
    return category, rh_category


def load_sidecars() -> dict[Path, dict]:
    if not DIM_SCORES_DIR.is_dir():
        return {}
    return {path: read_json(path, {}) for path in sorted(DIM_SCORES_DIR.glob("*.json"))}


async def collect_candidates() -> tuple[list[dict], list[dict], list[str]]:
    """Return (judgeable page audits, dead audits, raw transcripts absent from the page).

    Sweep routing alone is not enough to establish viewer membership: a raw transcript can
    route to this sweep while being omitted by ``load_mode`` because its original eval log
    has no usable ``audit_judge`` score. Intersecting the routed transcript index with the
    viewer-loaded audits keeps this one-off scoped to exactly what the user can see.
    """

    registry = read_json(REGISTRY_FILE, {})
    candidates: list[dict] = []
    dead: list[dict] = []
    absent_from_viewer: list[str] = []
    unexpected: list[str] = []

    for mode_dir in original_mode_dirs():
        transcripts = transcripts_from(str(mode_dir))
        async with transcripts.reader() as reader:
            infos = [info async for info in reader.index()]
            page_infos = [
                info
                for info in infos
                if sweep_key({"mode": mode_dir.name, "seed": str(info.task_id)})
                == CURRENT_SWEEP
            ]
            if not page_infos:
                continue

            print(
                f"  loading {mode_dir.name}/ "
                f"({len(page_infos)} raw transcript(s) routed to this page) ...",
                flush=True,
            )
            loaded = {traj_key(audit): audit for audit in await load_mode(mode_dir)}
            info_by_key = {}
            for info in page_infos:
                seed = str(info.task_id)
                key = traj_key(
                    {
                        "mode": mode_dir.name,
                        "task": info.task_set,
                        "seed": seed,
                        "epoch": info.task_repeat,
                    }
                )
                info_by_key[key] = info

            displayed = {
                key: audit
                for key, audit in loaded.items()
                if sweep_key(audit) == CURRENT_SWEEP
            }
            absent_from_viewer.extend(sorted(set(info_by_key) - set(displayed)))
            if not displayed:
                print("    no viewer audits in this directory; skipping", flush=True)
                continue
            print(f"    {len(displayed)} audit(s) are displayed in the viewer", flush=True)

            for key, audit in displayed.items():
                info = info_by_key.get(key)
                if info is None:
                    raise SystemExit(
                        f"viewer audit {key} has no matching raw transcript; "
                        "refusing a partial rerun"
                    )
                seed = str(info.task_id)
                family = SEED_FAMILY.get(seed)
                if family != EXPECTED_FAMILY:
                    unexpected.append(f"{key} (seed family {family!r})")
                    continue
                if audit.get("dead"):
                    dead.append(audit)
                    continue

                seed_path = SEEDS_ROOT / EXPECTED_FAMILY / seed
                selected = audit_dimension_set(seed_path)
                transcript = await reader.read(
                    info, TranscriptContent(messages="all", events="all")
                )
                if not transcript.timelines:
                    dead.append(audit)
                    continue

                provenance = dimension_provenance(selected)
                before_category, before_rh_category = output_category(
                    dict(audit.get("scores") or {}), audit.get("justification") or ""
                )
                candidates.append(
                    {
                        "key": key,
                        "id": registry.get(key),
                        "mode": mode_dir.name,
                        "task": info.task_set,
                        "seed": seed,
                        "epoch": info.task_repeat,
                        "audit": audit,
                        "transcript": transcript,
                        "dimensions": selected.dimensions,
                        "provenance": provenance,
                        "before_scores": dict(audit.get("scores") or {}),
                        "before_category": before_category,
                        "before_rh_category": before_rh_category,
                    }
                )

    if unexpected:
        details = "\n  ".join(unexpected)
        raise SystemExit(
            "The viewer catch-all currently contains non-training-data-misuse audits. "
            "The requested scope is therefore ambiguous; no API calls were made:\n  " + details
        )
    return candidates, dead, absent_from_viewer


def matching_completed(entry: dict | None, candidate: dict, model: str) -> bool:
    return bool(
        entry
        and entry.get("rerun_scope") == SCOPE
        and entry.get("model") == model
        and entry.get("dimension_digest")
        == candidate["provenance"]["dimension_digest"]
        and isinstance(entry.get("scores"), dict)
    )


def summary_record_from_entry(candidate: dict, entry: dict) -> dict:
    after_category, after_rh_category = output_category(
        entry["scores"], entry.get("justification") or ""
    )
    return {
        "key": candidate["key"],
        "id": candidate["id"],
        "seed": candidate["seed"],
        "epoch": candidate["epoch"],
        "before_scores": entry.get("previous_scores", candidate["before_scores"]),
        "after_scores": entry["scores"],
        "before_category": entry.get(
            "previous_viewer_outcome", candidate["before_category"]
        ),
        "after_category": after_category,
        "before_rh_category": entry.get(
            "previous_rh_category", candidate["before_rh_category"]
        ),
        "after_rh_category": after_rh_category,
    }


def print_change_summary(records: list[dict], dim_names: list[str]) -> None:
    if not records:
        print("\nNo completed judgments to summarize.")
        return

    print("\n" + "=" * 88)
    print(f"SCORE CHANGE SUMMARY ({len(records)} audits)")
    print("dimension                     before   after   mean Δ   mean |Δ|   changed")
    print("-" * 88)
    large: list[tuple[float, dict, str, float, float]] = []
    trajectory_changes: list[tuple[float, float, dict]] = []

    for dim in dim_names:
        pairs = [
            (record["before_scores"].get(dim), record["after_scores"].get(dim))
            for record in records
        ]
        pairs = [
            (before, after)
            for before, after in pairs
            if isinstance(before, (int, float)) and isinstance(after, (int, float))
        ]
        if not pairs:
            print(f"{dim:<29} {'n/a':>7} {'n/a':>7} {'n/a':>8} {'n/a':>10} {'0/0':>9}")
            continue
        deltas = [after - before for before, after in pairs]
        changed = sum(delta != 0 for delta in deltas)
        print(
            f"{dim:<29} {sum(p[0] for p in pairs) / len(pairs):>7.2f}"
            f" {sum(p[1] for p in pairs) / len(pairs):>7.2f}"
            f" {sum(deltas) / len(deltas):>+8.2f}"
            f" {sum(abs(delta) for delta in deltas) / len(deltas):>10.2f}"
            f" {changed:>4}/{len(pairs):<4}"
        )

    for record in records:
        deltas = []
        for dim in dim_names:
            before = record["before_scores"].get(dim)
            after = record["after_scores"].get(dim)
            if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
                continue
            delta = after - before
            deltas.append(abs(delta))
            if abs(delta) >= BIG_CHANGE:
                large.append((abs(delta), record, dim, before, after))
        trajectory_changes.append((max(deltas, default=0), sum(deltas), record))

    outcome_moves = Counter(
        (record["before_category"], record["after_category"])
        for record in records
        if record["before_category"] != record["after_category"]
    )
    rh_moves = Counter(
        (record["before_rh_category"], record["after_rh_category"])
        for record in records
        if record["before_rh_category"] != record["after_rh_category"]
    )
    print(f"\nViewer outcome changed for {sum(outcome_moves.values())}/{len(records)} audits:")
    if outcome_moves:
        for (before, after), count in outcome_moves.most_common():
            print(f"  {before} -> {after}: {count}")
    else:
        print("  none")
    print(f"Judge RH_CATEGORY changed for {sum(rh_moves.values())}/{len(records)} audits:")
    if rh_moves:
        for (before, after), count in rh_moves.most_common():
            print(f"  {before or 'missing'} -> {after or 'missing'}: {count}")
    else:
        print("  none")

    large.sort(key=lambda item: (-item[0], item[1]["id"] or 10**9, item[2]))
    print(f"\nLarge individual score changes (|Δ| >= {BIG_CHANGE}): {len(large)}")
    for magnitude, record, dim, before, after in large[:40]:
        label = f"#{record['id']}" if record["id"] is not None else record["key"]
        print(
            f"  {label} {record['seed']} e{record['epoch']}: "
            f"{dim} {before:g} -> {after:g} (Δ {after - before:+g})"
        )
    if len(large) > 40:
        print(f"  ... {len(large) - 40} more (full scores are stored in rejudge_scores.json)")

    trajectory_changes.sort(
        key=lambda item: (-item[0], -item[1], item[2]["id"] or 10**9)
    )
    print("\nTen audits with the largest overall score movement:")
    for max_delta, total_delta, record in trajectory_changes[:10]:
        label = f"#{record['id']}" if record["id"] is not None else record["key"]
        print(
            f"  {label} {record['seed']} e{record['epoch']}: "
            f"largest |Δ|={max_delta:g}, sum |Δ|={total_delta:g}"
        )
    print("=" * 88)


async def main() -> None:
    args = parse_args()
    load_dotenv(ENV_FILE)
    direct_cost.install()
    openrouter_cost.install()

    existing = read_json(REJUDGE_FILE, {})
    sidecars = load_sidecars()
    candidates, dead, absent_from_viewer = await collect_candidates()
    if args.ids is not None:
        available_ids = {
            candidate["id"] for candidate in candidates if candidate["id"] is not None
        }
        missing_ids = sorted(args.ids - available_ids)
        if missing_ids:
            raise SystemExit(
                "Requested viewer IDs are not live audits on the training-data-misuse "
                f"originals page: {missing_ids}. No API calls were made."
            )
        candidates = [candidate for candidate in candidates if candidate["id"] in args.ids]
    if not candidates:
        raise SystemExit("No live original audits found on the training-data-misuse page.")

    names_by_digest: dict[str, list[str]] = {}
    files_by_digest: dict[str, list[str]] = {}
    for candidate in candidates:
        provenance = candidate["provenance"]
        names_by_digest[provenance["dimension_digest"]] = provenance["judge_dimensions"]
        files_by_digest[provenance["dimension_digest"]] = provenance["judge_dimension_files"]
    if len(names_by_digest) != 1:
        raise SystemExit(
            "Selected audits resolve to different dimension sets. This one-off expects the "
            f"training-data-misuse family to share one set; found {len(names_by_digest)}."
        )
    digest = next(iter(names_by_digest))
    dim_names = names_by_digest[digest]

    historical_costs = [judge_cost_from_audit(c["audit"]) for c in candidates]
    priced = [cost for cost in historical_costs if cost is not None and not cost["unpriced"]]
    historical_total = sum(cost["cost"] for cost in priced)
    historical_exact = bool(priced) and all(cost["exact"] for cost in priced)

    completed = [
        candidate
        for candidate in candidates
        if matching_completed(existing.get(candidate["key"]), candidate, args.model)
    ]
    completed_keys = {candidate["key"] for candidate in completed}
    pending = (
        candidates
        if args.force
        else [candidate for candidate in candidates if candidate["key"] not in completed_keys]
    )

    print("\n" + "=" * 88)
    print("TRAINING-DATA-MISUSE ORIGINAL-AUDIT RE-JUDGE")
    if args.ids is not None:
        print(f"  requested viewer IDs: {sorted(args.ids)}")
    print(f"  selected: {len(candidates)} live audits across {len({c['mode'] for c in candidates})} run dirs")
    print(f"  dead/empty skipped: {len(dead)} (already stored as dead in the audit data)")
    print(
        f"  raw transcripts absent from viewer skipped: {len(absent_from_viewer)} "
        "(no usable viewer audit row)"
    )
    print(f"  model: {args.model}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  dimensions ({len(dim_names)}): {dim_names}")
    print(f"  rubric files: {files_by_digest[digest]}")
    print(f"  rubric digest: {digest[:16]}...")
    tilde = "" if historical_exact else "~"
    print(
        f"  historical judge cost for this same set: {tilde}${historical_total:.2f} "
        f"({len(priced)}/{len(candidates)} audits priced)"
    )
    print(f"  already completed with this exact setup: {len(completed)}")
    print(f"  API calls to make now: {len(pending)}")
    if args.force:
        print("  --force: matching completed entries will be judged again")
    print("=" * 88 + "\n")

    if not pending:
        records = [
            summary_record_from_entry(candidate, existing[candidate["key"]])
            for candidate in completed
        ]
        print("Nothing to spend: every selected audit already matches this model and rubric digest.")
        print_change_summary(records, dim_names)
        if not args.skip_viewer:
            print("\n[viewer] rebuilding (free) ...", flush=True)
            subprocess.run([sys.executable, str(PETRI_ROOT / "viewer.py")], check=True)
        return

    started = datetime.now(timezone.utc)
    run_file = RUN_RECORD_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    run_record: dict[str, Any] = {
        "scope": SCOPE,
        "started_at": started.isoformat(),
        "model": args.model,
        "concurrency": args.concurrency,
        "dimension_digest": digest,
        "judge_dimensions": dim_names,
        "judge_dimension_files": files_by_digest[digest],
        "n_selected": len(candidates),
        "n_pending": len(pending),
        "requested_viewer_ids": sorted(args.ids) if args.ids is not None else None,
        "n_dead_skipped": len(dead),
        "n_raw_transcripts_absent_from_viewer": len(absent_from_viewer),
        "raw_transcripts_absent_from_viewer": absent_from_viewer,
        "historical_judge_cost_usd": historical_total,
        "historical_cost_exact": historical_exact,
        "attempts": {},
    }
    write_json_atomic(run_file, run_record)

    scanners: dict[str, Any] = {}
    for candidate in pending:
        candidate_digest = candidate["provenance"]["dimension_digest"]
        if candidate_digest not in scanners:
            scanners[candidate_digest] = audit_judge(
                dimensions=list(candidate["dimensions"]), model=args.model
            )

    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = failed = stale_sidecars_removed = 0
    records: list[dict] = []

    async def checkpoint(candidate: dict, entry: dict, attempt: dict) -> None:
        nonlocal stale_sidecars_removed
        key = candidate["key"]
        active_dims = set(candidate["provenance"]["judge_dimensions"])
        removed = 0
        for path, values in sidecars.items():
            if path.stem in active_dims and key in values:
                del values[key]
                removed += 1

        # Sidecars apply after the full replacement in viewer_load. Write their removals
        # first, so an interruption cannot leave a stale override winning over a fresh entry.
        for path, values in sidecars.items():
            write_json_atomic(path, values)
        existing[key] = entry
        write_json_atomic(REJUDGE_FILE, existing)
        stale_sidecars_removed += removed
        attempt["stale_dimension_overlays_removed"] = removed
        run_record["attempts"][key] = attempt
        write_json_atomic(run_file, run_record)

    async def record_failed_attempt(key: str, attempt: dict) -> None:
        async with lock:
            run_record["attempts"][key] = attempt
            write_json_atomic(run_file, run_record)

    async def judge_one(index: int, candidate: dict) -> None:
        nonlocal done, failed
        label = f"#{candidate['id']}" if candidate["id"] is not None else candidate["key"]
        async with semaphore:
            print(
                f"  [{index}/{len(pending)}] judging {label} "
                f"{candidate['seed']} e{candidate['epoch']} ...",
                flush=True,
            )
            init_model_usage({})
            try:
                result = await scanners[candidate["provenance"]["dimension_digest"]](
                    candidate["transcript"]
                )
            except Exception as exc:
                usage = usage_to_dict(model_usage())
                cost = model_prices.sample_cost(usage)
                failed += 1
                attempt = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage": usage,
                    "cost_usd": cost["total"],
                    "cost_exact": cost["exact"],
                }
                await record_failed_attempt(candidate["key"], attempt)
                print(f"    FAILED {candidate['key']}: {attempt['error']}", flush=True)
                return

        usage = usage_to_dict(model_usage())
        cost = model_prices.sample_cost(usage)
        attempt = {
            "status": "succeeded",
            "usage": usage,
            "cost_usd": cost["total"],
            "cost_exact": cost["exact"],
            "unpriced_models": cost["unknown"],
        }
        if not isinstance(result.value, dict):
            failed += 1
            attempt["status"] = "failed"
            attempt["error"] = (
                (result.metadata or {}).get("stop_reason") or "judge returned no score object"
            )
            await record_failed_attempt(candidate["key"], attempt)
            print(f"    FAILED {candidate['key']}: {attempt['error']}", flush=True)
            return

        scores = {name: result.value.get(name) for name in dim_names}
        missing = [name for name, value in scores.items() if not isinstance(value, (int, float))]
        if missing:
            failed += 1
            attempt["status"] = "failed"
            attempt["error"] = f"judge omitted dimensions {missing}"
            await record_failed_attempt(candidate["key"], attempt)
            print(f"    FAILED {candidate['key']}: {attempt['error']}", flush=True)
            return

        metadata = result.metadata or {}
        justification = getattr(result, "explanation", None) or ""
        entry = {
            "mode": candidate["mode"],
            "task": candidate["task"],
            "seed": candidate["seed"],
            "epoch": candidate["epoch"],
            "model": args.model,
            "scores": scores,
            "summary": metadata.get("summary", ""),
            "highlights": metadata.get("highlights", ""),
            "justification": justification,
            "rerun_scope": SCOPE,
            "rejudged_at": datetime.now(timezone.utc).isoformat(),
            **candidate["provenance"],
            "previous_scores": candidate["before_scores"],
            "previous_viewer_outcome": candidate["before_category"],
            "previous_rh_category": candidate["before_rh_category"],
            "rejudge_usage": usage,
            "rejudge_cost_usd": cost["total"],
            "rejudge_cost_exact": cost["exact"],
            "rejudge_unpriced_models": cost["unknown"],
        }
        async with lock:
            await checkpoint(candidate, entry, attempt)
            records.append(summary_record_from_entry(candidate, entry))
            done += 1
        largest = max(
            (
                abs(scores[name] - candidate["before_scores"][name])
                for name in dim_names
                if isinstance(candidate["before_scores"].get(name), (int, float))
            ),
            default=0,
        )
        cost_mark = "" if cost["exact"] else "~"
        print(
            f"    done {label}: largest |score change|={largest:g}, "
            f"judge cost={cost_mark}${cost['total']:.4f}",
            flush=True,
        )

    await asyncio.gather(
        *(judge_one(index, candidate) for index, candidate in enumerate(pending, 1))
    )

    # Include same-setup entries skipped on this invocation, so a resumed run reports the
    # complete rubric rerun rather than only its final retry tranche.
    records.extend(
        summary_record_from_entry(candidate, existing[candidate["key"]])
        for candidate in completed
        if candidate["key"] in existing
    )
    run_costs = [attempt.get("cost_usd") for attempt in run_record["attempts"].values()]
    run_costs = [cost for cost in run_costs if isinstance(cost, (int, float))]
    run_exact = bool(run_costs) and all(
        attempt.get("cost_exact", False) for attempt in run_record["attempts"].values()
    )
    run_record.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "n_succeeded": done,
            "n_failed": failed,
            "stale_dimension_overlays_removed": stale_sidecars_removed,
            "total_cost_usd": sum(run_costs),
            "cost_exact": run_exact,
        }
    )
    write_json_atomic(run_file, run_record)

    print_change_summary(records, dim_names)
    run_tilde = "" if run_exact else "~"
    print(f"\nThis invocation: {done} succeeded, {failed} failed")
    print(f"Stale per-dimension overlays removed: {stale_sidecars_removed}")
    print(f"Recorded judge spend: {run_tilde}${sum(run_costs):.2f} -> {run_file}")
    print(f"Fresh full judge outputs -> {REJUDGE_FILE}")

    if not args.skip_viewer:
        print("\n[viewer] rebuilding (free) ...", flush=True)
        subprocess.run([sys.executable, str(PETRI_ROOT / "viewer.py")], check=True)
        print("[viewer] done; reload the training data misuse / original audits page.")
    else:
        print("\n--skip-viewer: run `uv run viewer.py` before reloading the page.")

    if failed:
        raise SystemExit(
            f"{failed} audit(s) failed and kept their previous scores. Re-run the same command; "
            "successful matching entries will be skipped."
        )


if __name__ == "__main__":
    asyncio.run(main())
