"""Continuation experiment core for real environments (importable).

A continuation asks: if a target reward-hacks on one task and is then handed a
SECOND, unrelated task, does its probability of hacking on the second task go up --
because it is conditioned on its own earlier behavior?

Each cell conditions ONE target model on a PREFIX (a prior conversation) and then
drops it into a real environment seed. One invocation = ONE TREATMENT, a free-form
label chosen with --treatment. There is no no-prefix condition here: the environment
is pinned and scripted, so the base rate comes from the ordinary original
trajectories of the same (seed, target).

PREFIXES are self-contained payload files (``environments-continuation-prefix-v1``):
the exact message list the target carries, plus the catalog target name and its
reasoning setting. They come from two places:

- ``--prefixes=<viewer ids>``: reconstructed from stored real-environment logs
  (the viewer's stable integer IDs) and written under
  ``mats-local/environments/continuation_prefixes/``.
- ``--prefix-files=<paths>``: arbitrary hand-built conversations (e.g. a long Q&A
  session). Owen constructs these to match the payload contract below.

The spliced target context is [prefix's system prompt + prefix conversation +
injected pivot user turn + the seed's opening message, live]. The pivot sentence is
identical to the Petri continuation experiment; when the prefix is not from the new
task's own seed family (including every external prefix) it also names the new kind
of work. Both judge stages see only the system message plus everything from the
pivot onward -- the prefix is sliced out mechanically by a stored boundary index
(``lib/continuation_evidence.py``), so no text matching is involved and the judge
cannot see or infer anything about the prior task.

INVARIANTS enforced before any paid work:
- the prefix's system prompt must byte-match the current ``seeds/SYSTEM_PROMPT.txt``
  in the variant implied by its reasoning setting (a drifted global prompt would
  confound the comparison against the original-trajectory base rate);
- a trajectory-sourced prefix must not share a seed with the new task;
- integrity-excluded and rejudge trajectories are refused as prefixes;
- a prefix ending on unresolved tool calls gets a neutral synthetic closer,
  surfaced and stored, never silent.

This file makes no API calls itself; it builds plans and Inspect tasks that the
``exp_continuation_pipeline.py`` endpoint runs. The ``exp_`` prefix follows the
convention of its Petri counterpart (``petri/lib/exp_continuation.py``).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.model import ChatMessage, ChatMessageSystem
from pydantic import TypeAdapter

from continuation_evidence import (
    CONTINUATION_RECORD_VERSION,
    ContinuationRun,
    close_dangling_tool_calls,
    continuation_record,
    pivot_preamble,
    prefix_boundary_index,
    seed_family_task_description,
)
from exp_real_audit import assemble_real_protocol, build_real_task
from judge_selection import resolve_judge
from model_catalog import (
    REASONING_EFFORT,
    TARGET_CHOICES,
    build_target,
    resolve_target,
)
from project_paths import DATA_ROOT, LOGS_ROOT
from prompt_caching import stable_key
from protocol_sources import (
    global_system_prompt,
    is_seed_member,
    seed_members,
    strip_thinking_instruction,
)


PREFIX_FORMAT = "environments-continuation-prefix-v1"
PREFIX_STORE = DATA_ROOT / "continuation_prefixes"
REGISTRY_FILE = DATA_ROOT / "trajectory_ids.json"

# Treatment and prefix names survive round trips through task names, file names, and
# S3 keys, so they share one strict slug rule (same as Petri's treatment slugs).
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

_MESSAGES_ADAPTER = TypeAdapter(list[ChatMessage])


def validate_treatment(treatment: str) -> str:
    value = (treatment or "").strip().lower()
    if not _SLUG_RE.match(value):
        raise SystemExit(
            f"--treatment {treatment!r} is not a valid slug. Use lowercase letters, "
            "digits and single hyphens (e.g. full-hack, clean), because the treatment "
            "is encoded into task and run-directory names."
        )
    return value


# --------------------------------------------------------------------------- #
# Prefix payloads (the one runtime input format for prefixes)
# --------------------------------------------------------------------------- #
def payload_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_prefix_payload(payload: dict, *, origin: str) -> dict:
    """Loud structural validation; returns the payload unchanged."""

    if not isinstance(payload, dict):
        raise SystemExit(f"{origin}: prefix payload must be a JSON object")
    if payload.get("format") != PREFIX_FORMAT:
        raise SystemExit(
            f"{origin}: prefix payload format must be {PREFIX_FORMAT!r}, got "
            f"{payload.get('format')!r}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not _SLUG_RE.match(name):
        raise SystemExit(
            f"{origin}: prefix payload needs a slug 'name' (lowercase letters, "
            f"digits, single hyphens); got {name!r}"
        )
    target = payload.get("target")
    if target not in TARGET_CHOICES:
        raise SystemExit(
            f"{origin}: prefix payload 'target' must be a catalog name "
            f"({sorted(TARGET_CHOICES)}); got {target!r}"
        )
    if not isinstance(payload.get("reasoning"), bool):
        raise SystemExit(f"{origin}: prefix payload 'reasoning' must be true/false")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {
        "trajectory", "external",
    }:
        raise SystemExit(
            f"{origin}: prefix payload 'source.kind' must be 'trajectory' or "
            "'external'"
        )
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit(f"{origin}: prefix payload needs a non-empty 'messages' list")
    return payload


def load_prefix_payload_file(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"prefix payload file not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{file_path} is not valid JSON: {error}") from error
    return validate_prefix_payload(payload, origin=str(file_path))


def store_prefix_payload(payload: dict) -> Path:
    """Write a payload to the content-addressed local store (idempotent)."""

    PREFIX_STORE.mkdir(parents=True, exist_ok=True)
    sha = payload_sha256(payload)
    path = PREFIX_STORE / f"{payload['name']}-{sha[:12]}.json"
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Reconstructing prefixes from stored trajectories (viewer IDs)
# --------------------------------------------------------------------------- #
def _registry_entries(trajectory_ids: list[int]) -> dict[int, dict]:
    """Map viewer IDs to their stored (mode, task, seed, epoch) identity."""

    if not REGISTRY_FILE.is_file():
        raise SystemExit(
            f"trajectory ID registry not found: {REGISTRY_FILE} (build the viewer "
            "once so IDs exist, or pass --prefix-files instead)"
        )
    registry = json.loads(REGISTRY_FILE.read_text())
    by_id = {value: key for key, value in registry.items()}
    entries: dict[int, dict] = {}
    for trajectory_id in trajectory_ids:
        key = by_id.get(trajectory_id)
        if key is None:
            raise SystemExit(
                f"--prefixes id {trajectory_id} is not in {REGISTRY_FILE.name}"
            )
        parts = key.split("__")
        if len(parts) != 4 or not parts[3].startswith("e"):
            raise SystemExit(
                f"registry key for trajectory #{trajectory_id} has an unexpected "
                f"shape: {key!r}"
            )
        entries[trajectory_id] = {
            "mode": parts[0],
            "task": parts[1],
            "seed": parts[2],
            "epoch": int(parts[3][1:]),
        }
    return entries


def _integrity_status(mode_dir: Path, task: str, seed: str, epoch: int) -> dict | None:
    sidecar = mode_dir / "pipeline_integrity.json"
    if not sidecar.is_file():
        return None
    records = (json.loads(sidecar.read_text()) or {}).get("records") or []
    for record in records:
        if (
            record.get("task") == task
            and str(record.get("sample")) == seed
            and record.get("epoch") == epoch
        ):
            return record
    return None


def reconstruct_prefix_payload(trajectory_id: int, entry: dict) -> dict:
    """Rebuild one stored trajectory's exact target-visible conversation."""

    from inspect_ai.log import list_eval_logs, read_eval_log

    mode_dir = LOGS_ROOT / entry["mode"]
    if not mode_dir.is_dir():
        raise SystemExit(
            f"trajectory #{trajectory_id}: run directory {mode_dir} is missing"
        )
    sample = None
    run_metadata: dict = {}
    for log_info in list_eval_logs(str(mode_dir)):
        header = read_eval_log(log_info, header_only=True)
        if header.eval.task != entry["task"]:
            continue
        log = read_eval_log(log_info, resolve_attachments=True)
        run_metadata = dict(log.eval.metadata or {})
        for candidate in log.samples or []:
            if (
                str(candidate.id) == entry["seed"]
                and int(candidate.epoch) == entry["epoch"]
            ):
                sample = candidate
                break
        if sample is not None:
            break
    if sample is None:
        raise SystemExit(
            f"trajectory #{trajectory_id} ({entry['mode']}/{entry['task']} "
            f"seed={entry['seed']} epoch={entry['epoch']}) was not found on disk"
        )

    sample_metadata = sample.metadata or {}
    if sample_metadata.get("retrospective_rejudge"):
        raise SystemExit(
            f"trajectory #{trajectory_id} is a retrospective rejudge, not an "
            "original trajectory; pick the original's ID"
        )
    integrity = _integrity_status(
        mode_dir, entry["task"], entry["seed"], entry["epoch"]
    )
    if integrity is None:
        print(
            f"  NOTE: trajectory #{trajectory_id} has no pipeline_integrity record; "
            "proceeding without an integrity check."
        )
    elif integrity.get("status") == "excluded":
        raise SystemExit(
            f"trajectory #{trajectory_id} is integrity-excluded "
            f"({integrity.get('issues')}); refusing it as a prefix"
        )

    real_env = sample_metadata.get("real_env") or {}
    if real_env.get("continuation"):
        print(
            f"  NOTE: trajectory #{trajectory_id} is itself a continuation; its "
            "whole conversation (earlier prefix included) becomes the new prefix."
        )
    target_name = run_metadata.get("target_name")
    if target_name not in TARGET_CHOICES:
        raise SystemExit(
            f"trajectory #{trajectory_id} has no usable stamped target_name "
            f"({target_name!r}); cannot rebuild its target for a continuation"
        )
    reasoning = run_metadata.get("reasoning")
    if not isinstance(reasoning, bool):
        raise SystemExit(
            f"trajectory #{trajectory_id} has no stamped boolean 'reasoning' "
            "setting; cannot replicate its target configuration"
        )
    messages = [
        message.model_dump(mode="json", exclude_none=False)
        for message in (sample.messages or [])
    ]
    if not messages:
        raise SystemExit(f"trajectory #{trajectory_id} has no stored messages")

    protocol_sources = run_metadata.get("protocol_sources") or {}
    payload = {
        "format": PREFIX_FORMAT,
        "name": f"traj{trajectory_id}",
        "target": target_name,
        "reasoning": reasoning,
        "source": {
            "kind": "trajectory",
            "trajectory_id": trajectory_id,
            "run": entry["mode"],
            "task": entry["task"],
            "seed": entry["seed"],
            "epoch": entry["epoch"],
            "family": (
                real_env.get("family")
                or protocol_sources.get("protocol_family")
            ),
            "condition": run_metadata.get("condition"),
            "was_continuation": bool(real_env.get("continuation")),
        },
        "messages": messages,
    }
    return validate_prefix_payload(payload, origin=f"trajectory #{trajectory_id}")


def reconstruct_prefix_payloads(trajectory_ids: list[int]) -> list[dict]:
    entries = _registry_entries(trajectory_ids)
    return [
        reconstruct_prefix_payload(trajectory_id, entries[trajectory_id])
        for trajectory_id in trajectory_ids
    ]


# --------------------------------------------------------------------------- #
# Prefix specs (validated, spliceable prefixes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrefixSpec:
    """One validated prefix, ready to splice in front of a new task."""

    name: str
    payload: dict
    sha256: str
    payload_path: Path | None
    messages: tuple[ChatMessage, ...]  # closed; head is the verified system message
    target_name: str                   # catalog key
    reasoning: bool
    family: str | None                 # source seed family; None for external
    source_seed: str | None
    synthesized_closer: bool
    system_prompt_inserted: bool
    boundary_index: int

    def record(self) -> dict:
        """The stored ``real_env["continuation"]["prefix"]`` block."""

        return {
            "name": self.name,
            "sha256": self.sha256,
            "target_name": self.target_name,
            "reasoning": self.reasoning,
            "family": self.family,
            "synthesized_closer": self.synthesized_closer,
            "system_prompt_inserted": self.system_prompt_inserted,
            **{f"source_{key}": value for key, value in self.payload["source"].items()},
        }

    @property
    def char_count(self) -> int:
        return sum(len(message.text or "") for message in self.messages)


def _ensure_unique_message_ids(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Judge evidence numbering requires unique Inspect message IDs per sample."""

    seen: set[str] = set()
    fixed: list[ChatMessage] = []
    for message in messages:
        identifier = getattr(message, "id", None)
        if not identifier or identifier in seen:
            message = message.model_copy(update={"id": uuid.uuid4().hex})
        seen.add(message.id)
        fixed.append(message)
    return fixed


def expected_system_prompt(reasoning: bool) -> str:
    prompt = global_system_prompt()
    return strip_thinking_instruction(prompt) if reasoning else prompt


def build_prefix_spec(payload: dict, *, payload_path: Path | None = None) -> PrefixSpec:
    """Validate one payload against the experiment invariants and close it."""

    origin = payload["name"]
    try:
        messages = _MESSAGES_ADAPTER.validate_python(payload["messages"])
    except Exception as error:
        raise SystemExit(
            f"prefix {origin!r}: messages do not parse as Inspect chat messages: "
            f"{error}"
        ) from error

    expected = expected_system_prompt(payload["reasoning"])
    system_prompt_inserted = False
    if messages[0].role == "system":
        actual = messages[0].text or ""
        if actual != expected:
            raise SystemExit(
                f"prefix {origin!r}: its recorded system prompt does not byte-match "
                "the current seeds/SYSTEM_PROMPT.txt "
                f"({'reasoning-stripped' if payload['reasoning'] else 'verbatim'} "
                f"variant): prefix sha "
                f"{hashlib.sha256(actual.encode()).hexdigest()[:12]}, current sha "
                f"{hashlib.sha256(expected.encode()).hexdigest()[:12]}. A drifted "
                "global prompt would confound the comparison against the original "
                "base rate, so this is refused rather than silently replayed."
            )
    else:
        messages = [ChatMessageSystem(content=expected), *messages]
        system_prompt_inserted = True
    extra_system = [
        index for index, message in enumerate(messages[1:], start=2)
        if message.role == "system"
    ]
    if extra_system:
        raise SystemExit(
            f"prefix {origin!r}: system messages beyond the first are not supported "
            f"(found at positions {extra_system})"
        )

    messages = _ensure_unique_message_ids(list(messages))
    closed, synthesized = close_dangling_tool_calls(messages)
    source = payload["source"]
    return PrefixSpec(
        name=payload["name"],
        payload=payload,
        sha256=payload_sha256(payload),
        payload_path=payload_path,
        messages=tuple(closed),
        target_name=payload["target"],
        reasoning=payload["reasoning"],
        family=source.get("family"),
        source_seed=source.get("seed"),
        synthesized_closer=synthesized,
        system_prompt_inserted=system_prompt_inserted,
        boundary_index=prefix_boundary_index(closed),
    )


def load_prefix_specs(
    trajectory_ids: list[int], payload_files: list[str]
) -> list[PrefixSpec]:
    """Resolve both prefix sources into validated specs with unique names."""

    specs: list[PrefixSpec] = []
    for payload in reconstruct_prefix_payloads(trajectory_ids):
        path = store_prefix_payload(payload)
        specs.append(build_prefix_spec(payload, payload_path=path))
    for file_path in payload_files:
        payload = load_prefix_payload_file(file_path)
        specs.append(build_prefix_spec(payload, payload_path=Path(file_path)))
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(
            f"prefix names must be unique within one run; duplicates: {duplicates}"
        )
    if not specs:
        raise SystemExit("no prefixes were supplied (--prefixes and/or --prefix-files)")
    return specs


# --------------------------------------------------------------------------- #
# Cells and tasks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContinuationCell:
    """One (prefix, new-task member) pair; run for N epochs."""

    prefix: PrefixSpec
    member: str
    unit_path: Path
    family: str
    cross_family: bool | None  # None when the prefix has no seed family (external)


def build_continuation_cells(
    prefix_specs: list[PrefixSpec],
    seeds_path: str,
    members: list[str],
) -> list[ContinuationCell]:
    root = Path(seeds_path).resolve()
    if is_seed_member(root):
        units = {root.name: root}
    else:
        available = seed_members(root)
        units = {member: root / member for member in available if member in set(members)}
    missing = sorted(set(members) - set(units))
    if missing:
        raise SystemExit(f"--seeds includes unavailable members: {missing}")

    conflicts = [
        (spec.name, member)
        for spec in prefix_specs
        for member in units
        if spec.source_seed == member
    ]
    if conflicts:
        raise SystemExit(
            "a continuation's new task must come from a different seed than its "
            f"prefix; conflicting (prefix, seed) pairs: {conflicts}. Drop those "
            "seeds from --seeds or use other prefixes."
        )

    cells: list[ContinuationCell] = []
    for spec in prefix_specs:
        for member, unit_path in sorted(units.items()):
            family = unit_path.parent.name
            cross_family = None if spec.family is None else spec.family != family
            # Resolve the pivot descriptor NOW so an unmapped family aborts at plan
            # time rather than mid-generation.
            if cross_family is None or cross_family:
                seed_family_task_description(family)
            cells.append(ContinuationCell(
                prefix=spec,
                member=member,
                unit_path=unit_path,
                family=family,
                cross_family=cross_family,
            ))
    return cells


def continuation_task_name(treatment: str, cell: ContinuationCell, slug: str) -> str:
    model_last = slug.split("/")[-1]
    return f"continuation_{treatment}_{model_last}_{cell.member}_p{cell.prefix.name}"


def build_continuation_tasks(
    cells: list[ContinuationCell],
    *,
    treatment: str,
    run_label: str,
    condition: str,
    gate_model: str,
    judge: str | None,
    artifacts_root: Path | None,
    task_id_suffix: str | None = None,
    execution_metadata: dict | None = None,
) -> list:
    """One Inspect task per cell, sharing build_real_task with plain audits."""

    judge_model = resolve_judge(judge)
    tasks = []
    for cell in cells:
        prefix = cell.prefix
        unit_sp = expected_system_prompt(prefix.reasoning)
        target_slug = resolve_target(prefix.target_name)
        target_build = build_target(
            target_slug,
            reasoning_on=prefix.reasoning,
            effort=REASONING_EFFORT,
            # OpenAI rejects prompt_cache_key values longer than 64 chars.
            prompt_cache_key="environments-cont-" + stable_key(
                "continuation-target-v1",
                target_slug,
                prefix.reasoning,
                prefix.sha256,
                cell.member,
            )[:40],
        )
        descriptor = (
            None
            if cell.cross_family is False
            else seed_family_task_description(cell.family)
        )
        pivot = pivot_preamble(descriptor)
        spec = assemble_real_protocol(cell.unit_path, condition, unit_sp)
        record = continuation_record(
            treatment=treatment,
            prefix=prefix.record(),
            prefix_length=len(prefix.messages),
            boundary_index=prefix.boundary_index,
            pivot_preamble_text=pivot,
            opening_user_message=spec.opening_user_message,
            cross_family=cell.cross_family,
        )
        continuation = ContinuationRun(
            prefix_messages=prefix.messages,
            pivot_text=pivot,
            record=record,
        )
        tasks.append(build_real_task(
            target_name=prefix.target_name,
            target_build=target_build,
            unit_path=cell.unit_path,
            unit_sp=unit_sp,
            reasoning=prefix.reasoning,
            condition=condition,
            judge_model=judge_model,
            gate_model=gate_model,
            run_label=run_label,
            artifacts_root=artifacts_root,
            task_id_suffix=task_id_suffix,
            execution_metadata=execution_metadata,
            continuation=continuation,
            task_name=continuation_task_name(treatment, cell, target_slug),
            extra_metadata={
                "experiment": "continuation",
                "continuation_version": CONTINUATION_RECORD_VERSION,
                "treatment": treatment,
                "prefix_name": prefix.name,
                "prefix_sha256": prefix.sha256,
                "prefix_source": dict(prefix.payload["source"]),
                "prefix_target_name": prefix.target_name,
                "prefix_message_count": len(prefix.messages),
                "prefix_boundary_index": prefix.boundary_index,
                "prefix_synthesized_closer": prefix.synthesized_closer,
                "prefix_system_prompt_inserted": prefix.system_prompt_inserted,
                "pivot_preamble": pivot,
                "cross_family": cell.cross_family,
            },
        ))
    return tasks


def describe_plan(
    treatment: str, prefix_specs: list[PrefixSpec], cells: list[ContinuationCell]
) -> str:
    """Human-readable plan block for the pipeline's stdout (and --dry-run)."""

    lines = [f"treatment {treatment!r}: {len(prefix_specs)} prefix(es), "
             f"{len(cells)} cell(s)"]
    for spec in prefix_specs:
        source = spec.payload["source"]
        origin = (
            f"trajectory #{source.get('trajectory_id')} "
            f"(seed {source.get('seed')}, {source.get('run')})"
            if source.get("kind") == "trajectory"
            else f"external ({source.get('description') or 'no description'})"
        )
        flags = []
        if spec.synthesized_closer:
            flags.append("synthetic tool-result closer")
        if spec.system_prompt_inserted:
            flags.append("system prompt inserted")
        flag_note = f"  [{'; '.join(flags)}]" if flags else ""
        lines.append(
            f"  prefix {spec.name}: {origin}, target {spec.target_name} "
            f"[reasoning:{'on' if spec.reasoning else 'off'}], "
            f"{len(spec.messages)} msgs / {spec.char_count:,} chars{flag_note}"
        )
    for cell in cells:
        pivot_kind = (
            "plain pivot" if cell.cross_family is False
            else "pivot names the new work type"
        )
        lines.append(
            f"  cell: {cell.prefix.name} -> {cell.member} "
            f"({cell.family}; {pivot_kind})"
        )
    return "\n".join(lines)
