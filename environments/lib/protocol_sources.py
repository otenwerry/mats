"""Environment-owned target protocol and seed discovery.

The real runner needs only the target-facing protocol: one global system prompt, a
family opening-message template, optional scripted follow-ups, and the member-specific
scenario fields.  Petri's auditor ``core.md`` and condition fragments are intentionally
not part of this module because no auditor model exists in the real environment.

All paths resolve under :mod:`environments/seeds`.  The loader also returns file hashes
so every run can stamp the exact source material it used.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from project_paths import ENVIRONMENTS_ROOT, SEEDS_ROOT, SYSTEM_PROMPT_PATH


SHARED_DIR_NAME = "_shared"
SCENARIO_FILENAME = "scenario.md"
USER_PROMPT_FILENAME = "USER_PROMPT.txt"
FOLLOW_UP_FILENAME = "FOLLOW_UP.txt"
FOLLOW_UP_FINALIZE_FILENAME = "FOLLOW_UP_FINALIZE.txt"

USER_PROMPT_PLACEHOLDER = "{{SEED_SPECIFIC_PROMPT}}"
SCENARIO_USER_PROMPT_MARK = "[USER_PROMPT]"
SCENARIO_TASK_MARK = "[TASK]"
SCENARIO_TOOLING_MARK = "[TOOLING]"

SUPPORTED_CONDITION = "allow"

THINKING_INSTRUCTION = (
    "Reason step by step inside `<thinking></thinking>` tags before taking actions. "
)


@dataclass(frozen=True)
class ScenarioFields:
    """The three explicit fields in one member's ``scenario.md``."""

    user_prompt: str
    task: str
    tooling: str
    source_path: Path


@dataclass(frozen=True)
class ProtocolSources:
    """Resolved, target-facing protocol for one real-environment member."""

    family: str
    member: str
    member_dir: Path
    system_prompt: str
    opening_user_message: str
    follow_up: str | None
    follow_up_finalize: str | None
    scenario: ScenarioFields
    source_files: tuple[Path, ...]

    @property
    def follow_up_user_messages(self) -> tuple[str, ...]:
        return tuple(
            text for text in (self.follow_up, self.follow_up_finalize) if text is not None
        )

    def provenance(self) -> dict:
        """Stored, queryable identity for every protocol source file."""

        return {
            "protocol_source": "environments",
            "protocol_family": self.family,
            "protocol_member": self.member,
            "protocol_source_files": [
                {
                    "path": _display_path(path),
                    "sha256": _sha256(path),
                }
                for path in self.source_files
            ],
        }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ENVIRONMENTS_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_seeds(path: Path, *, seeds_root: Path = SEEDS_ROOT) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(Path(seeds_root).resolve())
    except ValueError as exc:
        raise SystemExit(f"seed path {path} is outside {seeds_root}") from exc
    return resolved


def is_seed_member(path: str | Path, *, seeds_root: Path = SEEDS_ROOT) -> bool:
    """Whether ``path`` is one complete environments-owned member."""

    member = Path(path)
    if not member.is_dir() or member.name == SHARED_DIR_NAME:
        return False
    if not (
        (member / SCENARIO_FILENAME).is_file()
        and (member / "manifest.json").is_file()
        and (member / "workspace").is_dir()
    ):
        return False
    try:
        _inside_seeds(member, seeds_root=seeds_root)
    except SystemExit:
        return False
    return True


def seed_members(collection: str | Path, *, seeds_root: Path = SEEDS_ROOT) -> list[str]:
    """Sorted runnable member names immediately below one family directory."""

    root = _inside_seeds(Path(collection), seeds_root=seeds_root)
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if is_seed_member(child, seeds_root=seeds_root)
    )


def resolve_seeds(
    seed_dir: str | None,
    *,
    seeds_root: Path = SEEDS_ROOT,
) -> tuple[str, list[str]]:
    """Resolve ``--seed-dir`` without consulting Petri's seed tree.

    Real runs require a family or member name.  A family returns all immediate runnable
    members; a member returns itself as the sole choice.
    """

    if seed_dir is None or not seed_dir.strip():
        families = sorted(
            path.name
            for path in Path(seeds_root).iterdir()
            if path.is_dir() and seed_members(path, seeds_root=seeds_root)
        )
        raise SystemExit(
            f"--seed-dir is required; available environment families: {families}"
        )

    candidate = _inside_seeds(Path(seeds_root) / seed_dir, seeds_root=seeds_root)
    if is_seed_member(candidate, seeds_root=seeds_root):
        return str(candidate), [candidate.name]
    members = seed_members(candidate, seeds_root=seeds_root)
    if members:
        return str(candidate), members

    families = sorted(
        path.name
        for path in Path(seeds_root).iterdir()
        if path.is_dir() and seed_members(path, seeds_root=seeds_root)
    )
    raise SystemExit(
        f"--seed-dir={seed_dir!r} is not a runnable environments family or member; "
        f"available families: {families}"
    )


def resolve_condition(condition: str | None, _seeds_path: str | Path | None = None) -> str:
    """Resolve the real protocol's currently supported condition.

    The unused ``_seeds_path`` argument keeps migration call sites simple.  Petri's
    auditor-condition files are deliberately not read.
    """

    value = SUPPORTED_CONDITION if condition is None else condition.strip()
    if value != SUPPORTED_CONDITION:
        raise SystemExit(
            f"real environments currently support only --condition={SUPPORTED_CONDITION}; "
            f"got {value!r}"
        )
    return value


def resolve_reasoning(value: str | None) -> bool:
    """Resolve the target ``--reasoning=yes|no`` flag; absent means yes."""

    if value is None:
        return True
    normalized = value.strip().casefold()
    if normalized in {"yes", "on", "true", "1"}:
        return True
    if normalized in {"no", "off", "false", "0"}:
        return False
    raise SystemExit(f"--reasoning must be yes|no, got {value!r}")


def reject_retired_fixed_system_prompt_flag(
    args: Sequence[str] | None = None,
) -> None:
    """Fail if the retired ``--fixed_sp`` switch appears."""

    actual = sys.argv if args is None else args
    if any(argument.startswith("--fixed_sp") for argument in actual):
        raise SystemExit(
            "--fixed_sp was removed: environments always reads "
            "seeds/SYSTEM_PROMPT.txt. Drop the flag."
        )


def global_system_prompt(*, path: Path = SYSTEM_PROMPT_PATH) -> str:
    """Read the environment-owned global target prompt, failing on missing/empty."""

    if not path.is_file():
        raise SystemExit(f"global target system prompt is missing: {path}")
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"global target system prompt is empty: {path}")
    return text


def strip_thinking_instruction(system_prompt: str) -> str:
    """Remove the prompted scratchpad line when native model reasoning is enabled."""

    if THINKING_INSTRUCTION not in system_prompt:
        print(
            "  !! WARNING: reasoning is ON but the '<thinking> tags' instruction was "
            "not found in environments/seeds/SYSTEM_PROMPT.txt; using it unchanged"
        )
        return system_prompt
    return system_prompt.replace(THINKING_INSTRUCTION, "")


def parse_scenario(path: str | Path) -> ScenarioFields:
    """Parse one strict ``[USER_PROMPT]`` / ``[TASK]`` / ``[TOOLING]`` file."""

    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"scenario source is missing: {source}")
    text = source.read_text()
    marks = (
        SCENARIO_USER_PROMPT_MARK,
        SCENARIO_TASK_MARK,
        SCENARIO_TOOLING_MARK,
    )
    counts = {mark: text.count(mark) for mark in marks}
    if any(counts[mark] != 1 for mark in marks):
        raise SystemExit(
            f"{source} must contain each scenario marker exactly once; got {counts}"
        )
    positions = [text.index(mark) for mark in marks]
    if positions != sorted(positions):
        raise SystemExit(f"{source} scenario markers must appear in this order: {marks}")

    user_tail = text.split(SCENARIO_USER_PROMPT_MARK, 1)[1]
    user_prompt, task_tail = user_tail.split(SCENARIO_TASK_MARK, 1)
    task, tooling = task_tail.split(SCENARIO_TOOLING_MARK, 1)
    values = {
        "user_prompt": user_prompt.strip(),
        "task": task.strip(),
        "tooling": tooling.strip(),
    }
    empty = [name for name, value in values.items() if not value]
    if empty:
        raise SystemExit(f"{source} has empty scenario fields: {empty}")
    return ScenarioFields(source_path=source, **values)


def _read_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"protocol source exists but is empty: {path}")
    return text


def load_protocol_sources(
    member_path: str | Path,
    *,
    seeds_root: Path = SEEDS_ROOT,
) -> ProtocolSources:
    """Load and compose the complete target-facing protocol for one member."""

    member = _inside_seeds(Path(member_path), seeds_root=seeds_root)
    if not is_seed_member(member, seeds_root=seeds_root):
        raise SystemExit(
            f"{member} is not a runnable environment member (needs scenario.md, "
            "manifest.json, and workspace/)"
        )
    family = member.parent.name
    shared = member.parent / SHARED_DIR_NAME
    scenario_path = member / SCENARIO_FILENAME
    prompt_path = shared / USER_PROMPT_FILENAME
    follow_up_path = shared / FOLLOW_UP_FILENAME
    finalize_path = shared / FOLLOW_UP_FINALIZE_FILENAME
    readme_path = member / "workspace" / "README.md"

    if not prompt_path.is_file():
        raise SystemExit(f"family opening-prompt template is missing: {prompt_path}")
    if not readme_path.is_file():
        raise SystemExit(f"workspace task specification is missing: {readme_path}")

    scenario = parse_scenario(scenario_path)
    template = prompt_path.read_text().strip()
    count = template.count(USER_PROMPT_PLACEHOLDER)
    if count != 1:
        raise SystemExit(
            f"{prompt_path} must contain {USER_PROMPT_PLACEHOLDER} exactly once; got {count}"
        )
    opening = template.replace(USER_PROMPT_PLACEHOLDER, scenario.user_prompt).strip()
    follow_up = _read_optional(follow_up_path)
    follow_up_finalize = _read_optional(finalize_path)
    if follow_up_finalize is not None and follow_up is None:
        raise SystemExit(
            f"{finalize_path} cannot exist without the preceding {follow_up_path}"
        )

    system_prompt_path = Path(seeds_root) / SYSTEM_PROMPT_PATH.name
    source_files = [system_prompt_path, prompt_path, scenario_path, readme_path]
    if follow_up is not None:
        source_files.append(follow_up_path)
    if follow_up_finalize is not None:
        source_files.append(finalize_path)

    return ProtocolSources(
        family=family,
        member=member.name,
        member_dir=member,
        system_prompt=global_system_prompt(path=system_prompt_path),
        opening_user_message=opening,
        follow_up=follow_up,
        follow_up_finalize=follow_up_finalize,
        scenario=scenario,
        source_files=tuple(source_files),
    )


def protocol_source_plan(protocols: Iterable[ProtocolSources]) -> list[dict]:
    """Free pre-run summary suitable for printing or a dry-run manifest."""

    return [protocol.provenance() for protocol in protocols]


# Narrow migration aliases for the existing task builder.  They preserve its call shape
# while changing the filesystem authority to environments/seeds.  New code should prefer
# the clearer names above.
def is_pinned_seed_dir(path: str | Path) -> bool:
    return is_seed_member(path)


def is_pinned_collection(path: str | Path) -> bool:
    return bool(seed_members(path)) and not is_seed_member(path)


def pinned_collection_members(path: str | Path) -> list[str]:
    return seed_members(path)


def reject_fixed_sp_flag(args: Sequence[str] | None = None) -> None:
    reject_retired_fixed_system_prompt_flag(args)


def _parse_scenario(path: str | Path) -> tuple[str, str, str]:
    fields = parse_scenario(path)
    return fields.task, fields.tooling, fields.user_prompt


def _pinned_sources(path: str | Path) -> dict[str, Path | None]:
    """Migration view of local target-facing sources; no auditor files are invented."""

    member = _inside_seeds(Path(path))
    shared = member.parent / SHARED_DIR_NAME
    return {
        "core": None,
        "conditions": None,
        "user_prompt": shared / USER_PROMPT_FILENAME,
        "follow_up": shared / FOLLOW_UP_FILENAME,
        "follow_up_finalize": shared / FOLLOW_UP_FINALIZE_FILENAME,
        "env": member / "environment",
        "scenario": member / SCENARIO_FILENAME,
    }
