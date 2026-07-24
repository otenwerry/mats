"""Route and order judge rubrics for an audit from its seed directory.

Audit dimensions have three sources, loaded in this order:

1. ``dimensions/*.md`` -- temporary backwards-compatible global dimensions. These are
   the repository's current files and remain active until Owen moves them. ``README.md``
   is documentation, not a rubric, and is ignored at every scope.
2. ``dimensions/global/*.md`` -- the permanent global dimensions.
3. The most specific applicable seed scope. For a seed at
   ``seeds/p_hacking/checkout_redesign``, this is
   ``dimensions/p_hacking/checkout_redesign/*.md`` when that directory contains rubrics,
   otherwise ``dimensions/p_hacking/*.md``. The fallback lets a collection and a member
   run share one family-level rubric set.

Files are never loaded recursively inside one scope. ``archived/`` and
``rollback_only/`` remain special non-audit dimension directories. Duplicate dimension
names across the selected sources are rejected before an experiment can spend money.

After selecting the applicable files, ``dimensions/judge_order.json`` supplies one
explicit, repository-wide order. Dimensions absent from the current task are skipped,
but every active audit dimension on disk must occur exactly once in the order file. This
makes moving a rubric between global and seed-specific folders order-preserving.
"""

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from inspect_petri._judge.dimensions import JudgeDimension, _parse_dimension_file

from petri_paths import DIMENSIONS_DIR, PETRI_ROOT


SEEDS_ROOT = PETRI_ROOT / "seeds"
NON_AUDIT_SCOPE_NAMES = {"archived", "global", "rollback_only"}
JUDGE_ORDER_FILENAME = "judge_order.json"
NON_RUBRIC_MARKDOWN_FILENAMES = {"readme.md"}


@dataclass(frozen=True)
class AuditDimensionSet:
    """The concrete rubrics and provenance used by one audit task."""

    dimensions: tuple[JudgeDimension, ...]
    files: tuple[Path, ...]
    scope_dir: Path | None

    @property
    def names(self) -> list[str]:
        return [dimension.name for dimension in self.dimensions]


def _is_runnable_pinned_seed(directory: Path) -> bool:
    """Recognize both self-contained and composed pinned-seed layouts.

    This intentionally mirrors ``exp_rh_audit.is_pinned_seed_dir`` without importing
    that experiment module (which would create a circular dependency). Dimension routing
    is the low-level boundary shared by every experiment, so seed lookup belongs here.
    """

    if not directory.is_dir() or directory.name == "_shared":
        return False
    if (directory / "core.md").is_file() and (directory / "conditions").is_dir():
        return True
    shared = directory.parent / "_shared"
    return (
        (directory / "scenario.md").is_file()
        and (directory / "environment").is_dir()
        and (shared / "core.md").is_file()
        and (shared / "conditions").is_dir()
    )


@lru_cache(maxsize=None)
def resolve_seed_path(
    seed_name: str,
    stamped_seed_dir: str | None = None,
    *,
    seeds_root: Path = SEEDS_ROOT,
) -> Path:
    """Resolve one stored sample id to its concrete seed path.

    Current audit logs stamp ``seed_dir`` and therefore take the exact path. Older logs
    predate that field, so they fall back to a unique recursive match. Ambiguity fails
    loudly rather than letting a resample/rollback/continuation use another family's
    rubric or seed text.
    """

    root = Path(seeds_root)
    if stamped_seed_dir:
        candidate = Path(stamped_seed_dir)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise SystemExit(
                f"stamped seed_dir {stamped_seed_dir!r} is outside {root}"
            ) from exc
        if not candidate.exists():
            raise SystemExit(
                f"stamped seed_dir {stamped_seed_dir!r} does not exist under {root}"
            )
        return candidate

    markdown = sorted(root.glob(f"**/{seed_name}.md"))
    pinned = sorted(
        path for path in root.glob(f"**/{seed_name}") if _is_runnable_pinned_seed(path)
    )
    matches = [path.parent.resolve() for path in markdown] + [path.resolve() for path in pinned]
    # A path can theoretically be found through both layouts during an in-progress migration.
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise SystemExit(
            f"seed {seed_name!r} not found under {root} as either a .md seed or pinned directory"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"seed {seed_name!r} is ambiguous ({[str(path) for path in matches]}); "
            "use a trajectory whose log stamps seed_dir"
        )
    return matches[0]


def dimension_provenance(
    seed_path: str | Path,
    selected: AuditDimensionSet,
    *,
    seeds_root: Path = SEEDS_ROOT,
    dimensions_root: Path = DIMENSIONS_DIR,
    project_root: Path = PETRI_ROOT,
) -> dict:
    """Metadata every judged task stores for its one routed dimension set."""

    seed = Path(seed_path).resolve()
    try:
        seed_label = str(seed.relative_to(Path(seeds_root).resolve()))
    except ValueError:
        seed_label = str(seed)
    if selected.scope_dir is None:
        scope_label = None
    else:
        try:
            scope_label = str(selected.scope_dir.resolve().relative_to(Path(dimensions_root).resolve()))
        except ValueError:
            scope_label = str(selected.scope_dir)

    files = []
    for path in selected.files:
        try:
            files.append(str(path.resolve().relative_to(Path(project_root).resolve())))
        except ValueError:
            files.append(str(path))
    return {
        "seed_dir": seed_label,
        "dimension_scope": scope_label,
        "judge_dimensions": selected.names,
        "judge_dimension_files": files,
    }


def dimensions_for_seed(
    seed_path: str | Path,
    *,
    required: tuple[str, ...] = ("reward_hacking",),
    dimensions_root: Path = DIMENSIONS_DIR,
    seeds_root: Path = SEEDS_ROOT,
) -> AuditDimensionSet:
    """The single experiment-facing dimension resolver.

    All audit-like generation, resampling, continuation, rollback, and re-judging code
    should call this boundary instead of loading a global directory or importing a static
    list. ``required`` makes a missing outcome rubric fail before API spend.
    """

    selected = audit_dimension_set(
        seed_path, dimensions_root=dimensions_root, seeds_root=seeds_root
    )
    missing = [name for name in required if name not in selected.names]
    if missing:
        raise SystemExit(
            f"seed path {seed_path} is missing required judge dimensions {missing}; "
            f"resolved dimensions were {selected.names}"
        )
    return selected


def _is_rubric_markdown(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.casefold() == ".md"
        and path.name.casefold() not in NON_RUBRIC_MARKDOWN_FILENAMES
    )


def _markdown_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.iterdir() if _is_rubric_markdown(path)))


def _active_dimension_names(dimensions_root: Path) -> set[str]:
    """All audit dimension names across every global and seed-specific scope."""

    names = {file.stem for file in _markdown_files(dimensions_root)}
    names.update(file.stem for file in _markdown_files(dimensions_root / "global"))
    if dimensions_root.is_dir():
        for child in dimensions_root.iterdir():
            if not child.is_dir() or child.name in NON_AUDIT_SCOPE_NAMES:
                continue
            names.update(
                file.stem
                for file in child.rglob("*")
                if _is_rubric_markdown(file)
                and not set(file.relative_to(child).parts) & NON_AUDIT_SCOPE_NAMES
            )
    return names


def _judge_dimension_order(dimensions_root: Path) -> list[str]:
    """Load and validate the authoritative judge-schema field order."""

    path = dimensions_root / JUDGE_ORDER_FILENAME
    if not path.is_file():
        raise SystemExit(
            f"missing judge dimension order file: {path}. Add a JSON list containing "
            "every active audit dimension name in the desired judge order."
        )
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read judge dimension order from {path}: {exc}") from exc
    if not isinstance(raw, list) or not all(
        isinstance(name, str) and name.strip() == name and name for name in raw
    ):
        raise SystemExit(
            f"{path} must be a JSON list of non-empty dimension-name strings"
        )

    duplicates = sorted({name for name in raw if raw.count(name) > 1})
    if duplicates:
        raise SystemExit(f"duplicate names in {path}: {duplicates}")

    active_names = _active_dimension_names(dimensions_root)
    ordered_names = set(raw)
    missing = sorted(active_names - ordered_names)
    unknown = sorted(ordered_names - active_names)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing active dimensions {missing}")
        if unknown:
            details.append(f"unknown/inactive dimensions {unknown}")
        raise SystemExit(f"invalid judge dimension order in {path}: {'; '.join(details)}")
    return raw


def _seed_relative_path(seeds_path: str | Path, seeds_root: Path) -> Path:
    try:
        return Path(seeds_path).resolve().relative_to(seeds_root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"seed path {seeds_path} is outside {seeds_root}; cannot choose its dimension folder"
        ) from exc


def scoped_dimension_dir(
    seeds_path: str | Path,
    *,
    dimensions_root: Path = DIMENSIONS_DIR,
    seeds_root: Path = SEEDS_ROOT,
) -> Path | None:
    """Return the seed-specific rubric directory for one concrete seed path.

    The exact mirrored path wins when it contains rubrics. A nested member then falls
    back to its top-level seed-family directory. The returned fallback path may not exist
    yet; retaining it makes metadata and pre-run output show where rubrics should go.
    """

    relative = _seed_relative_path(seeds_path, seeds_root)
    if not relative.parts:
        return None

    exact = dimensions_root / relative
    if _markdown_files(exact):
        return exact

    family_name = relative.parts[0]
    family = dimensions_root / family_name
    if family_name not in NON_AUDIT_SCOPE_NAMES and _markdown_files(family):
        return family

    # Empty tracked directories (.gitkeep only) are useful during organization: prefer
    # the exact folder if it was explicitly created, then the family folder.
    if exact.is_dir():
        return exact
    if family_name not in NON_AUDIT_SCOPE_NAMES and family.is_dir():
        return family

    return exact


def audit_dimension_set(
    seeds_path: str | Path,
    *,
    dimensions_root: Path = DIMENSIONS_DIR,
    seeds_root: Path = SEEDS_ROOT,
    allow_empty: bool = False,
) -> AuditDimensionSet:
    """Load global + relevant seed-scoped rubrics for one audit task.

    Top-level ``dimensions/*.md`` files are treated as legacy globals so the repository
    keeps working while files are reorganized. Once those files are moved into
    ``dimensions/global/`` or a seed scope, no code change is needed.
    """

    scope = scoped_dimension_dir(
        seeds_path, dimensions_root=dimensions_root, seeds_root=seeds_root
    )
    source_dirs = [dimensions_root, dimensions_root / "global"]
    if scope is not None:
        source_dirs.append(scope)

    files_by_name: dict[str, Path] = {}
    dimensions: list[JudgeDimension] = []
    for directory in source_dirs:
        files = _markdown_files(directory)
        if not files:
            continue
        for file in files:
            previous = files_by_name.get(file.stem)
            if previous is not None:
                raise SystemExit(
                    f"duplicate judge dimension {file.stem!r}: {previous} and {file}. "
                    "Global and seed-specific dimension names must be unique."
                )
            files_by_name[file.stem] = file
        dimensions.extend(_parse_dimension_file(file) for file in files)

    if not dimensions and not allow_empty:
        locations = ", ".join(str(directory) for directory in source_dirs)
        raise SystemExit(
            "no audit judge dimensions found; expected .md rubrics in one or more of: "
            f"{locations}"
        )

    order = _judge_dimension_order(dimensions_root)
    selected_by_name = {dimension.name: dimension for dimension in dimensions}
    ordered_dimensions = [
        selected_by_name[name] for name in order if name in selected_by_name
    ]
    ordered_files = [files_by_name[dimension.name] for dimension in ordered_dimensions]
    return AuditDimensionSet(
        dimensions=tuple(ordered_dimensions), files=tuple(ordered_files), scope_dir=scope
    )


def audit_dimension_names_on_disk(
    *, dimensions_root: Path = DIMENSIONS_DIR,
) -> list[str]:
    """Union of active audit dimension names for viewer column ordering.

    Unlike one audit's routing, this scans every configured audit scope. Special-purpose
    ``archived`` and ``rollback_only`` trees are excluded.
    """

    return sorted(_active_dimension_names(dimensions_root))
