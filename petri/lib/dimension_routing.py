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
