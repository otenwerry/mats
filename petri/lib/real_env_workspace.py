"""Real-environment workspace plumbing: host-side trees -> sandbox /workspace, plus
in-sandbox snapshots and end-of-run artifact extraction.

Each real-env seed keeps its materialized tree at seeds/<family>/<member>/real_env/:
  workspace/    exactly what lands in the sandbox at /workspace (committed bytes)
  answers/      host-side canon (e.g. the test-label key) for the grader/gate; NEVER
                shipped into the sandbox beyond what workspace/ itself contains
  manifest.json sha256 provenance written by the envgen generator

Artifact extraction is capped per file (ARTIFACT_CAP_BYTES) with explicit truncated
flags stored alongside — capped copies go into sample metadata, full-fidelity copies to
a host-side sidecar dir next to the run's logs (surface-lossy rule: nothing is silently
dropped)."""

from __future__ import annotations

import posixpath
from pathlib import Path

from inspect_ai.util import sandbox

WORKSPACE_DIR = "/workspace"
MODELS_SUBDIR = "models"
# Per-file cap for artifact copies stored in sample metadata. Full copies go to the
# sidecar dir; the metadata copy carries an explicit "truncated" flag when capped.
ARTIFACT_CAP_BYTES = 200_000
# Extensions extracted as text artifacts when created/modified during the run.
ARTIFACT_SUFFIXES = {".py", ".md", ".csv", ".txt", ".json"}


def real_env_dir(seed_path: str | Path) -> Path:
    return Path(seed_path) / "real_env"


def has_real_env(seed_path: str | Path) -> bool:
    return (real_env_dir(seed_path) / "workspace").is_dir()


def answers_dir(seed_path: str | Path) -> Path:
    return real_env_dir(seed_path) / "answers"


def workspace_files(seed_path: str | Path) -> dict[str, str]:
    """Sample.files mapping: absolute sandbox path -> absolute host path.

    .gitkeep placeholders are skipped (they exist only so git keeps empty dirs);
    their parent dirs are reported by ensure_dirs() and created by the solver."""
    ws = real_env_dir(seed_path) / "workspace"
    if not ws.is_dir():
        raise SystemExit(
            f"{seed_path} has no real_env/workspace/ tree; generate it with the "
            "matching envgen/gen_*.py script before running a real audit"
        )
    mapping: dict[str, str] = {}
    for f in sorted(ws.rglob("*")):
        if f.is_file() and f.name != ".gitkeep":
            rel = f.relative_to(ws)
            mapping[posixpath.join(WORKSPACE_DIR, rel.as_posix())] = str(f)
    if not mapping:
        raise SystemExit(f"{ws} is empty")
    return mapping


def ensure_dirs(seed_path: str | Path) -> list[str]:
    """Sandbox dirs that must exist but ship empty (parents of .gitkeep files)."""
    ws = real_env_dir(seed_path) / "workspace"
    return sorted(
        posixpath.join(WORKSPACE_DIR, f.parent.relative_to(ws).as_posix())
        for f in ws.rglob(".gitkeep")
    )


# --------------------------------------------------------------------------- #
# in-sandbox operations (call only inside a running sample)
# --------------------------------------------------------------------------- #

async def create_empty_dirs(seed_path: str | Path) -> None:
    dirs = ensure_dirs(seed_path)
    if dirs:
        await sandbox().exec(["mkdir", "-p", *dirs])


def parse_sha_listing(stdout: str) -> dict[str, str]:
    """`sha256sum` output -> {sandbox path: sha256}."""
    snapshot: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, path = line.partition("  ")
        if len(digest) == 64 and path:
            snapshot[path] = digest
    return snapshot


async def workspace_snapshot() -> dict[str, str]:
    """{path: sha256} for every file under /workspace, excluding models/ (weights are
    large and their identity is captured by models_listing instead)."""
    result = await sandbox().exec(
        ["bash", "-lc",
         f"find {WORKSPACE_DIR} -type f -not -path '{WORKSPACE_DIR}/{MODELS_SUBDIR}/*' "
         "-print0 | xargs -0 -r sha256sum"],
        timeout=120,
    )
    if not result.success:
        raise RuntimeError(f"workspace snapshot failed: {result.stderr[:500]}")
    return parse_sha_listing(result.stdout)


async def models_listing() -> list[dict]:
    """Name + size of everything under /workspace/models/ (no content extraction)."""
    result = await sandbox().exec(
        ["bash", "-lc",
         f"find {WORKSPACE_DIR}/{MODELS_SUBDIR} -type f -printf '%s %p\\n' 2>/dev/null || true"],
        timeout=60,
    )
    listing = []
    for line in result.stdout.splitlines():
        size, _, path = line.strip().partition(" ")
        if size.isdigit() and path:
            listing.append({"path": path, "bytes": int(size)})
    return listing


async def read_artifact(path: str, cap: int = ARTIFACT_CAP_BYTES) -> dict:
    """One artifact file -> {path, content, truncated, bytes} with the cap surfaced."""
    result = await sandbox().exec(
        ["bash", "-lc", f"wc -c < {_shquote(path)} && head -c {cap} {_shquote(path)}"],
        timeout=60,
    )
    if not result.success:
        return {"path": path, "content": None, "truncated": False,
                "error": (result.stderr or "unreadable")[:200]}
    first_line, _, rest = result.stdout.partition("\n")
    try:
        size = int(first_line.strip())
    except ValueError:
        size = len(rest)
    return {
        "path": path,
        "content": rest,
        "bytes": size,
        "truncated": size > cap,
    }


async def extract_artifacts(
    start_snapshot: dict[str, str],
    end_snapshot: dict[str, str],
    always_include: tuple[str, ...] = (),
) -> dict:
    """Everything the run created or modified (by snapshot diff), as capped copies.

    always_include: workspace-relative deliverable names (e.g. "RESULT.md") pulled even
    if unchanged, so the grader/judge always see the final deliverables."""
    changed = sorted(
        path for path, digest in end_snapshot.items()
        if start_snapshot.get(path) != digest
    )
    wanted: list[str] = []
    for path in changed:
        if Path(path).suffix.lower() in ARTIFACT_SUFFIXES:
            wanted.append(path)
    for name in always_include:
        path = posixpath.join(WORKSPACE_DIR, name)
        if path in end_snapshot and path not in wanted:
            wanted.append(path)
    skipped = [p for p in changed if p not in wanted]
    files = [await read_artifact(p) for p in wanted]
    return {
        "files": files,
        "changed_paths": changed,
        "skipped_non_text": skipped,   # surfaced, not silently dropped
        "models": await models_listing(),
    }


def _shquote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"
