"""Filesystem paths for the environments project (the sibling of petri/lib/petri_paths.py).

The real-environment runs keep their own data root so nothing mixes with petri's: separate
run dirs, separate viewer output, separate annotations + trajectory-id registry, separate
load cache and overlay sidecars (see viewer_load.set_data_root). Code and pinned seed
prose still come from petri -- only DATA is ours.
"""

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator

ENVIRONMENTS_ROOT = Path(__file__).resolve().parent.parent   # mats/environments/
MATS_ROOT = ENVIRONMENTS_ROOT.parent                          # mats/
PETRI_ROOT = MATS_ROOT / "petri"
ENV_FILE = MATS_ROOT / ".env"                                 # shared API keys

DATA = MATS_ROOT.parent / "mats-local" / "environments"       # gitignored data root
LOGS = DATA / "logs"
OUT = DATA / "viewer"
ANN_FILE = DATA / "annotations.json"

SANDBOX_DIR = ENVIRONMENTS_ROOT / "sandbox"
ENV_SEEDS = ENVIRONMENTS_ROOT / "seeds"
# This project's OWN judge rubrics (forked from petri/dimensions/ 2026-07-31; free to
# drift). Routed through petri's shared dimension_routing with this root.
ENV_DIMENSIONS = ENVIRONMENTS_ROOT / "dimensions"


@contextmanager
def annotation_file_lock(data_root: Path = DATA) -> Iterator[None]:
    """Serialize annotation reads/writes across concurrently running pipelines."""
    lock_path = Path(data_root) / ".annotations.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
