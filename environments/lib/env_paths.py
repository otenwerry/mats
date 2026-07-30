"""Filesystem paths for the environments project (the sibling of petri/lib/petri_paths.py).

The real-environment runs keep their own data root so nothing mixes with petri's: separate
run dirs, separate viewer output, separate annotations + trajectory-id registry, separate
load cache and overlay sidecars (see viewer_load.set_data_root). Code and pinned seed
prose still come from petri -- only DATA is ours.
"""

from pathlib import Path

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
