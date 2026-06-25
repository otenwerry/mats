"""Single source of truth for every filesystem path the petri code uses.

All resource/data paths are anchored to the petri/ project root (this file's
grandparent: lib/ -> petri/), NOT to each module's own location, so the core modules
can live in lib/ and the standalone tools in tools/ without their paths breaking.

  petri/                  <- PETRI_ROOT
    lib/petri_paths.py    <- __file__
  ../.env  (mats/.env)    <- ENV_FILE
  ../../mats-local/petri  <- DATA  (gitignored eval logs, viewer, sidecars)
"""
from pathlib import Path

PETRI_ROOT = Path(__file__).resolve().parent.parent          # petri/
ENV_FILE = PETRI_ROOT.parent / ".env"                         # mats/.env (API keys)
DATA = PETRI_ROOT.parent.parent / "mats-local" / "petri"      # gitignored data root
LOGS = DATA / "logs"
OUT = DATA / "viewer"
DIMENSIONS_DIR = PETRI_ROOT / "dimensions"                    # judge dimension rubrics
SEED_DIRS = [PETRI_ROOT / "seeds"]                            # audit seed scenarios
