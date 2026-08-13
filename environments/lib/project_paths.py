"""Paths owned by the real-environments project.

This module deliberately has no Petri paths.  Importing it is safe in free tooling: it
does not read secrets, create directories, or import Inspect.
"""

from __future__ import annotations

from pathlib import Path


ENVIRONMENTS_ROOT = Path(__file__).resolve().parent.parent
MATS_ROOT = ENVIRONMENTS_ROOT.parent
SUPERMATS_ROOT = MATS_ROOT.parent

# API clients may load this explicitly at an experiment endpoint.  Helpers should not
# read it merely because they were imported.
ENV_FILE = MATS_ROOT / ".env"

SEEDS_ROOT = ENVIRONMENTS_ROOT / "seeds"
SYSTEM_PROMPT_PATH = SEEDS_ROOT / "SYSTEM_PROMPT.txt"

DATA_ROOT = SUPERMATS_ROOT / "mats-local" / "environments"
LOGS_ROOT = DATA_ROOT / "logs"
VIEWER_ROOT = DATA_ROOT / "viewer"
CONTINUATION_PREFIXES_ROOT = DATA_ROOT / "continuation_prefixes"
ANNOTATIONS_PATH = DATA_ROOT / "annotations.json"
