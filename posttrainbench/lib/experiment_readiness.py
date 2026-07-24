"""Central safety block for paid PostTrainBench experiments.

Free viewing, reconstruction, and dry-run planning remain available. Every paid
PTB entry point must call ``require_paid_experiments_ready`` immediately before
its first paid setup or API/CLI call.
"""

from __future__ import annotations


PAID_EXPERIMENTS_READY = False

_BLOCK_MESSAGE = """
========================================================================
POSTTRAINBENCH PAID EXPERIMENTS ARE NOT READY — RUN BLOCKED BEFORE SPEND

The unresolved choice is:
  * use the original Claude Code version for closer trajectory fidelity, but
    lose the verified one-hour prompt-cache setup; or
  * use Claude Code >= 2.1.108 for verified caching, but change the scaffold
    version that supplies system instructions and tools.

OpenCode prompt-cache verification is also not implemented.

Do not bypass this by selecting --prefix-cache=off. Decide the experiment
protocol first, then update posttrainbench/lib/experiment_readiness.py and the
PTB documentation together.

See:
  posttrainbench/README.md
  agent-notes/PROMPT_CACHING.md
========================================================================
""".strip()


def require_paid_experiments_ready() -> None:
    """Refuse paid PTB work until the protocol decision is explicitly resolved."""
    if not PAID_EXPERIMENTS_READY:
        raise SystemExit(_BLOCK_MESSAGE)
