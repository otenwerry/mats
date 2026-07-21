"""Assemble every active pinned seed in every collection under seeds/.

Runs both conditions exactly as exp_rh_audit does at launch and reports prompt sizes.
Free: import + local reads only; no model or API calls.
"""

import sys
from pathlib import Path


PETRI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))

from exp_rh_audit import (  # noqa: E402
    assemble_pinned_seed,
    is_pinned_collection,
    pinned_collection_members,
    pinned_conditions,
)


SEEDS_ROOT = PETRI / "seeds"
collections = sorted(d for d in SEEDS_ROOT.iterdir() if is_pinned_collection(d))
assert collections, f"no pinned collections found under {SEEDS_ROOT}"

total = 0
for collection in collections:
    print(f"\n[{collection.name}]")
    for name in pinned_collection_members(collection):
        member = collection / name
        conditions = pinned_conditions(member)
        assert conditions, f"{collection.name}/{name}: no conditions"
        for condition in conditions:
            sample = assemble_pinned_seed(member, condition)
            assert sample.id == name
            assert sample.metadata["condition"] == condition
            assert "{{TASK}}" not in sample.input and "{{TOOLING}}" not in sample.input
            pinned_files = sample.input.count("<pinned_file ")
            print(
                f"{name:22s} cond={condition:8s} assembled {len(sample.input):6d} chars, "
                f"{pinned_files} pinned files"
            )
            total += 1

print(f"\nall {total} pinned seed-condition assemblies are clean")
