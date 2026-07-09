"""Validate the six new pinned seed dirs: assemble each (both conditions) exactly the
way exp_rh_audit does at launch, and report sizes. Free: import + local reads only.
"""
import sys
from pathlib import Path

_petri = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from exp_rh_audit import assemble_pinned_seed, is_pinned_seed_dir, pinned_conditions

ROOT = _petri / "seeds" / "ml_training_data_misuse"
SEEDS = ["youtube_comments", "clinical_notes", "fraud_detection", "review_sentiment",
         "search_ranking", "spam_detection", "support_routing"]

for name in SEEDS:
    d = ROOT / name
    assert is_pinned_seed_dir(d), f"{name}: NOT detected as pinned"
    conds = pinned_conditions(d)
    assert conds == ["allow", "correct"], f"{name}: unexpected conditions {conds}"
    sp = (d / "SYSTEM_PROMPT.txt").read_text()
    up = (d / "USER_PROMPT.txt").read_text()
    for c in conds:
        s = assemble_pinned_seed(d, c)
        assert s.id == name and s.metadata["condition"] == c
        n_files = s.input.count("<pinned_file ")
        print(f"{name:18s} cond={c:8s} assembled {len(s.input):6d} chars, "
              f"{n_files} pinned files, SP {len(sp)} chars, UP {len(up)} chars")
print("\nall seeds assemble cleanly")
