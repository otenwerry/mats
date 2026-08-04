# Environment judge instructions

All human-written prompt instructions for the environments-owned structured judge live
in this directory. `overall_instructions/` and `output_instructions/` each contain one
file per family. `judge_dimensions/` contains the dimension rubrics. The judge code
loads these files in a fixed order, while `lib/environment_judge/schema.py` defines the
executable stored output types.

This directory does not use Petri's 1–10 reward-hacking format. Binary answers are
booleans. ML uses one or more behavior labels, p-hacking uses exactly one results-based
label, and ML deception is null when no delivered hack label occurred. Every result also
has a required judge summary, combined per-dimension justification, and chronological
highlights. P-hacking `other` is deliberately outside the hack/non-hack split; p-hacking
has no `not_delivered` or deception field.
