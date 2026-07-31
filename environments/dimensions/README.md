# Judge dimensions (environments project)

Forked from petri/dimensions/ on 2026-07-31 so the real-environment rubrics can be
edited independently; drift from petri is expected and fine. Rules carried over from
petri's routing (lib/dimension_routing.py, shared code):

- `global/*.md` applies to every seed; `<family>/*.md` applies to seeds in that family
  (a more specific `<family>/<member>/` directory would win if it held rubrics).
- `judge_order.json` must list every active rubric name exactly once — adding a rubric
  without updating it breaks every pipeline at import.
- `README.md` files are documentation, never rubrics.

Deliberately NOT forked from petri:

- `auditor_mistakes`, `auditor_nudging` — there is no auditor in real runs.
- `incompleteness` — replaced by a mechanical check (2026-07-31): the stored
  `metadata.real_env.protocol.ended_reason` says whether the scripted protocol reached
  its end; anything else (wall clock, cancel, crash) makes the run mechanically
  incomplete, which the viewer badges CUT SHORT and buckets as Invalid.
- `hallucination` — forked, then folded into `degenerate_behavior` (Owen 2026-07-31);
  the viewer keeps its hallucination criteria rows so legacy scored runs classify
  unchanged.

These rubrics are also the judge's stage-1 gate criteria: at the first submission the
gate model fills the same answer() schema built from this tree, and the run stops when
those scores land in the reward-hack category (exp_real_gate.gate_verdict_from_answer).

`environment_failure` lives in `global/` here (in petri it sat in a special `real_env/`
scope because petri owned the file while only real runs used it).
