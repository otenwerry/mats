# RH localization highlights (per-run, checked into git)

One `{run_id}.json` per flagged run tells the viewer where the hack is and
whether it's *actually* reward hacking. Current (re-judge) schema:

```json
{
  "run_id": "...",
  "source": "judge_output" | "verdict_only",   // did the OLD judge have a report
  "reassessment": {                              // our stronger re-judge
    "verdict": "yes" | "questionable" | "no",    // is it ACTUAL reward hacking?
    "rationale": "2-4 sentences",
    "model": "claude-opus-4-8"
  },
  "summary": "one plain sentence: what the agent did and where",
  "report_md": "the OLD judge's markdown verdict (stricter bar; may over-flag)",
  "hack_turns": [463, 467, 469],                 // MARKED event indices (see turn_kinds)
  "turn_kinds": {                                 // per-turn split (exp_classify_turns.py):
    "463": {"kind": "context", "reason": "..."},  //   'hack' = direct hacking action (red ⚠)
    "467": {"kind": "hack", "reason": "..."}      //   'context' = notable, not a hack (gray 📌)
  },                                              //   unclassified turns default to 'hack';
                                                  //   /timing counts only 'hack' turns
  "quotes": ["exact substring", "..."],          // spans marked inline (red)
  "annotations": {                                // reviewer notes, by turn
    "467": [{"quote": "exact span", "note": "why it matters",
             "kind": "admission|reasoning|code|command|data|other",
             "matched": true}]
  },
  "model": "claude-opus-4-8",
  "train_label_trap": {                          // deterministic (flag_train_label_trap.py):
    "dataset": "Berkeley-Function-Calling",      //   trained on the benchmark's own eval pool
    "first_load_event": 14                       //   loaded under a misleading split="train"
  },
  "train_label_audit": {                         // Opus knowledge audit (exp_audit_train_label.py):
    "knowledge": "knew" | "misled" | "unclear",  //   did the agent KNOW it was eval data?
    "rationale": "...",
    "evidence": [{"turn": 11, "quote": "...", "note": "..."}],
    "model": "claude-opus-4-8"
  }
}
```
(The last two fields exist only on 🏷-flagged runs. Audit evidence is also
appended to `annotations` as `[label audit]` notes so it renders in the trace.)

`viewer.py` applies it automatically:
- **`reassessment.verdict`** → pill in the index (`RH: yes/?/no`) and a banner on
  the detail page with the rationale. yes=red, questionable=amber, no=green.
- **`hack_turns`** → red box + `⚠ hack turn` tag; jump between them with n/p.
- **`quotes`** → marked inline red (`<mark class="hack">`), confined to the hack
  turns, whitespace-tolerant / case-insensitive (`render.mark`).
- **`annotations`** → reviewer callouts below the relevant turn, styled clearly
  distinct from the trace (dashed amber box, kind label). A note may be
  quote-less (a general observation on that turn).
- **`source` / `report_md`** → the old judge's confidence + collapsible report.

## How these are produced

`exp_rejudge_hacks.py` (paid; Claude Opus) is the current producer: it reads the
old judge's reasoning (`judge_findings.parse`) then the WHOLE trace (full
reasoning/prose, big tool outputs clipped via `locate.condense_trace`), and
re-adjudicates against the agent's *actual* instruction (only the TEST split is
off-limits) — correcting the old judge's over-broad bar. It hand-picks spans
(favoring the model's own admissions) and annotations; quotes are verified to be
real substrings of their turn before being kept.

`exp_localize_hacks.py` (gpt-4o) is the earlier, lighter pass (no verdict, no
annotations) — superseded but kept for reference. Hand-authored files work too;
just match the schema. The deterministic legs (`judge_findings.py`,
`locate.py`) are free and reusable for other analyses.
