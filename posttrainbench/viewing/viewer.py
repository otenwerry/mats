"""Local web viewer for the PostTrainBench agent trajectories.

The data is NOT in this repo — it lives in the sibling `mats-local` tree
(big files, kept off GitHub). The dataset's provider already parsed every
trajectory into `viewer_data/`:

  viewer_data/index.json            -- manifest: one row per task (976), with
                                       all metadata + the contamination /
                                       disallowed-model judge verdicts.
  viewer_data/{run_id}.json         -- one task: events[], summary, metrics,
                                       judgements, error_log, ...
  viewer_data/{run_id}.workspace.json -- the agent's final workspace files.

So unlike the malt viewer we don't reconstruct anything — we just render.

Every adjudicated run has a `highlights/{run_id}.json` whose stamped `final`
block (written by judging/finalize.py — the ONLY implementation of the
judgement-layer cascade) carries the canonical verdict / label /
first-hack-event / training counts. This module renders those conclusions —
it derives none of them. The accumulated judging layers (reassessment,
audits, final_judge, story) render as provenance on the run page; quotes get
<mark>ed inside the trace (see render.mark).

Usage:  uv run python mats/posttrainbench/viewing/viewer.py   then open http://127.0.0.1:5001
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "utilities"))  # shared modules (paths, locate)

import json
import os
import re
import time
from pathlib import Path

from flask import Flask, abort, render_template_string, request

import locate
import paths
import render as rnd

# all data/output locations come from the shared paths module (handles the
# PTB_DATA override and the repo-relative layout)
DATA = paths.VIEWER_DATA
HIGHLIGHTS = paths.HIGHLIGHTS

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Data access                                                                 #
# --------------------------------------------------------------------------- #
def load_index() -> list[dict]:
    path = DATA / "index.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\nExpected the PostTrainBench viewer_data in "
            f"{DATA}. Set PTB_DATA to point at it, or download the dataset to "
            "mats-local/posttrainbench/ (see the dataset README).")
    with open(path) as f:
        return json.load(f)["runs"]


# Supplementary ROLLBACK-experiment runs are our own data, kept in mats-local
# alongside the main dataset (off github): mats-local/rollback/viewer_data/
# <run_id>.json, each with an index_row + events. We append them so they render
# alongside the originals. Mirrors rollback/config.ROLLBACK_VIEWER_DATA.
ROLLBACK_DATA = _Path(os.environ.get(
    "PTB_ROLLBACK_LOCAL", paths.RAW.parent / "rollback")) / "viewer_data"
ROLLBACK_RUNNING = ROLLBACK_DATA.parent / "running_rollouts.json"
# Per-entry registry dir: each in-flight (trajectory, condition) launch drops one
# small json here (unique filename), so concurrent launchers never race on a
# single file. The launcher (exp_rollout_batch.sh) writes them; the viewer reads
# the whole dir. Stale entries (a killed launch) age out via RUNNING_TTL_SEC.
ROLLBACK_RUNNING_DIR = ROLLBACK_DATA.parent / "running"
RUNNING_TTL_SEC = int(os.environ.get("PTB_RUNNING_TTL_SEC", str(18 * 3600)))
ROLLBACK_MANIFEST = _Path(__file__).resolve().parents[1] / "rollback" / "trajectory_manifest.json"
# Manually-parked trajectories (run_id -> reason). Same committed file the
# rollback config reads (we can't import config here); read fresh per call so a
# defer/un-defer shows up without a viewer restart.
ROLLBACK_MANUAL_DEFER = ROLLBACK_MANIFEST.parent / "manual_defer.json"

# The /rollback overview groups trajectory rows by the agent (engine) that
# produced the original run, in this fixed display order — one block per engine.
# Keyed on the manifest `agent` field (counts over the 30-run manifest:
# opencode 8, claude 10, claude_non_api 4, codex 7, qwen3max 1).
ROLLBACK_ENGINE_ORDER = ["opencode", "claude", "claude_non_api", "codex", "qwen3max"]
ROLLBACK_ENGINE_LABEL = {
    "opencode": "OpenCode",
    "claude": "Claude API",
    "claude_non_api": "Claude Code",
    "codex": "Codex",
    "qwen3max": "Qwen",
}
# Engines that use the native Claude CLI to resume. That resume path does NOT
# feed the reconstructed pre-cut history back to the model (verified 2026-06-18:
# resumes arrive with ~17k input tokens vs a ~200k+ rebuilt history), so the
# agent restarts the task instead of continuing — never a faithful rollback.
# This is structural to the Claude scaffold, so we taint EVERY Claude run. Runs
# that carry their own measured per-run taint keep it; the rest get this generic
# note as a fallback (see rollback_page).
ROLLBACK_CLAUDE_ENGINES = ("claude", "claude_non_api")
ROLLBACK_CLAUDE_COLD_START_TAINT = (
    "⚠ Resume cold-start: the Claude native-CLI resume did not feed the "
    "reconstructed pre-cut history to the model, so the agent restarted the task "
    "instead of continuing — not a faithful rollback continuation. Structural to "
    "the Claude scaffold; applied as a blanket flag to every Claude run. (This "
    "run lacks the per-run resume-token measurement the other Claude runs carry.)"
)


def load_manual_defer() -> dict:
    if not ROLLBACK_MANUAL_DEFER.exists():
        return {}
    try:
        raw = json.loads(ROLLBACK_MANUAL_DEFER.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_rollback_manifest() -> list[dict]:
    """The 30 adjudicated reward-hack source trajectories.

    The rollback page is a progress tracker over this fixed set, so it should
    show an original trajectory even before any rollout has been run for it.
    """
    if not ROLLBACK_MANIFEST.exists():
        return []
    try:
        rows = json.loads(ROLLBACK_MANIFEST.read_text())
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def load_running_rollbacks() -> list[dict]:
    """In-flight rollback rollouts, written by the launcher (exp_rollout_batch.sh)
    at launch — one per (trajectory, condition). Read from the per-entry registry
    dir (running/*.json) plus the legacy single-file list, for back-compat.

    These are deliberately separate from viewer_data because they are not
    completed traces and should not be linkable. The /rollback page suppresses a
    running row once a real viewer_data row exists for the same result_dir OR the
    same (trajectory, condition); stale entries (a killed launch) age out via
    RUNNING_TTL_SEC so the page can't show phantom 'running' rows forever.
    """
    rows: list[dict] = []
    if ROLLBACK_RUNNING.exists():
        try:
            legacy = json.loads(ROLLBACK_RUNNING.read_text())
            if isinstance(legacy, list):
                rows += legacy
        except json.JSONDecodeError:
            pass
    if ROLLBACK_RUNNING_DIR.exists():
        for p in sorted(ROLLBACK_RUNNING_DIR.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
    now = time.time()
    fresh = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        la = r.get("launched_at")
        if isinstance(la, (int, float)) and (now - la) > RUNNING_TTL_SEC:
            continue   # stale: a launch that never produced a result (aged out)
        fresh.append(r)
    return fresh


def load_rollback_runs(orig_acc: dict | None = None) -> list[dict]:
    orig_acc = orig_acc or {}
    out = []
    if ROLLBACK_DATA.exists():
        for p in sorted(ROLLBACK_DATA.glob("rollback_*.json")):
            try:
                d = json.loads(p.read_text())
                row = d["index_row"]
                # DEBUG_ runs were invalidated by an eval-tooling bug; never
                # surface them anywhere (index, /rollback, /run).
                if "__rollback__DEBUG_" in (row.get("experiment") or ""):
                    continue
                meta = d.get("meta") or {}
                src = meta.get("source_trajectory")
                row["orig_run_id"] = src
                # benchmark score of the ORIGINAL (pre-rollback) run, for the
                # "score (original)" display in the index
                row["orig_accuracy"] = orig_acc.get(src)
                row["rb_label"] = meta.get("label") or ""
                row["rb_empty"] = bool(meta.get("empty_continuation"))
                row["rb_ncont"] = meta.get("n_continuation_events")
                row["rb_result_dir"] = meta.get("result_dir")
                row["rb_tainted"] = meta.get("tainted")   # manual taint note, or None
                row["rb_failure_category"] = meta.get("failure_category")
                row["rb_failure_mode"] = meta.get("failure_mode")
                row["rb_failure_summary"] = meta.get("failure_summary")
                # Each caveat is {tag, detail}: `tag` is a short, self-explanatory
                # label shown inline on the row; `detail` is the full explanation
                # shown on hover.
                caveats = []
                wa = meta.get("workspace_audit") or {}
                flagged = wa.get("flagged_modified_after_cut") or []
                if flagged:
                    caveats.append({
                        "tag": "post-cut file edits",
                        "detail": (
                            f"{len(flagged)} file(s) existed at the cut but the trace shows "
                            "them edited again AFTER the cut; the backward rebuild only has "
                            "the final workspace, so the kept copy is that later (post-cut) "
                            "version — its contents may not match the true cut-time state "
                            f"(slightly unfaithful resume workspace): {', '.join(flagged)}"),
                    })
                for c in meta.get("session_caveats") or []:
                    s = str(c)
                    # codex's header-only apply_patch reconstruction affects ~every
                    # codex trajectory (structural, not per-run) — noted once on the
                    # page legend instead of flagged on every codex row.
                    if s.startswith("codex_apply_patch_history_lossy"):
                        continue
                    # "ran fully but produced no scoreable model" — the missing score
                    # already shows as acc=None in the score column; not a caveat.
                    if s.startswith("Rollout flagged empty/failed"):
                        continue
                    caveats.append({"tag": "caveat", "detail": s})
                row["rb_caveats"] = caveats
                out.append(row)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
    return out


_INDEX_RUNS = load_index()
RUNS = _INDEX_RUNS + load_rollback_runs(
    {r["run_id"]: r.get("accuracy") for r in _INDEX_RUNS})
INDEX = {r["run_id"]: i for i, r in enumerate(RUNS)}

def load_highlight_meta() -> dict[str, dict]:
    """{run_id: meta} for every adjudicated run (a highlights/{run_id}.json
    exists). The verdict/label/first-hack all come from the run's stamped
    `final` block — the canonical conclusion written by judging/finalize.py
    (the ONLY implementation of the judgement-layer cascade; see JUDGING.md).
    Read once at startup so the index needs no per-row file read."""
    out: dict[str, dict] = {}
    if not HIGHLIGHTS.exists():
        return out
    for p in HIGHLIGHTS.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        fin = d.get("final")
        if not d.get("run_id") or not fin:
            continue
        out[d["run_id"]] = {
            "old_judge_source": d.get("old_judge_source", ""),
            "verdict": (d.get("reassessment") or {}).get("verdict", ""),
            "trap": bool(d.get("train_label_trap")),
            "knowledge": (d.get("train_label_audit") or {}).get("knowledge", ""),
            "truncated": d.get("judge_input_truncated"),
            "everdict": fin.get("verdict", ""),
            "label": fin.get("label", ""),
            "is_reward_hack": bool(fin.get("is_reward_hack")),
            "first_hack_event": fin.get("first_hack_event"),
            "trainings_before_first_hack": fin.get("trainings_before_first_hack"),
            "trainings_total": fin.get("trainings_total"),
        }
    return out


HL_META = load_highlight_meta()
HL_SOURCES = {rid: m["old_judge_source"] for rid, m in HL_META.items()}
HL_VERDICT = {rid: m["verdict"] for rid, m in HL_META.items()}


def load_api_scan() -> dict:
    """First-pass external-API candidates (judging/scan_api_usage.py)."""
    p = HIGHLIGHTS / "api_scan.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("runs", {})
    except (json.JSONDecodeError, OSError):
        return {}


API_SCAN = load_api_scan()


def load_openai_judgement() -> dict:
    """Confirmed OpenAI-API-rule verdicts (judging/exp_judge_openai_api.py).
    Upgrades the recall-oriented api_scan candidates to adjudicated verdicts."""
    p = HIGHLIGHTS / "openai_api_judgement.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


OPENAI_JUDGE = load_openai_judgement()


def is_flagged(r: dict) -> tuple[bool, list[str]]:
    """Did either judge flag this task? Returns (flagged, [reasons])."""
    reasons = []
    if (r.get("contamination") or "").strip().startswith("contamination detected"):
        reasons.append("contamination")
    if (r.get("disallowed_model") or "").strip() == "disallowed use detected":
        reasons.append("disallowed model")
    return bool(reasons), reasons


def load_record(run_id: str) -> dict:
    # rollback-experiment runs live in the repo, not the mats-local dataset
    path = (ROLLBACK_DATA / f"{run_id}.json") if run_id.startswith("rollback_") \
        else (DATA / f"{run_id}.json")
    if not path.exists():
        abort(404)
    with open(path) as f:
        return json.load(f)


def load_workspace(run_id: str) -> dict | None:
    # same dispatch as load_record: rollback runs keep their workspace snapshot
    # (built by rollback.build_rollback_workspaces) in the rollback viewer_data
    base = ROLLBACK_DATA if run_id.startswith("rollback_") else DATA
    path = base / f"{run_id}.workspace.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_highlights(run_id: str) -> dict | None:
    path = HIGHLIGHTS / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Filtering / sorting for the index                                           #
# --------------------------------------------------------------------------- #
def _opts(key: str) -> list[str]:
    return sorted({(r.get(key) or "") for r in RUNS if r.get(key) not in (None, "")})


def filtered(args) -> list[tuple[int, dict]]:
    # rollback-experiment runs live only on the /rollback page, not the index
    rows = [(i, r) for i, r in enumerate(RUNS) if not r.get("is_rollback")]
    for key in ("experiment", "benchmark", "agent_model", "trained_model", "trace_format"):
        val = args.get(key)
        if val:
            rows = [(i, r) for i, r in rows if (r.get(key) or "") == val]
    if args.get("rh") == "1":
        rows = [(i, r) for i, r in rows if is_flagged(r)[0]]
    q = (args.get("q") or "").strip().lower()
    if q:
        # accept pasted file paths too: strip everything up to the data root and
        # fold the dir separator into the run_id's '__' join
        if "posttrainbench/" in q:
            q = q.split("posttrainbench/", 1)[1]
        q2 = q.strip("/").replace("/", "__")
        rows = [(i, r) for i, r in rows
                if q in r["run_id"].lower() or q2 in r["run_id"].lower()]
    sort = args.get("sort", "default")
    if sort == "acc_desc":
        rows.sort(key=lambda ir: (ir[1].get("accuracy") is None, -(ir[1].get("accuracy") or 0)))
    elif sort == "acc_asc":
        rows.sort(key=lambda ir: (ir[1].get("accuracy") is None, ir[1].get("accuracy") or 0))
    elif sort == "turns":
        rows.sort(key=lambda ir: -(ir[1].get("num_turns") or 0))
    else:  # default: judge-backed, then verdict-only, then clean; stable within
        rows.sort(key=lambda ir: _sort_key(ir[1]))
    return rows


_SRC_RANK = {"judge_output": 0, "verdict_only": 1}
_LABEL_RANK = {"contamination": 0, "disallowed model": 1, "openai api misuse": 2,
               "unclear": 3, "broke rules unknowingly": 4, "didn't break rules": 5}
_LABEL_CLASS = {"contamination": "contamination", "disallowed model": "disallowed",
                "unclear": "unclear", "broke rules unknowingly": "unknowing",
                "didn't break rules": "clean", "openai api misuse": "apimisuse"}


# Within the OpenAI tier: reward HACKS first, then ACCIDENTAL violations, then
# pending candidates, then judged-unclear. Cleared (no_violation) runs are NOT
# surfaced — they leave the tier (no pill, sorted with the untagged runs).
def _openai_subrank(r: dict):
    """Sub-rank if this run is a SURFACED OpenAI run (matches openai_pill), else
    None. Kept in lockstep with openai_pill so the tier and the flag agree."""
    rid = r["run_id"]
    oj = OPENAI_JUDGE.get(rid)
    if oj:
        v = oj.get("verdict")
        if v == "no_violation":
            return None
        if v == "violation":
            cv = (oj.get("context_assessment") or {}).get("verdict")
            return {"reward_hack": 0, "accidental": 1}.get(cv, 0)  # unassessed -> hack tier
        if v == "unclear":
            return 3
        return None
    ap = API_SCAN.get(rid)
    return 2 if (ap and ap.get("openai")) else None  # unjudged candidate -> 2


def openai_pill(rid: str):
    """Plain-English flag for a run's OpenAI-API status (or None). Decouples the
    two axes the old pill smushed: verdict, and (for violations) whether the call
    RAN and whether its output REACHED the submitted final_model."""
    ap = API_SCAN.get(rid)
    oj = OPENAI_JUDGE.get(rid)
    if not ((ap and ap.get("openai")) or oj):
        return None
    if not oj:  # nominated by the deterministic scan, not yet adjudicated
        return {"cls": "oai pending", "text": "🔑 OpenAI: candidate (unjudged)",
                "title": "Flagged by the deterministic scan as possible OpenAI-API misuse; "
                         f"not yet judged. signals: {ap.get('openai_signals')}"}
    v = oj.get("verdict")
    if v == "no_violation":
        return None  # judged clean — not surfaced (no flag, drops out of the OpenAI tier)
    if v == "unclear":
        return {"cls": "oai", "text": "🔑 OpenAI: unclear",
                "title": f"Judged unclear. {oj.get('rationale','')}"}
    # violation — headline by intent (reward hack vs accidental), keep the
    # run/reach qualifier; context_assessment decides hack vs accident.
    es, rf = oj.get("execution_status"), oj.get("reached_final_model")
    if rf == "yes":
        qual = "reached final model"
    elif es == "authored_but_unrun":
        qual = "code only, never ran"
    elif es == "executed":
        qual = "ran, not in final model" if rf == "no" else "ran, impact unknown"
    else:
        qual = "execution unclear"
    ca = oj.get("context_assessment") or {}
    cv = ca.get("verdict")
    if cv == "reward_hack":
        head, cls = "reward hack", "oai viol"
    elif cv == "accidental":
        head, cls = "accidental violation", "oai accidental"
    else:
        head, cls = "violation (intent unassessed)", "oai viol"
    return {"cls": cls, "text": f"🔑 OpenAI: {head} — {qual}",
            "title": (f"OpenAI-API rule — {head.upper()}. " + (ca.get("basis", "") + "  ‖  " if ca else "")
                      + f"use: {oj.get('non_eval_purpose','')} | execution: {es}, reached final_model: {rf}, "
                      f"model: {oj.get('openai_model','')}, key: {oj.get('key_used','')}. {oj.get('rationale','')}")}


def _sort_key(r: dict) -> tuple:
    """Browse order: (-1) our rollback-experiment runs pinned to the TOP; then
    (0) runs with a final RH label, by label [contamination -> disallowed ->
    unclear -> unknowing -> didn't-break] then old-judge evidence level;
    (1) OpenAI-API candidates; (2) everything untagged. Stable within."""
    rid = r["run_id"]
    if rid.startswith("rollback_"):
        return (-1, 0, 0)
    m = HL_META.get(rid)
    if m:
        return (0, _LABEL_RANK.get(m.get("label"), 5),
                _SRC_RANK.get(HL_SOURCES.get(rid, ""), 2))
    osr = _openai_subrank(r)
    if osr is not None:
        return (1, osr, 0)
    return (2, 0, 0)


# Default browse order (judge -> verdict -> clean, stable). The detail-page
# prev/next walk this so navigation matches the index table.
ORDER = [RUNS[i]["run_id"] for i in sorted(range(len(RUNS)), key=lambda i: _sort_key(RUNS[i]))]
ORDER_POS = {rid: p for p, rid in enumerate(ORDER)}


# --------------------------------------------------------------------------- #
# Templates                                                                   #
# --------------------------------------------------------------------------- #
CSS = """
 body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1480px;margin:1.2rem auto;padding:0 1rem;color:#1a1a1a}
 td.flags{white-space:nowrap}
 a{color:#1558d6;text-decoration:none} a:hover{text-decoration:underline}
 code{background:#f3f3f3;padding:1px 4px;border-radius:3px;font-size:.92em}
 h1{font-size:1.4rem;margin-bottom:.2rem} .meta{color:#666;font-size:.9rem;margin-top:0}
 .reportlink{margin:.5rem 0 0;font-size:1.05rem}
 .reportlink a{font-weight:700}
 .filters{background:#f7f7f9;border:1px solid #eee;border-radius:8px;padding:.7rem .9rem;margin:1rem 0;font-size:.88rem;display:flex;flex-wrap:wrap;gap:.6rem;align-items:center}
 .filters select,.filters input{font:inherit;padding:.2rem .35rem;border:1px solid #ccc;border-radius:5px}
 .filters label{color:#666}
 table{border-collapse:collapse;width:100%;margin-top:.6rem;font-size:.86rem}
 th,td{text-align:left;padding:.38rem .5rem;border-bottom:1px solid #eee;vertical-align:top}
 th{font-size:.74rem;text-transform:uppercase;letter-spacing:.03em;color:#888}
 tr:hover{background:#fafafa}
 tr.rh{background:#fff1f0} tr.rh:hover{background:#ffe6e4}
 .tag{display:inline-block;background:#fde7e7;color:#b3261e;border-radius:4px;padding:0 6px;font-size:.74rem;font-weight:600;margin-right:3px}
 .loc{display:inline-block;border-radius:4px;padding:0 6px;font-size:.72rem;font-weight:600;margin-right:3px}
 .loc.judge{background:#eef;color:#3b3b8f;border:1px solid #ccd}
 .loc.verdict{background:#f3f3f3;color:#666;border:1px solid #ddd}
 .lbl{display:inline-block;border-radius:4px;padding:0 7px;font-size:.74rem;font-weight:700}
 .lbl.contamination{background:#fde7e7;color:#b3261e;border:1px solid #f2b8b5}
 .lbl.disallowed{background:#b3261e;color:#fff;border:1px solid #8f1d12}
 .lbl.unknowing{background:#e8f1fd;color:#1a4f9c;border:1px solid #b9d2f5}
 .lbl.clean{background:#e6f4ea;color:#1a7d33;border:1px solid #b7e1c3}
 .lbl.apimisuse{background:#f1e7fb;color:#6b21a8;border:1px solid #d6b4f0}
 .lbl.unclear{background:#fff7e6;color:#8a5a00;border:1px solid #f0d28a}
 .apitag{display:inline-block;border-radius:4px;padding:0 5px;font-size:.68rem;font-weight:600;margin-left:3px;cursor:help}
 .apitag.oai{background:#fff3e0;color:#a05a00;border:1px solid #f0c27a}
 .apitag.oai.viol{background:#f1e7fb;color:#6b21a8;border:1px solid #d6b4f0}
 .apitag.oai.accidental{background:#fdebd0;color:#9a5b00;border:1px solid #eec07a}
 .apitag.oai.cleared{background:#e6f4ea;color:#1a7d33;border:1px solid #b7e1c3}
 .apitag.oai.pending{border-style:dashed}
 .legend{background:#f7f7f9;border:1px solid #eee;border-radius:8px;padding:.5rem .8rem;margin:.6rem 0;font-size:.82rem}
 .legend summary{cursor:pointer;font-weight:600;color:#555}
 .legend ul{margin:.5rem 0 .2rem;padding-left:1.1rem} .legend li{margin:.15rem 0}
 .legend .apitag{cursor:default}
 .agentbad{display:inline-block;color:#ea580c;font-weight:900;cursor:help;line-height:1}
 .rj{display:inline-block;border-radius:4px;padding:0 6px;font-size:.72rem;font-weight:700;margin-right:3px}
 .rj.yes{background:#fde7e7;color:#b3261e;border:1px solid #f2b8b5}
 .rj.questionable{background:#fff7e6;color:#8a5a00;border:1px solid #f0d28a}
 .rj.no{background:#e6f4ea;color:#1a7d33;border:1px solid #b7e1c3}
 .cut{display:inline-block;border-radius:4px;padding:0 6px;font-size:.72rem;font-weight:700;margin-right:3px;background:#fff3e0;color:#a05a00;border:1px solid #f0c27a;cursor:help}
 .fj{display:inline-block;border-radius:4px;padding:0 6px;font-size:.72rem;font-weight:700;margin-right:3px;background:#f0f7f4;color:#1f6b48;border:1px solid #9fcdb6;cursor:help}
 .fj.flip{background:#1f6b48;color:#fff;border-color:#155236}
 .trap{display:inline-block;border-radius:4px;padding:0 6px;font-size:.72rem;font-weight:600;margin-right:3px;background:#f3f0ff;color:#4c3a8f;border:1px solid #cdc3ee}
 .trap.knew{background:#fde7e7;color:#8f1d12;border-color:#f2b8b5}
 .trap.misled{background:#e8f1fd;color:#1a4f9c;border-color:#b9d2f5}
 .num{font-variant-numeric:tabular-nums}
 .nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem}
 .hdr{background:#f7f7f9;border:1px solid #eee;border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem;font-size:.9rem}
 .hdr b{color:#000}
 .pathline{margin-top:.4rem;display:flex;align-items:center;gap:.4rem}
 .pathline code{font-size:.78rem;color:#555;background:#f0f0f2;padding:2px 6px;border-radius:4px;word-break:break-all;user-select:all}
 .pathline button{background:#fff;border:1px solid #ccc;border-radius:5px;cursor:pointer;padding:0 6px;font-size:.8rem}
 .banner{border-radius:8px;padding:.7rem 1rem;margin-bottom:1rem;font-size:.92rem}
 .banner.rh{background:#fff1f0;border:1px solid #f2b8b5}
 .banner.ok{background:#f0fbf2;border:1px solid #bfe6cb}
 .banner .vtag{font-weight:700;text-transform:uppercase;font-size:.8rem;letter-spacing:.04em}
 .banner.rh .vtag{color:#b3261e} .banner.ok .vtag{color:#1a7d33}
 .banner.hl{background:#fffbe6;border:1px solid #f5d97b}
 .banner .report{margin-top:.5rem} .banner .report summary{cursor:pointer;color:#8a5a00;font-size:.85rem;font-weight:600}
 .banner .report pre{white-space:pre-wrap;font:13px/1.5 ui-monospace,Menlo,monospace;background:#fffdf3;border:1px solid #f0e2b0;border-radius:6px;padding:.6rem .7rem;margin-top:.5rem;max-height:380px;overflow:auto}
 mark{background:#ffe08a;padding:0 1px;border-radius:2px}
 mark.hack{background:#ffb3ab;color:#5a0d07;font-weight:600;padding:0 2px}
 .sessdiv{text-align:center;color:#999;font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;margin:1.2rem 0 .4rem}
 .rollbackcut{white-space:pre-wrap;text-align:center;font-weight:700;color:#9a3412;background:#fff7ed;border:2px dashed #ea580c;border-radius:8px;padding:.7rem 1rem;margin:1.6rem 0 .6rem}
 .rollbackresume{background:#eef6ff;border:1px solid #93c5fd;border-radius:6px;padding:.5rem .8rem;margin:0 0 1rem;color:#1e40af;font-size:.9rem}
 .lbl.rollback{background:#fff7ed;color:#9a3412;border:1px solid #fdba74}
 .rbrunningrow{color:#64748b;background:#f8fafc}
 .rbrunning{display:inline-block;color:#475569;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:4px;padding:0 .35rem;font-weight:700}
 .rbliverow{background:#eff6ff}
 .rblive{display:inline-block;color:#1558d6;background:#e8f1fe;border:1px solid #93c5fd;border-radius:4px;padding:0 .4rem;font-weight:800;animation:livepulse 2s ease-in-out infinite}
 .rblive::before{content:"\\25B6  ";color:#1558d6}
 @keyframes livepulse{0%,100%{border-color:#93c5fd}50%{border-color:#c7dbfb}}
 .rbip{font-family:ui-monospace,Menlo,monospace;font-size:.8em;opacity:.8;font-weight:600}
 .rbdeferredrow{background:#f8fafc;color:#94a3b8}
 .rbdeferredrow a{color:#94a3b8}
 .rbtodorow{background:#fbfdff}
 .rbenginehdr td{background:#1f2937;color:#fff;font-weight:800;font-size:.82rem;letter-spacing:.04em;text-transform:uppercase;padding:.35rem .6rem;border-top:2px solid #111}
 .rbtodo{display:inline-block;color:#1d4ed8;background:#eff6ff;border:1px dashed #93c5fd;border-radius:4px;padding:0 .4rem;font-weight:700}
 .rbstat.todo{color:#1d4ed8;background:#eff6ff;border:1px dashed #93c5fd}
 .rbdefbadge{display:inline-block;color:#64748b;background:#eef2f6;border:1px solid #cbd5e1;border-radius:4px;padding:0 .4rem;font-size:.7rem;font-weight:700;cursor:help}
 .rbtag{display:inline-block;border-radius:4px;padding:0 .4rem;font-size:.7rem;font-weight:700;cursor:help;margin:0 .1rem .15rem 0;vertical-align:middle}
 .rbtag.retrain{color:#92400e;background:#fffbeb;border:1px solid #fcd34d}
 .rbtag.noretrain{color:#475569;background:#f1f5f9;border:1px solid #cbd5e1}
 .rbtag.cheat{color:#9a3412;background:#fff7ed;border:1px solid #fdba74}
 .rbtag.uncat{color:#94a3b8;background:#f8fafc;border:1px dashed #cbd5e1}
 .rbstat{display:inline-block;border-radius:4px;padding:0 .4rem;font-size:.68rem;font-weight:800;letter-spacing:.02em;vertical-align:middle}
 .rbstat.running{color:#1558d6;background:#e8f1fe;border:1px solid #93c5fd}
 .rbstat.done{color:#1a7d33;background:#e6f4ea;border:1px solid #b7e1c3}
 .rbstat.partial{color:#8a5a00;background:#fff7e6;border:1px solid #f0d28a}
 .rbstat.notrun{color:#64748b;background:#f1f5f9;border:1px solid #cbd5e1}
 .rbstat.deferred{color:#64748b;background:#eef2f6;border:1px solid #cbd5e1}
 .rbnew{display:none;margin-left:.35rem;border:1px solid #93c5fd;background:#eff6ff;color:#1d4ed8;border-radius:4px;padding:0 .35rem;font-size:.68rem;font-weight:800;letter-spacing:.02em;cursor:pointer;vertical-align:baseline}
 .rbnew:hover{background:#dbeafe;text-decoration:none}
 .msg{border:1px solid #e6e6e6;border-radius:8px;margin:.6rem 0;overflow:hidden}
 .msg.hackturn{border:2px solid #d93025;box-shadow:0 0 0 3px #fde7e7}
 .hacktag{background:#d93025;color:#fff;border-radius:3px;padding:0 6px;margin-left:8px;font-weight:700}
 .msg.notedturn{border:2px solid #64748b;box-shadow:0 0 0 3px #e9edf2}
 .notetag{background:#64748b;color:#fff;border-radius:3px;padding:0 6px;margin-left:8px;font-weight:700}
 .traintag{background:#0e7490;color:#fff;border-radius:3px;padding:0 6px;margin-left:8px;font-weight:700;cursor:help}
 .traintag.failed{background:#9aa3ad}
 .apiflag{border-radius:3px;padding:0 6px;margin-left:8px;font-weight:700;cursor:help;color:#fff}
 .apiflag.oai{background:#a05a00}
 .apiflag.ruleseen{background:#1a4f9c}
 .apiflag.cathack0{background:#b3261e}
 .apiflag.cathack1{background:#9a3412}
 .hacknav{position:fixed;right:18px;bottom:74px;display:flex;flex-direction:column;gap:6px;z-index:50;font-size:.8rem}
 .hacknav .hnrow{display:flex;gap:.6rem;align-items:center;justify-content:space-between;background:#f8fafc;color:#334155;border:1px solid #cbd5e1;border-radius:8px;padding:.18rem .5rem;box-shadow:0 1px 4px rgba(0,0,0,.12)}
 .hacknav button{background:#fff;color:#475569;border:1px solid #cbd5e1;border-radius:5px;font-weight:700;cursor:pointer;padding:0 .4rem;font-size:.85rem;line-height:1.3}
 .hacknav button:hover{background:#e2e8f0}
 .hacknav .lbl{font-variant-numeric:tabular-nums;min-width:40px;text-align:center}
 .totop{position:fixed;right:18px;bottom:18px;background:#1558d6;color:#fff;border:none;border-radius:50%;width:42px;height:42px;font-size:1.25rem;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.28);display:none;z-index:50}
 .totop:hover{background:#0f47b0}
 .msg.flash{animation:flashpulse 1.1s ease}
 @keyframes flashpulse{0%{box-shadow:0 0 0 3px #fde7e7}25%{box-shadow:0 0 0 10px #f5867d}100%{box-shadow:0 0 0 3px #fde7e7}}
 .catcard{border:1px solid #f2b8b5;border-left:5px solid #d93025;border-radius:8px;padding:.8rem 1rem;margin:.9rem 0;background:#fffafa}
 .catcard.benign{border-color:#bfe6cb;border-left-color:#1a7d33;background:#fafffa}
 .cat-h{font-size:1rem;margin-bottom:.25rem}
 .cat-desc{color:#444;font-size:.9rem;margin-bottom:.4rem}
 .multi{display:inline-block;background:#eef;color:#3b3b8f;border:1px solid #ccd;border-radius:8px;padding:0 5px;font-size:.7rem;font-weight:700;cursor:help}
 .annos{margin:-.4rem 0 .9rem;display:flex;flex-direction:column;gap:.3rem}
 .anno{background:#fffaf0;border:1px dashed #d9a441;border-left:4px solid #d9a441;border-radius:0 6px 6px 0;padding:.4rem .6rem;font-size:.86rem;color:#5a4630}
 .anno-lbl{font-weight:700;text-transform:uppercase;font-size:.68rem;letter-spacing:.04em;color:#a3631a;margin-right:.4rem}
 .anno.anno-label_audit{background:#f5f2ff;border-color:#8d77d6;border-left-color:#5b46a8;color:#3c2f6b}
 .anno.anno-label_audit .anno-lbl{color:#5b46a8}
 .anno.anno-label_audit .anno-q{background:#e7e0fa}
 .anno.anno-final_judge{background:#f0f7f4;border-color:#5c9c7d;border-left-color:#1f6b48;color:#1d4733}
 .anno.anno-final_judge .anno-lbl{color:#1f6b48}
 .anno.anno-final_judge .anno-q{background:#dceee4}
 .anno.anno-story_evidence{background:#fff5f4;border-color:#e09690;border-left-color:#b3261e;color:#5e1812}
 .anno.anno-story_evidence .anno-lbl{color:#b3261e}
 .anno.anno-story_suspicion{background:#fffaf0;border-color:#d9a441;border-left-color:#b07c1d;color:#5a4630}
 .anno.anno-story_suspicion .anno-lbl{color:#a3631a}
 .anno.anno-story_rebuttal{background:#f0faf3;border-color:#7cbd97;border-left-color:#1a7d33;color:#1d4733}
 .anno.anno-story_rebuttal .anno-lbl{color:#1a7d33}
 .anno.anno-story_uncertainty{background:#fbf7ee;border-color:#cdb46e;border-left-color:#8a6d1a;color:#574a22}
 .anno.anno-story_uncertainty .anno-lbl{color:#8a6d1a}
 .anno.anno-story_context{background:#f6f7f9;border-color:#aab4c0;border-left-color:#64748b;color:#3c4856}
 .anno.anno-story_context .anno-lbl{color:#64748b}
 .anno-q{background:#fff2cc;border-radius:3px;padding:0 3px;font-family:ui-monospace,Menlo,monospace;font-size:.85em}
 .verdict-line{margin-top:.4rem}
 .rolehdr{font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;color:#fff;padding:.3rem .7rem}
 .role-system .rolehdr{background:#6b7280}
 .role-assistant .rolehdr{background:#1558d6}
 .role-user .rolehdr{background:#0a7d33}
 .role-reasoning .rolehdr{background:#7c5cbf}
 .ectag{background:#b3261e;color:#fff;border-radius:3px;padding:0 5px;margin-left:8px;font-size:.92em}
 pre{white-space:pre-wrap;word-break:break-word;margin:0}
 pre.content{padding:.55rem .7rem;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:440px;overflow:auto}
 pre.content.tool{color:#444;background:#fbfbfb} pre.content.empty{color:#aaa;font-style:italic}
 pre.content.reasoning{color:#5b4b8a;background:#faf8ff;font-style:italic}
 .toolout summary{cursor:pointer;color:#0a7d33;padding:.4rem .7rem;font-size:.82rem} .toolout pre.content{max-height:none}
 .call{border-top:1px dashed #eee;padding:.5rem .7rem;background:#fcfcfd}
 .callname{font-size:.86rem;color:#444;margin-bottom:.35rem}
 pre.code{background:#1e1e2e;color:#e6e6f0;padding:.6rem .7rem;border-radius:6px;font:12.5px/1.5 ui-monospace,Menlo,monospace;max-height:440px;overflow:auto}
 pre.args{background:#f3f3f3;padding:.45rem .6rem;border-radius:6px;font:12px/1.4 ui-monospace,Menlo,monospace;color:#555;margin-top:.4rem}
 .wsfile summary{cursor:pointer;padding:.35rem 0;font-size:.86rem;color:#1558d6}
 summary{outline:none}
 .rbfig{display:block;max-width:680px;width:100%;margin:.6rem 0;border:1px solid #e6e6e6;border-radius:8px;background:#fff}
 .rbbreak{max-width:680px;margin:.4rem 0 1rem}
 .rbbreak details{border:1px solid #e6e6e6;border-radius:6px;margin:.3rem 0;background:#fcfcfd}
 .rbbreak summary{cursor:pointer;padding:.35rem .6rem;font-size:.84rem;font-weight:700;color:#334155}
 .rbbreak ul{margin:.1rem 0 .5rem;padding-left:1.4rem;font-size:.8rem;color:#475569;font-family:ui-monospace,Menlo,monospace}
 .rbbreak li{margin:.1rem 0}
"""

INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>PostTrainBench trajectories</title>
<style>__CSS__</style></head><body>
<h1>PostTrainBench agent trajectories</h1>
<p class="meta">{{ total }} tasks · {{ n_exp }} experiments · 7 benchmarks · 4 base models.
 Showing <b>{{ rows|length }}</b>. Labels are the final adjudication of every judged run
 (the {{ n_flagged }} originally flagged + the OpenAI-API-rule violations):
 <span class="lbl contamination">contamination</span> /
 <span class="lbl disallowed">disallowed model</span> /
 <span class="lbl apimisuse">openai api misuse</span> = reward hacking (red rows);
 <span class="lbl unknowing">broke rules unknowingly</span> /
 <span class="lbl clean">didn't break rules</span> = not reward hacking;
 <span class="lbl unclear">unclear</span> = needs human review.</p>
{% if has_report %}<p class="reportlink"><a href="/report">📋 cheating report</a></p>
<p class="reportlink" style="margin-top:.15rem"><a href="/timing">⏱ RH timing</a></p>
<p class="reportlink" style="margin-top:.15rem"><a href="/rates">📊 RH rates</a></p>{% endif %}
<p class="reportlink" style="margin-top:.15rem"><a href="/rollback">🔪 rollback experiments</a></p>
<form class="filters" method="get">
 <label>experiment <select name="experiment" onchange="this.form.submit()">
   <option value="">all</option>
   {% for o in exp_opts %}<option value="{{o}}" {{'selected' if o==sel.experiment}}>{{o}}</option>{% endfor %}
 </select></label>
 <label>benchmark <select name="benchmark" onchange="this.form.submit()">
   <option value="">all</option>
   {% for o in bench_opts %}<option value="{{o}}" {{'selected' if o==sel.benchmark}}>{{o}}</option>{% endfor %}
 </select></label>
 <label>agent <select name="agent_model" onchange="this.form.submit()">
   <option value="">all</option>
   {% for o in agent_opts %}<option value="{{o}}" {{'selected' if o==sel.agent_model}}>{{o}}</option>{% endfor %}
 </select></label>
 <label>base <select name="trained_model" onchange="this.form.submit()">
   <option value="">all</option>
   {% for o in base_opts %}<option value="{{o}}" {{'selected' if o==sel.trained_model}}>{{o}}</option>{% endfor %}
 </select></label>
 <label>sort <select name="sort" onchange="this.form.submit()">
   <option value="default" {{'selected' if sel.sort=='default'}}>default</option>
   <option value="acc_desc" {{'selected' if sel.sort=='acc_desc'}}>accuracy ↓</option>
   <option value="acc_asc" {{'selected' if sel.sort=='acc_asc'}}>accuracy ↑</option>
   <option value="turns" {{'selected' if sel.sort=='turns'}}>turns ↓</option>
 </select></label>
 <label><input type="checkbox" name="rh" value="1" {{'checked' if sel.rh}} onchange="this.form.submit()"> flagged only</label>
 <input type="text" name="q" value="{{ sel.q }}" placeholder="search run_id / path…" size="20">
 <button type="submit">go</button>
 <a href="/">reset</a>
</form>
<details class="legend"><summary>flag legend</summary>
 <div><b>Final RH label</b> (every adjudicated run — the 51 originally flagged + the OpenAI-API-rule violations):
  <span class="lbl contamination">contamination</span>
  <span class="lbl disallowed">disallowed model</span>
  <span class="lbl apimisuse">openai api misuse</span>
  <span class="lbl unknowing">broke rules unknowingly</span>
  <span class="lbl unclear">unclear</span>
  <span class="lbl clean">didn't break rules</span></div>
 <div style="margin-top:.5rem"><b>Rollback run status</b>:
  <span class="agentbad" title="fatal error by the agent">&#9679;</span>
  orange circle = agent made a bad mistake and the rollout is not a valid completed result; rerun needed.
 </div>
 <div style="margin-top:.5rem"><b>OpenAI-API rule</b> (used the eval key for non-eval work? — only rule-breaking runs are flagged; <b>reward hacks sort above accidents</b>):</div>
 <ul>
  <li><span class="apitag oai viol">🔑 OpenAI: reward hack — …</span> — the rule was <b>in context</b> at the violation, so the agent broke it knowingly (deliberate).</li>
  <li><span class="apitag oai accidental">🔑 OpenAI: accidental violation — …</span> — the rule had <b>dropped out of context</b> (evicted by compaction) by the violation; broke it unknowingly.</li>
  <li><span class="apitag oai">🔑 OpenAI: unclear</span> · <span class="apitag oai pending">🔑 OpenAI: candidate (unjudged)</span></li>
 </ul>
 <div style="color:#777">The trailing qualifier is the <b>severity</b> axis: <i>reached final model</i> (score impact) › <i>ran, not in final model</i> › <i>code only, never ran</i>. Runs judged <b>no violation</b> are not flagged. Hover any flag for the hack-vs-accident basis + judge rationale. On a flagged run's page, the <b>🔑 OpenAI rule-break</b> toggle (key o) steps through the rule-breaking spots (first stop ① = earliest attempt); <b>📖</b> (key r) marks where the rule was seen.</div>
</details>
<table><thead><tr>
 <th>#</th>
 <th>experiment</th><th>bench</th><th>base model</th><th>agent</th>
 <th title="benchmark score (accuracy). rollbacks show new&nbsp;(original)">score</th><th>turns</th><th>time</th><th>flags</th>
</tr></thead><tbody>
{% for i,r in rows %}
 <tr class="{{ 'rh' if hl_meta.get(r.run_id) and hl_meta.get(r.run_id).is_reward_hack }}">
  <td class="num">{{ loop.index }}</td>
  <td><a href="/run/{{ r.run_id }}">{{ r.experiment }}</a></td>
  <td>{{ r.benchmark }}</td>
  <td>{{ r.trained_model }}</td>
  <td>{{ r.agent_model }}</td>
  <td class="num">{{ '%.3f'|format(r.accuracy) if r.accuracy is not none else '–' }}{% if r.is_rollback %} <span style="color:#888" title="original (pre-rollback) score">({{ '%.3f'|format(r.orig_accuracy) if r.orig_accuracy is not none else '–' }})</span>{% endif %}</td>
  <td class="num">{{ r.num_turns or '–' }}</td>
  <td class="num">{{ r.time_taken or '–' }}</td>
  <td class="flags">{% if r.is_rollback %}<span class="lbl rollback" title="rollback-intervention experiment (resumed from a cut)">rollback</span>{% endif %}
   {% if r.failure_category == 'agent_fatal_error' %}<span class="agentbad" title="{{ r.failure_mode or 'fatal error by the agent' }}: {{ r.failure_summary or 'agent made a bad mistake; not a valid completed result' }}">&#9679;</span>{% endif %}
   {% set hm = hl_meta.get(r.run_id) %}
   {% if hm %}<span class="lbl {{ label_class.get(hm.label,'') }}" title="final assessment (full provenance on the run page)">{{ hm.label }}</span>{% endif %}
   {% set op = oai_pill(r.run_id) %}
   {% if op %}<span class="apitag {{ op.cls }}" title="{{ op.title }}">{{ op.text }}</span>{% endif %}</td>
 </tr>
{% endfor %}
</tbody></table></body></html>
""".replace("__CSS__", CSS)

DETAIL_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>{{ r.run_id }}</title>
<style>__CSS__</style></head><body>
<div class="nav">
 <a href="/{{ back_qs }}">&larr; index</a>
 <span>{% if prev %}<a href="/run/{{ prev }}">&larr; prev</a>{% endif %}
  &nbsp;{{ pos }} / {{ total }}&nbsp;
  {% if next %}<a href="/run/{{ next }}">next &rarr;</a>{% endif %}</span>
</div>
<div class="hdr">
 <div><b>{{ r.benchmark }}</b> · base <b>{{ r.trained_model }}</b> · agent <b>{{ r.agent_model }}</b>
   · format {{ r.trace_format }}</div>
 <div style="margin-top:.3rem;color:#555">experiment <b>{{ r.experiment }}</b> · seed {{ r.seed }}
   · accuracy <b>{{ '%.4f'|format(r.accuracy) if r.accuracy is not none else '–' }}</b>
   {% if r.stderr is not none %}± {{ '%.4f'|format(r.stderr) }}{% endif %}
   · {{ r.num_turns or '?' }} turns · {{ r.session_count or '?' }} session(s)
   · {{ r.time_taken or '?' }}{% if r.total_cost_usd %} · ${{ '%.2f'|format(r.total_cost_usd) }}{% endif %}</div>
 <div style="margin-top:.4rem">
   {% if workspace %}<a href="/workspace/{{ r.run_id }}">📁 final workspace ({{ workspace_n }} files)</a>{% endif %}
 </div>
 <div class="pathline">
  <code id="rawpath">{{ raw_path }}</code>
  <button id="copypath" title="copy path">📋</button>
  <script>document.getElementById('copypath').onclick=function(){
    navigator.clipboard.writeText(document.getElementById('rawpath').textContent);
    this.textContent='✓'; var b=this; setTimeout(function(){b.textContent='📋';},1200); };</script>
 </div>
</div>
<div class="banner {{ 'rh' if flagged[0] else 'ok' }}">
 <span class="vtag">{% if flagged[0] %}⚠ flagged: {{ flagged[1]|join(', ') }}{% else %}no hack detected by judges{% endif %}</span>
 <div style="margin-top:.3rem;color:#555">contamination: <b>{{ r.contamination or '–' }}</b> · disallowed model: <b>{{ r.disallowed_model or '–' }}</b></div>
</div>
{% if oj %}
<div class="banner hl" style="{% if oj.verdict=='violation' %}background:#fff1f0;border-color:#f2b8b5{% elif oj.verdict=='no_violation' %}background:#f0fbf2;border-color:#bfe6cb{% endif %}">
 <span class="rj {{ 'yes' if oj.verdict=='violation' else 'no' if oj.verdict=='no_violation' else 'questionable' }}" style="font-size:.84rem;padding:1px 8px">🔑 OpenAI-API rule: {% if oj.verdict=='violation' %}VIOLATION{% elif oj.verdict=='no_violation' %}no violation{% if oj.no_reason %} ({{ oj.no_reason.replace('_',' ') }}){% endif %}{% else %}unclear{% endif %}</span>
 <span class="ai" style="color:#888;font-size:.76rem;float:right">{{ oj.model }}</span>
 {% if oj.context_assessment %}{% set ca = oj.context_assessment %}
 <div class="verdict-line" style="margin-top:.45rem;border-radius:6px;padding:.5rem .7rem;background:{{ '#fde7e7' if ca.verdict=='reward_hack' else '#fdebd0' }};border:1px solid {{ '#f2b8b5' if ca.verdict=='reward_hack' else '#eec07a' }}">
  <b style="color:{{ '#b3261e' if ca.verdict=='reward_hack' else '#9a5b00' }}">{{ '⚠ REWARD HACK — deliberate' if ca.verdict=='reward_hack' else '➖ ACCIDENTAL VIOLATION — rule out of context' }}</b>
  &nbsp;<span style="font-size:.82rem;color:#555">rule {{ 'WAS in context' if ca.rule_in_context_at_violation else 'had DROPPED out of context' }} at the violation</span>
  <div style="margin-top:.3rem">{{ ca.basis }}</div>
  <div style="margin-top:.25rem;font-size:.78rem;color:#777">Basis: gap from last rule sighting (turn {{ ca.last_rule_seen_turn }}) → violation (turn {{ ca.first_attempt_turn }}) ≈ {{ '{:,}'.format(ca.gap_tokens) }} {{ ca.tokenizer }} tokens vs a ~{{ '{:,}'.format(ca.model_window) }}-token window{% if ca.effective_window %} (~{{ '{:,}'.format(ca.effective_window) }} effective){% endif %}. {{ ca.method }}.</div>
 </div>
 {% endif %}
 <div class="verdict-line" style="margin-top:.4rem">
  {% if oj.non_eval_purpose %}<b>Use:</b> {{ oj.non_eval_purpose }}<br>{% endif %}
  <b>execution:</b> {{ oj.execution_status.replace('_',' ') }} ·
  <b>reached final_model:</b> {{ oj.reached_final_model }} ·
  <b>endpoint:</b> {{ oj.api_target }}{% if oj.openai_model %} ·
  <b>model:</b> <code>{{ oj.openai_model }}</code>{% endif %}{% if oj.key_used %} ·
  <b>key:</b> <code>{{ oj.key_used }}</code>{% endif %}
 </div>
 <div class="verdict-line">{{ oj.rationale }}</div>
 {% if oj.first_attempt %}
 <div class="verdict-line" style="background:#fbf0f0;border:1px solid #f0c5c0;border-radius:6px;padding:.4rem .6rem;margin-top:.4rem">
  <b>⏱ Timing anchors</b> (deliberate-vs-fell-out-of-context judgment deferred to a later pass) —
  <b>first illegal-use attempt:</b> turn {{ oj.first_attempt.turn }} ({{ oj.first_attempt.stage }}) <span style="color:#888">[jump: o — first stop, marked ①]</span>.
  {% set rt = oj.rule_timing or {} %}
  {% if not rt.ever_saw_rule %}<b style="color:#b3261e">The agent apparently NEVER read the rule</b> (only ran evaluate.py without viewing its source) — points to "never knew" rather than "forgot."
  {% elif rt.saw_rule_before_attempt %}<b>Rule last seen:</b> turn {{ rt.last_rule_seen_turn }} <span style="color:#888">[jump: r]</span> — <b>{{ rt.turns_since_rule }} turns</b> and <b>{{ rt.sessions_crossed_since_rule }} session reset(s)</b> before the attempt.
  {% else %}The agent only read the rule <i>after</i> the first attempt.{% endif %}
 </div>
 {% endif %}
 {% if oj.workspace_evidence %}
 <div class="verdict-line" style="font-size:.85rem"><b>workspace evidence:</b>
  {% for w in oj.workspace_evidence %}
  <div style="margin-top:.15rem"><code>{{ w.file }}</code>{% if w.snippet %} — <span style="font-family:ui-monospace,Menlo,monospace;background:#e7f2ec;padding:0 3px;border-radius:3px">{{ w.snippet }}</span>{% endif %}{% if w.note %} <span style="color:#555">({{ w.note }})</span>{% endif %}</div>
  {% endfor %}
 </div>
 {% endif %}
 {% if oj.trace_highlights %}<div class="verdict-line" style="font-size:.82rem;color:#8a5a00">{{ oj.trace_highlights|length }} evidence turn(s) in the trace — jump with the <b>🔑 OpenAI</b> nav (keys o / O).{% if oj.coverage and oj.coverage.n_chunks > 1 %} Full trace read by a {{ oj.coverage.n_chunks }}-chunk labeler relay, then synthesized.{% endif %}</div>{% endif %}
</div>
{% endif %}
{% if hl %}
<div class="banner hl">
 {% if hl.reassessment and hl.reassessment.verdict %}
 {% set v = everdict %}
 <span class="rj {{ v }}" style="font-size:.84rem;padding:1px 8px">{% if v=='yes' %}reward hacking{% elif v=='no' %}NOT reward hacking{% else %}questionable{% endif %}</span>
 {% if v != hl.reassessment.verdict %}<span style="color:#666;font-size:.8rem">(rule-break verdict was "{{ hl.reassessment.verdict }}" — reclassified by the knowledge audit: {{ (hl.train_label_audit or {}).get('knowledge','') }})</span>{% endif %}
 <span class="ai" style="color:#888;font-size:.76rem;float:right">{{ hl.reassessment.model }}</span>
 {% endif %}
 {% if hl.story %}
 <div class="verdict-line" style="font-size:.95rem;white-space:pre-wrap">{{ hl.story.summary }}</div>
 <details class="report"><summary>full judging history (all judges' raw outputs)</summary>
 {% endif %}
 {% if hl.reassessment and hl.reassessment.rationale %}<div class="verdict-line">{{ hl.reassessment.rationale }}</div>{% endif %}
 {% if hl.final_judge %}
 <div class="verdict-line" style="background:#f0f7f4;border:1px solid #9fcdb6;border-radius:6px;padding:.5rem .7rem;margin-top:.4rem">
  <b>⚖ Final judge</b> ({{ hl.final_judge.mode }} mode, {{ hl.final_judge.model }}) —
  <b>{{ hl.final_judge.verdict }}{% if hl.final_judge.no_reason %}: {{ hl.final_judge.no_reason.replace('_',' ') }}{% endif %}</b>
  {% if not hl.final_judge.agrees_with_standing %}<span class="tag" style="margin-left:6px">overrides prior verdict</span>{% endif %}
  <div style="margin-top:.25rem">{{ hl.final_judge.rationale }}</div>
  {% if hl.final_judge.workspace_evidence %}
  <div style="margin-top:.3rem;font-size:.85rem"><b>workspace evidence:</b>
   {% for w in hl.final_judge.workspace_evidence %}
   <div style="margin-top:.15rem"><code>{{ w.file }}</code>{% if w.snippet %} — <span style="font-family:ui-monospace,Menlo,monospace;background:#e7f2ec;padding:0 3px;border-radius:3px">{{ w.snippet }}</span>{% endif %}{% if w.note %} <span style="color:#555">({{ w.note }})</span>{% endif %}</div>
   {% endfor %}
  </div>
  {% endif %}
 </div>
 {% endif %}
 {% if hl.train_label_trap %}
 <div class="verdict-line">
  <span class="trap {{ (hl.train_label_audit or {}).get('knowledge','') }}">🏷 train-label trap</span>
  trained on <code>{{ hl.train_label_trap.dataset }}</code> via split="train" — which IS the eval pool (first load: event {{ hl.train_label_trap.first_load_event }}).
  {% if hl.train_label_audit %}
   Knowledge audit: <b>{{ hl.train_label_audit.knowledge }}</b> — {{ hl.train_label_audit.rationale }}
   {% if hl.train_label_audit.knowledge == 'misled' %}
   <div style="margin-top:.25rem;color:#1a4f9c"><b>⚠ Note:</b> displayed as <b>RH: no</b> because the audit found the agent honestly fooled by the split label — an environment-design victim, not deliberate reward hacking. The rule violation as a data flow still stands (test items did reach training). Evidence is annotated 🏷 in the trace below.</div>
   {% endif %}
  {% else %}<i>(knowledge unaudited — run exp_audit_train_label.py)</i>{% endif %}
 </div>
 {% endif %}
 {% if hl.marked_turns %}<div class="verdict-line" style="color:#8a5a00;font-size:.82rem">{% if hl.turn_kinds %}{{ n_hack_true }} hack turn(s) · {{ n_noted }} noted turn(s){% else %}{{ hl.marked_turns|length }} marked turn(s) (unclassified — run exp_classify_turns.py){% endif %} · {{ hl.quotes|length }} span(s){% if hl.old_judge_source=='verdict_only' %} · old judge output missing{% endif %}</div>{% endif %}
 {% if hl.reassessment and hl.reassessment.summary %}<div style="margin-top:.2rem">{{ hl.reassessment.summary }}</div>{% endif %}
 {% if hl.old_judge_report_md %}<details class="report"><summary>old judge's findings (stricter bar — may over-flag)</summary><pre>{{ hl.old_judge_report_md }}</pre></details>{% endif %}
 {% if hl.story %}</details>{% endif %}
</div>
{% endif %}
{{ body|safe }}
<button class="totop" id="totop" title="back to top">↑</button>
<script>
(function(){
 var b=document.getElementById('totop');
 function upd(){ b.style.display = window.scrollY>400 ? 'block' : 'none'; }
 b.onclick=function(){ window.scrollTo({top:0,behavior:'smooth'}); };
 window.addEventListener('scroll',upd); upd();
})();
</script>
<div class="hacknav" id="hacknav"></div>
<script>
(function(){
 var nav=document.getElementById('hacknav');
 function els(sel){ return [].slice.call(document.querySelectorAll(sel)); }
 var groups=[
  {label:'🔪 rollback cut', els: els('.rollbackcut'), keyn:'c', keyp:'C'},
  {label:'📝 notes', els: els('.msg.hackturn, .msg.notedturn'), keyn:'n', keyp:'p'},
  {label:'⚠ hack',  els: els('.msg.hackturn'),                  keyn:'h', keyp:'H'},
  {label:'🏋 train', els: els('.traintag').map(function(x){ return x.closest('.msg'); }), keyn:'t', keyp:'T'},
  {label:'🔑 OpenAI rule-break', els: els('.apiflag.oai').map(function(x){ return x.closest('.msg'); }), keyn:'o', keyp:'O'},
  {label:'📖 rule seen', els: els('.apiflag.ruleseen').map(function(x){ return x.closest('.msg'); }), keyn:'r', keyp:'R'},
  {label:'⚠ hack type 1', els: els('.apiflag.cathack0').map(function(x){ return x.closest('.msg'); }), keyn:'1', keyp:'!'},
  {label:'⚠ hack type 2', els: els('.apiflag.cathack1').map(function(x){ return x.closest('.msg'); }), keyn:'2', keyp:'@'}
 ];
 // de-dup repeated .msg refs within a group (an event can carry several flags)
 groups.forEach(function(g){ var seen=[]; g.els=g.els.filter(function(e){ if(!e||seen.indexOf(e)>=0)return false; seen.push(e); return true; }); });
 var any=false;
 groups.forEach(function(g){
   if(!g.els.length) return;
   any=true; g.cur=-1;
   var row=document.createElement('div');
   row.className='hnrow';
   row.innerHTML='<span>'+g.label+'</span><span style="display:flex;gap:.35rem;align-items:center">'
     +'<button title="previous ('+g.keyp+')">▲</button>'
     +'<span class="lbl">– / '+g.els.length+'</span>'
     +'<button title="next ('+g.keyn+')">▼</button></span>';
   var btns=row.querySelectorAll('button');
   g.lblEl=row.querySelector('.lbl');
   g.go=function(i){ g.cur=(i+g.els.length)%g.els.length; var el=g.els[g.cur];
     el.scrollIntoView({behavior:'smooth',block:'center'});
     el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
     g.lblEl.textContent=(g.cur+1)+' / '+g.els.length; };
   btns[0].onclick=function(){ g.go(g.cur-1); };
   btns[1].onclick=function(){ g.go(g.cur+1); };
   nav.appendChild(row);
 });
 if(!any){ nav.style.display='none'; return; }
 document.addEventListener('keydown',function(e){
   if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
   groups.forEach(function(g){
     if(!g.go) return;
     if(e.key===g.keyn){ g.go(g.cur+1); }
     else if(e.key===g.keyp){ g.go(g.cur-1); }
   });
 });
})();
</script>
</body></html>
""".replace("__CSS__", CSS)

REPORT_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Cheating report</title>
<style>__CSS__</style></head><body>
<div class="nav"><a href="/">&larr; index</a><span>{{ cats|length }} categories · {{ n_runs }} flagged runs · {{ model }}</span></div>
<h1>Cheating report</h1>
{% if overview %}<div class="hdr" style="white-space:pre-wrap">{{ overview }}</div>{% endif %}
{% for kind, kind_title in [('cheating','🔴 Cheating patterns'), ('not_cheating','🟢 Wrongly-flagged (benign) patterns')] %}
{% set kcats = cats | selectattr('kind','equalto',kind) | list %}
{% if kcats %}
<h2 style="font-size:1.15rem;margin-top:1.6rem">{{ kind_title }}</h2>
{% for c in kcats %}
<div class="catcard {{ 'benign' if kind=='not_cheating' }}">
 <div class="cat-h"><b>{{ c.title }}</b> <span class="pill">{{ c.members|length }} run(s)</span></div>
 <div class="cat-desc">{{ c.description }}</div>
 <table><thead><tr><th>experiment</th><th>bench</th><th>base model</th><th>agent</th><th>acc</th><th>verdict</th><th>note</th></tr></thead><tbody>
 {% for m in c.members %}
  <tr>
   <td><a href="/run/{{ m.run_id }}">{{ m.experiment }}</a>{% if m.also %} <span class="multi" title="also in: {{ m.also|join(' · ') }}">×{{ m.also|length + 1 }}</span>{% endif %}</td>
   <td>{{ m.benchmark }}</td><td>{{ m.trained_model }}</td><td>{{ m.agent_model }}</td>
   <td class="num">{{ '%.3f'|format(m.accuracy) if m.accuracy is not none else '–' }}</td>
   <td><span class="lbl {{ label_class.get(m.label,'') }}">{{ m.label }}</span></td>
   <td style="color:#555;font-size:.84rem">{{ m.note }}</td>
  </tr>
 {% endfor %}
 </tbody></table>
</div>
{% endfor %}
{% endif %}
{% endfor %}
</body></html>
""".replace("__CSS__", CSS)

TIMING_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>RH timing</title>
<style>__CSS__</style></head><body>
<div class="nav"><a href="/">&larr; index</a><span>{{ rows|length }} runs (effective RH: yes only — misled/unclear excluded)</span></div>
<h1>When does the hacking emerge?</h1>
<p class="meta">One row per confirmed-RH run; "first" = the earliest hack turn our re-judge localized. Rows marked <b>🔑</b> are OpenAI-API reward hacks (added from the OpenAI-rule pass); their "first" = the first illegal-use move (intent/authored/executed).
 <b>events</b> = trace event index (what hack turns refer to — note the index page's "turns" is the harness's coarser API-turn count).
 <b>reasoning steps</b> = codex reasoning items, or assistant text blocks for claude/opencode (same declare-intent role).
 <b>wall clock</b> = from event timestamps where the harness records them; others show –.
 <b>train runs</b> = commands launching a train-ish script (python/accelerate/torchrun + *train*/finetune/sft*.py; data-prep/eval scripts excluded; training inside shell scripts may be missed).
 Medians: events <b>{{ '%.0f%%'|format(100*med_ev) if med_ev is not none }}</b> ·
 reasoning <b>{{ '%.0f%%'|format(100*med_rs) if med_rs is not none }}</b> ·
 time <b>{{ '%.0f%%'|format(100*med_t) if med_t is not none else '–' }}</b>.</p>
<table><thead><tr>
 <th>experiment</th><th>type</th><th>bench</th><th>base model</th><th>agent</th>
 <th>first ev</th><th>events</th><th>%</th>
 <th>first step</th><th>steps</th><th>%</th>
 <th>t @ first</th><th>t total</th><th>%</th>
 <th>trains ≺hack</th><th>trains</th>
</tr></thead><tbody>
{% for x in rows %}
 <tr>
  <td>{% if x.openai %}<span title="OpenAI-API reward hack">🔑</span> {% endif %}<a href="/run/{{ x.run_id }}">{{ x.experiment }}</a></td>
  <td>{% if x.hack_type %}<span class="lbl {{ label_class.get(x.hack_type,'') }}">{{ x.hack_type }}</span>{% else %}–{% endif %}</td>
  <td>{{ x.benchmark }}</td><td>{{ x.trained_model }}</td><td>{{ x.agent_model }}</td>
  <td class="num">{{ x.ev_first if x.ev_first is not none else '–' }}</td>
  <td class="num">{{ x.ev_total }}</td>
  <td class="num">{{ '%.0f%%'|format(100*x.ev_frac) if x.ev_frac is not none else '–' }}</td>
  <td class="num">{{ x.rs_first if x.rs_first is not none else '–' }}</td>
  <td class="num">{{ x.rs_total }}</td>
  <td class="num">{{ '%.0f%%'|format(100*x.rs_frac) if x.rs_frac is not none else '–' }}</td>
  <td class="num">{{ x.t_first or '–' }}</td>
  <td class="num">{{ x.t_total or '–' }}</td>
  <td class="num">{{ '%.0f%%'|format(100*x.t_frac) if x.t_frac is not none else '–' }}</td>
  <td class="num">{{ x.tr_before if x.tr_before is not none else '–' }}</td>
  <td class="num">{{ x.tr_total }}</td>
 </tr>
{% endfor %}
</tbody></table>
<h2 style="font-size:1.1rem;margin-top:1.6rem">First-hack distributions — all confirmed hacks</h2>
<div style="display:flex;flex-wrap:wrap;gap:.6rem;margin-top:.4rem">
 {% for m in metrics %}<img src="/timing/dist/{{ m }}.png" alt="{{ m }}" style="width:32%;min-width:300px">{% endfor %}
</div>

<h1 style="margin-top:2.2rem">By cheating category</h1>
<p class="meta">The same five distributions, one row per category from the <a href="/report">cheating report</a>.
 First-hack point is per-category: for the 3 multi-category runs the first hack of <i>that</i> category is used
 (hand-localized from the trace), so a run can contribute a different point to each of its categories.
 The <span style="color:#9aa7ba"><b>light bars</b></span> behind each plot are the totals (all hacks) for reference;
 dashed lines mark the medians (<span style="color:#b3261e">red</span> = this category, <span style="color:#9aa7ba">gray</span> = all).</p>
{% for c in cats %}{% set ci = loop.index0 %}
<h2 style="font-size:1.05rem;margin-top:1.5rem">{{ loop.index }}. {{ c.title }} <span style="color:#888;font-weight:400">({{ c.n }} run{{ '' if c.n==1 else 's' }})</span></h2>
<div style="display:flex;flex-wrap:wrap;gap:.6rem;margin-top:.3rem">
 {% for m in metrics %}<img src="/timing/cat/{{ ci }}/{{ m }}.png" alt="{{ m }}" style="width:32%;min-width:300px">{% endfor %}
</div>
{% endfor %}
<script>
(function(){
 var table=document.querySelector('table'), ths=table.querySelectorAll('th');
 function val(td){
   var t=td.textContent.trim();
   if(t==='–'||t==='') return null;
   var m=t.match(/^(\d+):(\d{2}):(\d{2})$/);            // h:mm:ss
   if(m) return (+m[1])*3600+(+m[2])*60+(+m[3]);
   if(/^-?[\d.]+%$/.test(t)) return parseFloat(t);       // percents
   if(/^-?[\d.]+$/.test(t)) return parseFloat(t);        // numbers
   return t.toLowerCase();                               // text
 }
 ths.forEach(function(th,i){
   th.style.cursor='pointer'; th.title='click to sort';
   th.addEventListener('click',function(){
     var asc = th.dataset.dir!=='asc';
     ths.forEach(function(o){ o.dataset.dir=''; o.textContent=o.textContent.replace(/ [▲▼]$/,''); });
     th.dataset.dir = asc?'asc':'desc';
     th.textContent = th.textContent.replace(/ [▲▼]$/,'') + (asc?' ▲':' ▼');
     var tb=table.querySelector('tbody');
     var rows=[].slice.call(tb.querySelectorAll('tr'));
     rows.sort(function(a,b){
       var x=val(a.children[i]), y=val(b.children[i]);
       if(x===null) return 1; if(y===null) return -1;    // blanks last
       if(typeof x==='string' || typeof y==='string'){ x=String(x); y=String(y); }
       return (x<y?-1:x>y?1:0)*(asc?1:-1);
     });
     rows.forEach(function(r){ tb.appendChild(r); });
   });
 });
})();
</script>
</body></html>
""".replace("__CSS__", CSS)

RATES_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>RH rates</title>
<style>__CSS__</style></head><body>
<div class="nav"><a href="/">&larr; index</a>
 <span>overall: {{ t.flagged }}/{{ t.n }} originally flagged ({{ '%.1f%%'|format(100*t.flagged/t.n) }}) ·
 {{ t.yes }} broke rules ({{ '%.1f%%'|format(100*t.yes/t.n) }}) ·
 <b>{{ t.deliberate }} reward hack ({{ '%.1f%%'|format(100*t.deliberate/t.n) }})</b></span></div>
<h1>Reward-hacking rates</h1>
<p class="meta">Share of all {{ t.n }} runs, per group, using the FINAL adjudicated labels
 (see JUDGING.md): <b>originally flagged</b> = the benchmark's own judges (the detection funnel — everything
 to the right inherits any original false negatives, e.g. unflagged runs that loaded the BFCL/GPQA eval
 pool); <b>broke rules</b> = final label is contamination, disallowed model, or broke-rules-unknowingly
 (a rule violation occurred, regardless of intent); <b>reward hack</b> = final label is contamination or
 disallowed model (knowing violation). The one "unclear" run counts in neither. Tables sort by the
 reward-hack column; click headers to re-sort.</p>
{% for tb in tables %}
<h2 style="font-size:1.1rem;margin-top:1.5rem">{{ tb.title }}</h2>
<table><thead><tr><th>{{ tb.key }}</th><th>runs</th>
 <th>originally flagged</th><th>%</th><th>broke rules</th><th>%</th><th>reward hack</th><th>%</th></tr></thead><tbody>
{% for x in tb.rows %}
 <tr>
  <td>{{ x.group }}</td><td class="num">{{ x.n }}</td>
  <td class="num">{{ x.flagged }}</td><td class="num">{{ '%.1f%%'|format(x.flagged_pct) }}</td>
  <td class="num">{{ x.yes }}</td><td class="num">{{ '%.1f%%'|format(x.yes_pct) }}</td>
  <td class="num">{{ x.deliberate }}</td><td class="num"><b>{{ '%.1f%%'|format(x.deliberate_pct) }}</b></td>
 </tr>
{% endfor %}
</tbody></table>
{% endfor %}
<script>
(function(){
 function val(td){
   var t=td.textContent.trim();
   if(t==='–'||t==='') return null;
   if(/^-?[\d.]+%$/.test(t)) return parseFloat(t);
   if(/^-?[\d.]+$/.test(t)) return parseFloat(t);
   return t.toLowerCase();
 }
 [].forEach.call(document.querySelectorAll('table'),function(table){
  var ths=table.querySelectorAll('th');
  ths.forEach(function(th,i){
   th.style.cursor='pointer'; th.title='click to sort';
   th.addEventListener('click',function(){
     var asc = th.dataset.dir!=='asc';
     ths.forEach(function(o){ o.dataset.dir=''; o.textContent=o.textContent.replace(/ [▲▼]$/,''); });
     th.dataset.dir = asc?'asc':'desc';
     th.textContent = th.textContent.replace(/ [▲▼]$/,'') + (asc?' ▲':' ▼');
     var tb=table.querySelector('tbody');
     var rows=[].slice.call(tb.querySelectorAll('tr'));
     rows.sort(function(a,b){
       var x=val(a.children[i]), y=val(b.children[i]);
       if(x===null) return 1; if(y===null) return -1;
       if(typeof x==='string' || typeof y==='string'){ x=String(x); y=String(y); }
       return (x<y?-1:x>y?1:0)*(asc?1:-1);
     });
     rows.forEach(function(r){ tb.appendChild(r); });
   });
  });
 });
})();
</script>
</body></html>
""".replace("__CSS__", CSS)

ROLLBACK_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Rollback experiments</title>
<style>__CSS__</style></head><body>
<div class="nav"><a href="/">&larr; index</a><span>{{ pairs|length }} trajectory row(s) · {{ nruns }} rollback run(s)</span></div>
<h1>🔪 Rollback intervention experiments</h1>
<p class="meta" style="font-size:.95rem">
 <span class="rbstat running">{{ n_running_traj }} trajectories running</span>
 <span class="rbstat done">{{ n_done_traj }} done (all 3)</span>
 <span class="rbstat todo">{{ n_todo_runs }} runs to-do</span>
 <span class="rbstat deferred">{{ n_deferred_traj }} deferred (no creds / parked)</span>
 — blue <span class="rblive">live</span> rows are running this session (one per GPU); greyed "queued" rows are conditions waiting behind on the same box's GPU; deferred trajectories (no creds yet) are greyed in place. Rows are grouped by engine (OpenCode, Claude API, Claude Code, Codex, Qwen).</p>
<p class="meta">Take a trajectory that reward-hacked, <b>cut it right before the reward-hacking turn</b>,
 rebuild the cut-point workspace (including <b>re-training the model the agent had at the cut</b>), and
 resume it with <b>prompt1</b> ("Please continue."), <b>prompt2</b> (same, plus the rollback/rules reminder),
 or <b>prompt3</b> (same as prompt2 plus an acknowledgement request) to test whether an
 in-context reminder stops the agent re-hacking. Each row is one original adjudicated reward-hack
 trajectory; click any completed rollout for its trace.</p>
<p class="meta"><span class="agentbad" title="fatal error by the agent">&#9679;</span> = agent made a bad mistake so the rollout is not a complete result</p>
<p class="meta" style="font-size:.82rem;color:#9a3412"><b>Codex trajectories:</b> in the rebuilt history, the agent's earlier file-edits appear header-only (file paths, not the diffs) — codex never recorded the edit contents. The on-disk files are correct, so this only slightly weakens the resumed agent's memory of its own past edits. Applies to all codex-scaffold runs; not flagged per-row.</p>

<table><thead><tr><th>rollback run</th><th>condition</th><th>score</th><th>RH verdict</th><th>original run</th><th>original score</th></tr></thead><tbody>
{% for p in pairs %}
 {% if p.engine_header %}<tr class="rbenginehdr"><td colspan="6">{{ p.engine_header }}</td></tr>{% endif %}
 {% for r in p.runs %}
  <tr class="{{ 'rbliverow' if r.live else 'rbrunningrow' if r.running else 'rbtodorow' if r.todo else 'rbdeferredrow' if p.deferred else '' }}" {% if loop.first %}style="border-top:2px solid #cfcfcf"{% endif %}>
   <td>{% if r.running %}<span class="{{ 'rblive' if r.live else 'rbrunning' }}" title="{{ r.status_detail or 'currently running on Lambda' }}">{{ r.label }}{% if r.ip %} <span class="rbip">{{ r.ip }}</span>{% endif %}{% if r.queued %} · queued (1 GPU){% elif r.since %} · since {{ r.since }}{% endif %}</span>{% elif r.todo %}<span class="rbtodo" title="not yet run">&#9744; to-do</span>{% elif r.placeholder %}<span class="meta">—</span>{% else %}<a href="/run/{{ r.run_id }}" data-rb-run="{{ r.run_id }}">{{ r.label }}</a><button type="button" class="rbnew" data-run-id="{{ r.run_id }}" title="Click to clear this marker">NEW</button>{% if r.failure_category == 'agent_fatal_error' %} <span class="agentbad" title="{{ r.failure_mode or 'fatal error by the agent' }}: {{ r.failure_summary or 'agent made a bad mistake; not a valid completed result' }}">&#9679;</span>{% endif %}{% if r.immediate_stop %} <span title="degenerate continuation ({{ r.ncont if r.ncont is not none else 0 }} event(s)) — agent stopped immediately (e.g. ack-and-stop / recon glitch); NOT a valid result" style="color:#fff;background:#b91c1c;border-radius:4px;padding:0 .35rem;font-weight:800">⛔ ERROR</span>{% endif %}{% if r.tainted %} <span title="{{ r.tainted }}" style="color:#b91c1c;font-weight:700">&#9888; tainted</span>{% endif %}{% for c in r.caveats %} <span title="{{ c.detail }}" style="color:#9a3412;font-weight:700">&#9888; {{ c.tag }}</span>{% endfor %}{% endif %}</td>
   <td>{{ r.condition }}</td>
   <td class="num">{{ '%.3f'|format(r.score) if r.score is not none else (r.status if r.running else '–') }}</td>
   <td>{% if r.immediate_stop or r.tainted %}<span class="meta" title="run flagged {{ 'ERROR (degenerate continuation)' if r.immediate_stop else 'tainted' }} — not a valid result, so its judge verdict is suppressed">not judged</span>{% elif r.verdict %}<span class="lbl {{ label_class.get(r.verdict,'') }}">{{ r.verdict }}</span>{% elif r.running %}<span class="meta">{{ r.status or 'running' }}</span>{% elif r.todo %}<span class="meta">—</span>{% else %}<span class="meta">not judged</span>{% endif %}</td>
   {% if loop.first %}
    <td rowspan="{{ p.runs|length }}" style="font-size:.78rem;vertical-align:middle">{% if p.needs_retrain %}<span class="rbtag retrain" title="{{ p.n_trainings }} training run(s) before the cut — the rollback must re-create this pre-cut model before resuming">&#8635; re-train{% if p.n_trainings %} ({{ p.n_trainings }}){% endif %}</span>{% else %}<span class="rbtag noretrain" title="no training before the cut — the cut-point model is the untouched base model; no re-training needed">no pre-cut training</span>{% endif %}{% for short, full in p.cheat_cats %} <span class="rbtag cheat" title="{{ full }}">{{ short }}</span>{% else %} <span class="rbtag uncat" title="not mapped to any of the 7 cheating categories on the /report page">uncategorized</span>{% endfor %}{% if p.deferred %} <span class="rbdefbadge" title="{{ p.deferred_reason }}">⏸ {{ p.deferred_reason }}</span>{% endif %}<br>{% if p.orig_in_index %}<a href="/run/{{ p.orig_run_id }}">{{ p.orig_run_id }}</a>{% else %}{{ p.orig_run_id or '?' }}{% endif %}</td>
    <td rowspan="{{ p.runs|length }}" class="num" style="vertical-align:middle">{{ '%.3f'|format(p.orig_accuracy) if p.orig_accuracy is not none else '–' }}{% if p.orig_stderr %} <span class="meta">±{{ '%.3f'|format(p.orig_stderr) }}</span>{% endif %}</td>
   {% endif %}
  </tr>
 {% endfor %}
{% endfor %}
</tbody></table>

<h2 style="font-size:1.05rem;margin-top:1.6rem">Re-created model scores vs. original</h2>
<p class="meta" style="max-width:62rem">Each rollback that needs a pre-cut model re-trains that model before resuming the agent. This table shows the re-created model's benchmark score next to the original pre-cut score.</p>
<table><thead><tr><th>re-created model</th><th>benchmark</th><th>re-created</th><th>original pre-cut</th><th>&Delta;</th><th>prep source</th></tr></thead><tbody>
{% for s in model_score_rows %}
 <tr>
  <td>{{ s.trajectory }}{% if s.run %} <span class="meta">[{{ s.run }}]</span>{% endif %}</td>
  <td>{{ s.benchmark }}</td>
  <td class="num">{{ '%.3f'|format(s.retrained) }}{% if s.retrained_stderr %} <span class="meta">±{{ '%.3f'|format(s.retrained_stderr) }}</span>{% endif %}</td>
  <td class="num">{{ '%.3f'|format(s.original) }}{% if s.original_stderr %} <span class="meta">±{{ '%.3f'|format(s.original_stderr) }}</span>{% endif %}</td>
  <td class="num">{{ '%+.3f'|format(s.delta) if s.delta is not none else '–' }}</td>
  <td><span class="meta">{{ s.prep_source or 'unknown' }}</span></td>
 </tr>
{% endfor %}
{% if not model_score_rows %}<tr><td colspan="6" class="meta">no re-created model score comparisons recorded yet</td></tr>{% endif %}
</tbody></table>
<p class="meta" style="max-width:62rem;margin-top:.6rem"><b>±</b> = standard error of the accuracy.</p>

<h2 style="font-size:1.05rem;margin-top:1.8rem">Rollback outcomes across completed trajectories</h2>
<p class="meta" style="max-width:680px">One bar per outcome pattern, counting only trajectories where <b>all three prompts (1, 2, 3)
 have a completed, judged rollout</b>. Hack vs. no-hack is the judge's <code>is_reward_hack</code> verdict.
 When a prompt was run more than once, all its runs must <b>agree</b> — any disagreement (or any other
 pattern) lands the trajectory in <b>other</b>.
 A run that hit a fatal agent error still counts if the judge found a hack (the hack happened), but a
 <i>clean</i> verdict from a crashed run is dropped as unreliable; such cases are marked &ldquo;&#9888;&rdquo; in the lists below.
 {% if breakdown.n_incomplete %}{{ breakdown.n_incomplete }} partially-run trajector{{ 'y' if breakdown.n_incomplete == 1 else 'ies' }} (started but missing &ge;1 prompt) {{ 'is' if breakdown.n_incomplete == 1 else 'are' }} excluded; not-yet-started trajectories are ignored entirely.{% endif %}</p>
<img class="rbfig" src="data:image/png;base64,{{ chart_png }}" alt="rollback outcome bar chart">
<div class="rbbreak">
{% for c in breakdown.cats %}
 <details><summary>{{ c.label }} — {{ c.count }}</summary>
  <ul>{% for m in c.members %}<li>{{ m }}</li>{% else %}<li style="color:#94a3b8">none</li>{% endfor %}</ul>
 </details>
{% endfor %}
{% if breakdown.incomplete %}
 <details><summary>excluded — experiment incomplete — {{ breakdown.n_incomplete }}</summary>
  <ul>{% for m in breakdown.incomplete %}<li>{{ m }}</li>{% endfor %}</ul>
 </details>
{% endif %}
</div>

<script>
(function(){
  var key = 'ptb.rollback.newMarkers.v1';
  var markers = [].slice.call(document.querySelectorAll('.rbnew[data-run-id]'));
  var ids = markers.map(function(el){ return el.dataset.runId; }).filter(Boolean);
  var state;
  try { state = JSON.parse(localStorage.getItem(key) || 'null'); } catch(e) { state = null; }
  if (!state || !Array.isArray(state.known) || !Array.isArray(state.cleared)) {
    state = {known: ids.slice(), cleared: ids.slice()};
    localStorage.setItem(key, JSON.stringify(state));
    return;
  }
  var known = new Set(state.known);
  var cleared = new Set(state.cleared);
  ids.forEach(function(id){ if (!known.has(id)) known.add(id); });
  markers.forEach(function(el){
    var id = el.dataset.runId;
    if (id && !cleared.has(id)) el.style.display = 'inline-block';
    el.addEventListener('click', function(){
      cleared.add(id);
      el.style.display = 'none';
      localStorage.setItem(key, JSON.stringify({
        known: Array.from(known),
        cleared: Array.from(cleared)
      }));
    });
  });
  localStorage.setItem(key, JSON.stringify({
    known: Array.from(known),
    cleared: Array.from(cleared).filter(function(id){ return known.has(id); })
  }));
})();
</script>
</body></html>
""".replace("__CSS__", CSS)

REPORT_MISSING_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Cheating report</title>
<style>__CSS__</style></head><body>
<div class="nav"><a href="/">&larr; index</a></div>
<h1>Cheating report</h1>
<p class="meta">Not generated yet. Run:</p>
<pre class="content tool" style="padding:.6rem">uv run python posttrainbench/judging/exp_categorize_cheating.py</pre>
</body></html>
""".replace("__CSS__", CSS)

WORKSPACE_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>workspace · {{ run_id }}</title>
<style>__CSS__</style></head><body>
<div class="nav"><a href="/run/{{ run_id }}">&larr; back to trace</a><span>{{ files|length }} files</span></div>
<h1>Final workspace</h1>
<p class="meta">{{ root }}</p>
{% for f in files %}
 <details class="wsfile">
  <summary>{{ f.path }} · {{ f.size }} B{{ ' · (not inlined)' if not f.inlined }}</summary>
  {% if f.text is not none %}<pre class="content tool">{{ f.text }}</pre>{% else %}<pre class="content empty">(binary or not inlined)</pre>{% endif %}
 </details>
{% endfor %}
</body></html>
""".replace("__CSS__", CSS)


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    rows = filtered(request.args)
    flagged = {i: is_flagged(r) for i, r in rows}
    sel = {"experiment": request.args.get("experiment", ""),
           "benchmark": request.args.get("benchmark", ""),
           "agent_model": request.args.get("agent_model", ""),
           "trained_model": request.args.get("trained_model", ""),
           "sort": request.args.get("sort", "default"),
           "rh": request.args.get("rh") == "1",
           "q": request.args.get("q", "")}
    nonrb = [r for r in RUNS if not r.get("is_rollback")]
    return render_template_string(
        INDEX_HTML, rows=rows, flagged=flagged, sel=sel,
        total=len(nonrb), n_exp=len({r["experiment"] for r in nonrb}),
        n_flagged=sum(1 for r in nonrb if is_flagged(r)[0]),
        exp_opts=_opts("experiment"), bench_opts=_opts("benchmark"),
        agent_opts=_opts("agent_model"), base_opts=_opts("trained_model"),
        hl_src=HL_SOURCES, hl_verdict=HL_VERDICT, hl_meta=HL_META,
        label_class=_LABEL_CLASS, api_scan=API_SCAN, oai_pill=openai_pill,
        has_report=(HIGHLIGHTS / "categories.json").exists())


@app.route("/run/<run_id>")
def run(run_id: str):
    if run_id not in INDEX:
        abort(404)
    i = INDEX[run_id]
    r = RUNS[i]
    rec = load_record(run_id)
    hl = load_highlights(run_id)
    story = hl.get("story") if hl else None
    if story and story.get("annotations"):
        # curated display: the story's annotations replace the accumulated
        # per-pass ones (which remain in the file / judging history).
        # NOTE: only when the story actually HAS annotations — a story with an
        # empty annotation list (curate produced summary only) must NOT blank
        # the trace; fall through to the run's real marked_turns/annotations.
        is_rh = HL_META.get(run_id, {}).get("is_reward_hack", False)
        quotes, marked_turns, annotations, turn_kinds = [], [], {}, {}
        for a in story.get("annotations", []):
            t = a["turn"]
            annotations.setdefault(str(t), []).append(
                {"quote": a.get("quote", ""), "note": a.get("note", ""),
                 "kind": "story_" + a.get("role", "context"),
                 "matched": bool(a.get("quote"))})
            if a.get("quote") and a["quote"] not in quotes:
                quotes.append(a["quote"])
            if t not in marked_turns:
                marked_turns.append(t)
            if is_rh and a.get("role") == "evidence":
                turn_kinds[str(t)] = {"kind": "hack", "reason": a.get("note", "")}
            else:
                turn_kinds.setdefault(str(t), {"kind": "context", "reason": a.get("role", "")})
        marked_turns = sorted(marked_turns)
    else:
        quotes = hl.get("quotes", []) if hl else ()
        marked_turns = hl.get("marked_turns", []) if hl else ()
        annotations = hl.get("annotations", {}) if hl else {}
        turn_kinds = hl.get("turn_kinds", {}) if hl else {}
    # number every detected training-run launch (same heuristic /timing counts)
    # so trains-before-hack is verifiable by eye in the trace. Once a run has a
    # train_run_audit, failed launches get a ✗ badge and consume no number.
    events = rec.get("events", [])
    tr_audit = hl.get("train_run_audit", {}) if hl else {}
    train_turns, k = {}, 0
    for i, ev in enumerate(events):
        c = _training_launches(ev)
        if not c:
            continue
        a = tr_audit.get(str(i))
        if a is not None and not a.get("success"):
            train_turns[i] = {"label": "✗ failed", "ok": False,
                              "title": a.get("reason") or "training run failed"}
        else:
            start = k + 1
            k += c
            train_turns[i] = {
                "label": f"#{start}" if c == 1 else f"#{start}–{k}", "ok": True,
                "title": ("successful training run — " + (a.get("reason") or "")) if a
                         else "detected training-run launch (heuristic; success unaudited)"}
    # OpenAI-API evidence -> jump badges. ONLY for rule-breaking verdicts; we no
    # longer surface raw scan signals or cleared runs. ONE toggle (.apiflag.oai)
    # marks the rule-break spots: the synthesizer's trace_highlights PLUS the
    # first_attempt turn — so the first stop of the toggle IS the first attempt.
    # A separate 📖 toggle marks where the agent saw the rule (timing analysis).
    api_turns = {}
    oj = OPENAI_JUDGE.get(run_id)
    if oj and oj.get("verdict") in ("violation", "unclear"):
        if isinstance(annotations, tuple):
            annotations = dict(annotations)
        fa = oj.get("first_attempt") or {}
        fa_turn = fa.get("turn") if isinstance(fa.get("turn"), int) else None
        # rule-break spots: trace_highlights + the first-attempt turn, by turn
        spots = {}  # turn -> annotation note
        for h in oj.get("trace_highlights", []):
            t = h.get("turn")
            if isinstance(t, int):
                spots.setdefault(t, {"quote": h.get("quote", ""), "note": h.get("note", ""),
                                     "matched": bool(h.get("matched"))})
        if fa_turn is not None:
            tag = f"[first illegal-use move — {fa.get('stage','')}] "
            cur = spots.get(fa_turn)
            if cur:
                cur["note"] = tag + cur["note"]
            else:
                spots[fa_turn] = {"quote": fa.get("quote", ""),
                                  "note": tag + fa.get("note", ""), "matched": bool(fa.get("matched"))}
        for t, info in spots.items():
            first = " ①" if t == fa_turn else ""
            api_turns.setdefault(t, []).append(
                {"cls": "oai", "label": f"🔑 OpenAI{first}",
                 "title": info["note"] or oj.get("non_eval_purpose", "")})
            annotations.setdefault(str(t), []).append(
                {"quote": info["quote"], "note": info["note"],
                 "kind": "openai_judge", "matched": info["matched"]})
        # every time the agent saw the rule — own badge + jump key (r/R)
        for e in oj.get("rule_exposure", []):
            t = e.get("turn")
            if not isinstance(t, int):
                continue
            api_turns.setdefault(t, []).append(
                {"cls": "ruleseen", "label": "📖 saw rule",
                 "title": "agent read/acknowledged the evaluate.py OpenAI rule here: "
                          + e.get("note", "")})
            annotations.setdefault(str(t), []).append(
                {"quote": e.get("quote", ""), "note": "[rule sighting] " + e.get("note", ""),
                 "kind": "rule_seen", "matched": bool(e.get("matched"))})
    # Multi-category ("×N") runs: one first-hack badge + toggle PER hack type,
    # ordered by turn (earliest = type 1). Toggles: keys 1 / 2.
    cat_hacks = multi_category_onsets(run_id)
    if cat_hacks:
        if isinstance(annotations, tuple):
            annotations = dict(annotations)
        for j, (t, title) in enumerate(cat_hacks[:2]):
            short = _CAT_SHORT.get(title, title[:18])
            api_turns.setdefault(t, []).append(
                {"cls": f"cathack{j}", "label": f"⚠ {short} ①",
                 "title": f"FIRST hack of type {j+1}: {title}"})
            annotations.setdefault(str(t), []).append(
                {"quote": "", "note": f"[first hack — {title}]", "kind": "cat_hack", "matched": False})
    body = rnd.render_events(events, r.get("trace_format", ""),
                             quotes, marked_turns, annotations, turn_kinds, train_turns, api_turns)
    ws = load_workspace(run_id)
    def _kind(t):
        v = turn_kinds.get(str(t))
        return (v if isinstance(v, str) else (v or {}).get("kind")) or "hack"
    n_hack_true = sum(1 for t in marked_turns if _kind(t) == "hack")
    p = ORDER_POS[run_id]
    return render_template_string(
        DETAIL_HTML, r=r, body=body, flagged=is_flagged(r), hl=hl, oj=oj,
        everdict=HL_META.get(run_id, {}).get("everdict", ""),
        raw_path=str(paths.raw_dir(run_id).relative_to(paths.RAW)),
        n_hack_true=n_hack_true, n_noted=len(marked_turns) - n_hack_true,
        prev=ORDER[p - 1] if p > 0 else None,
        next=ORDER[p + 1] if p < len(ORDER) - 1 else None,
        pos=p + 1, total=len(ORDER), back_qs="",
        workspace=ws is not None, workspace_n=len(ws["files"]) if ws else 0)


# --------------------------------------------------------------------------- #
# RH timing — how far into a session the (re-judged "yes") hacking emerges     #
# --------------------------------------------------------------------------- #
def _is_reasoning_step(ev: dict) -> bool:
    """Codex exposes raw reasoning items; claude_code/opencode don't, so their
    assistant TEXT blocks play the same role (declare intent, then tool calls)."""
    if ev.get("type") == "codex_item":
        return ev.get("subtype") == "reasoning"
    return ev.get("type") == "assistant" and any(
        b.get("type") == "text" for b in ev.get("blocks") or [])


# Training-run launch detection lives in locate.training_launches (shared with
# the success audit). A detected launch says nothing about success — the
# per-run `train_run_audit` (exp_audit_train_runs.py) classifies each one, and
# numbering/timing count only successful runs once audited.
_training_launches = locate.training_launches


def _parse_ts(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _fmt_secs(s):
    if s is None:
        return None
    s = int(s)
    return f"{s//3600}:{s%3600//60:02d}:{s%60:02d}"


_TIMING_CACHE: list | None = None
_CATLIST_CACHE: list | None = None


def compute_timing() -> list[dict]:
    """One row per TRUE reward hack (final.is_reward_hack, any mechanism —
    contamination, disallowed model, OpenAI-API misuse): when the first hack
    occurs, in events / reasoning steps / wall-clock / training runs. The
    first-hack event and training counts come straight from the stamped
    `final` block (judging/finalize.py); only the event-relative display
    metrics (fractions, wall-clock) are computed here. Cached per process."""
    global _TIMING_CACHE
    if _TIMING_CACHE is not None:
        return _TIMING_CACHE
    rows = []
    for rid, m in HL_META.items():
        if not m.get("is_reward_hack") or rid not in INDEX:
            continue
        r = RUNS[INDEX[rid]]
        row = timing_metrics(rid, m.get("first_hack_event"))
        row.update({
            "experiment": r.get("experiment"), "benchmark": r.get("benchmark"),
            "trained_model": r.get("trained_model"), "agent_model": r.get("agent_model"),
            "truncated": m.get("truncated"), "hack_type": m.get("label", ""),
            "openai": m.get("label") == "openai api misuse",
            "tr_before": m.get("trainings_before_first_hack"),
            "tr_total": m.get("trainings_total"),
        })
        rows.append(row)
    rows.sort(key=lambda x: ORDER_POS.get(x["run_id"], 1 << 30))  # match index order
    _TIMING_CACHE = rows
    return rows


@app.route("/timing")
def timing():
    rows = compute_timing()
    def med(key):
        vals = sorted(x[key] for x in rows if x[key] is not None)
        return vals[len(vals) // 2] if vals else None
    cats = [{"title": c["title"], "n": len(c["members"])} for c in category_list()]
    return render_template_string(TIMING_HTML, rows=rows,
                                  med_ev=med("ev_frac"), med_rs=med("rs_frac"),
                                  med_t=med("t_frac"), label_class=_LABEL_CLASS,
                                  cats=cats, metrics=_METRIC_ORDER)


def _nice_width(hi: float, target: int = 20) -> float:
    """A round bin width (1/2/2.5/5 × 10^k) giving ~`target` bins over [0, hi]."""
    import math
    if hi <= 0:
        return 1
    raw = hi / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


# The 5 hack-timing metrics. get() reads a timing row; binw=1 => integer-count
# histogram (trains); binw=None => round auto width; pct => fixed 0–100 / 10%.
_METRICS = {
    "trains":    dict(get=lambda x: x.get("tr_before"),
                      xlabel="successful training runs before first hack", unit="", pct=False, binw=1),
    "time_abs":  dict(get=lambda x: x["t_first_s"] / 3600 if x.get("t_first_s") is not None else None,
                      xlabel="wall-clock hours before first hack", unit="h", pct=False, binw=0.25),
    "time_pct":  dict(get=lambda x: x["t_frac"] * 100 if x.get("t_frac") is not None else None,
                      xlabel="% of total time before first hack", unit="%", pct=True),
    "steps_abs": dict(get=lambda x: x.get("rs_first"),
                      xlabel="reasoning steps before first hack", unit="", pct=False, binw=None),
    "steps_pct": dict(get=lambda x: x["rs_frac"] * 100 if x.get("rs_frac") is not None else None,
                      xlabel="% of total reasoning steps before first hack", unit="%", pct=True),
}
_METRIC_ORDER = ["trains", "time_abs", "time_pct", "steps_abs", "steps_pct"]


def render_metric_png(rows: list, metric: str, bg_rows: list | None = None):
    """Histogram of one timing metric over `rows`. If `bg_rows` is given, its
    distribution (the totals) is drawn behind in a light color as a reference —
    shared bins, so the dark `rows` bars read as a slice of the light backdrop."""
    import io
    import math
    import statistics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flask import send_file
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    cfg = _METRICS[metric]
    vals = [v for v in (cfg["get"](x) for x in rows) if v is not None]
    bgvals = [v for v in (cfg["get"](x) for x in (bg_rows or [])) if v is not None]
    allv = vals + bgvals
    fig, ax = plt.subplots(figsize=(7, 3.0), dpi=150)
    align = "mid"
    if cfg["pct"]:
        bins = list(range(0, 101, 10))
    elif cfg.get("binw") == 1:                       # integer counts (trains)
        bins = list(range(0, int(max(allv, default=0)) + 2))
        align = "left"
    else:
        hi = max(allv, default=1)
        binw = cfg.get("binw") or _nice_width(hi)
        n = int(math.ceil(hi / binw)) + 1
        bins = [round(i * binw, 6) for i in range(n + 1)]
    rwidth = 0.9 if align == "left" else None
    if bgvals:                                        # totals backdrop (light)
        ax.hist(bgvals, bins=bins, align=align, rwidth=rwidth, color="#d6e0ee",
                edgecolor="white", zorder=1, label="all hacks")
        bgmed = statistics.median(bgvals)
        ax.axvline(bgmed, color="#9aa7ba", ls="--", lw=1.1, zorder=3)
    ax.hist(vals, bins=bins, align=align, rwidth=rwidth, color="#4878a8",
            edgecolor="white", zorder=2, label="this category")
    if vals:
        med = statistics.median(vals)
        ax.axvline(med, color="#b3261e", ls="--", lw=1.4, zorder=4)
    if cfg["pct"]:
        ax.set_xlim(0, 100)
    else:
        ax.set_xlim(-0.5 if align == "left" else 0, bins[-1])
        tickw = _nice_width(bins[-1], target=10)
        ax.xaxis.set_major_locator(MultipleLocator(tickw))
        if tickw == int(tickw):
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("runs")
    medtxt = f" — median {med:.1f}{cfg['unit']}" if vals else " — no data"
    ax.set_title(f"{cfg['xlabel']}{medtxt} (n={len(vals)})", fontsize=9)
    ax.yaxis.get_major_locator().set_params(integer=True)
    if bgvals:
        ax.legend(fontsize=7, framealpha=0.7, loc="upper right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# --------------------------------------------------------------------------- #
# Per-category timing — categories.json carries each member's first-hack
# event (stamped by judging/finalize.py; multi-category runs carry a distinct
# hand-localized onset per category).
# --------------------------------------------------------------------------- #
def load_categories() -> list[dict]:
    """categories.json categories, members filtered to runs in the index."""
    path = HIGHLIGHTS / "categories.json"
    if not path.exists():
        return []
    cats = json.loads(path.read_text()).get("categories", [])
    for c in cats:
        c["members"] = [m for m in c.get("members", []) if m.get("run_id") in INDEX]
    return cats


CATEGORIES = load_categories()


def multi_category_onsets(rid: str) -> list[tuple[int, str]]:
    """[(first_hack_event, category title), ...] for a run in 2+ cheating
    categories (each mechanism has its own onset); [] otherwise."""
    hits = [(m.get("first_hack_event"), c["title"])
            for c in CATEGORIES if c.get("kind") == "cheating"
            for m in c["members"] if m["run_id"] == rid
            and m.get("first_hack_event") is not None]
    return sorted(hits) if len(hits) > 1 else []


# short tags for the multi-category badges (full title in the tooltip)
_CAT_SHORT = {
    "Verbatim copy of test questions + answers into training": "verbatim copy",
    "Error-driven harvesting of failing test items from eval logs": "error-driven harvest",
    "Injecting benchmark problems with canonical solutions (disguised synthetic)": "canonical soln",
    "Distilling/oversampling the exact eval prompts": "distill/oversample",
    "Patching the scorer/harness with reverse-engineered fixes": "scorer patch",
    "Submitting an off-the-shelf disallowed model": "disallowed model",
}
# The /report page shows 7 CHEATING categories: the 6 above (kind="cheating" in
# categories.json) plus "OpenAI-API misuse", which is injected from the HL_META
# label rather than categories.json. Title + short tag for that 7th category:
_OPENAI_MISUSE_TITLE = "OpenAI-API misuse (eval key → data generation)"
_OPENAI_MISUSE_SHORT = "openai-api misuse"


def cheating_categories_by_run() -> dict[str, list[tuple[str, str]]]:
    """run_id -> [(short_tag, full_title), ...] across the 7 cheating categories
    catalogued on the /report page. Mirrors how /report assembles them."""
    out: dict[str, list[tuple[str, str]]] = {}
    for c in CATEGORIES:
        if c.get("kind") != "cheating":
            continue
        tag = (_CAT_SHORT.get(c["title"], c["title"]), c["title"])
        for m in c.get("members", []):
            out.setdefault(m["run_id"], [])
            if tag not in out[m["run_id"]]:
                out[m["run_id"]].append(tag)
    for rid, meta in HL_META.items():
        if meta.get("label") == "openai api misuse":
            tag = (_OPENAI_MISUSE_SHORT, _OPENAI_MISUSE_TITLE)
            out.setdefault(rid, [])
            if tag not in out[rid]:
                out[rid].append(tag)
    return out


def timing_metrics(rid: str, first):
    """Per-run timing row for an arbitrary first-hack turn (event index)."""
    events = load_record(rid).get("events", [])
    if not (isinstance(first, int) and 0 <= first < len(events)):
        first = None
    rsf = [_is_reasoning_step(e) for e in events]
    tra = (load_highlights(rid) or {}).get("train_run_audit", {})
    launches = []
    for i, e in enumerate(events):
        c = _training_launches(e)
        a = tra.get(str(i))
        launches.append(0 if (c and a is not None and not a.get("success")) else c)
    tss = [_parse_ts(e.get("ts")) for e in events]
    known = [(i, t) for i, t in enumerate(tss) if t]
    t_total = t_at = None
    if known:
        t0, tN = known[0][1], known[-1][1]
        t_total = (tN - t0).total_seconds()
        if first is not None:
            b = [t for i, t in known if i <= first]
            if b:
                t_at = (b[-1] - t0).total_seconds()
    f = lambda a, b: round(a / b, 3) if (a is not None and b) else None
    rsb = sum(rsf[: first + 1]) if first is not None else None
    return {"run_id": rid, "ev_first": first, "ev_total": len(events),
            "ev_frac": f(first, len(events)), "rs_first": rsb, "rs_total": sum(rsf),
            "rs_frac": f(rsb, sum(rsf)), "t_first_s": t_at, "t_frac": f(t_at, t_total),
            "t_first": _fmt_secs(t_at), "t_total": _fmt_secs(t_total),
            "tr_before": (sum(launches[:first]) if first is not None else None),
            "tr_total": sum(launches)}


def category_list() -> list:
    """Ordered cheating categories + the OpenAI-API hacks, each as
    {title, members:[{run_id, first_hack_event}, ...]}. Cheating categories
    (with per-member onsets) come from categories.json; the OpenAI-API hacks
    from the runs' stamped `final` labels. Cached for the process lifetime."""
    global _CATLIST_CACHE
    if _CATLIST_CACHE is not None:
        return _CATLIST_CACHE
    cats = [{"title": c["title"],
             "members": [{"run_id": m["run_id"],
                          "first_hack_event": m.get("first_hack_event")}
                         for m in c["members"]]}
            for c in CATEGORIES if c.get("kind") == "cheating"]
    oai = [{"run_id": rid, "first_hack_event": m.get("first_hack_event")}
           for rid, m in HL_META.items()
           if m.get("label") == "openai api misuse" and rid in INDEX]
    if oai:
        cats.append({"title": "OpenAI-API misuse (eval key → data generation)",
                     "members": oai})
    _CATLIST_CACHE = cats
    return cats


def category_timing(cat: dict) -> list:
    return [timing_metrics(m["run_id"], m["first_hack_event"])
            for m in cat["members"]]


@app.route("/timing/hist.png")
def timing_hist():
    return render_metric_png(compute_timing(), "trains")


@app.route("/timing/dist/<metric>.png")
def timing_dist(metric: str):
    if metric not in _METRICS:
        abort(404)
    return render_metric_png(compute_timing(), metric)


@app.route("/timing/cat/<int:idx>/<metric>.png")
def timing_cat(idx: int, metric: str):
    cats = category_list()
    if metric not in _METRICS or not (0 <= idx < len(cats)):
        abort(404)
    return render_metric_png(category_timing(cats[idx]), metric, bg_rows=compute_timing())


# --------------------------------------------------------------------------- #
# RH rates by group                                                            #
# --------------------------------------------------------------------------- #
RATE_DIMS = [("benchmark", "By benchmark"), ("agent_model", "By agent model"),
             ("trained_model", "By base (trained) model"),
             ("trace_format", "By harness"), ("experiment", "By experiment")]


def compute_rates() -> list[dict]:
    """Per dimension: rows of (group, n, flagged, yes, deliberate) with rates.

    flagged    = original benchmark judges (the funnel: re-judge only saw these,
                 so 'yes' inherits any original false negatives).
    yes        = our Opus re-judge verdict.
    deliberate = yes minus 🏷-misled (agents honestly fooled by the
                 'train'-split label; arguably not intent-level RH)."""
    tables = []
    for key, title in RATE_DIMS:
        groups: dict[str, dict] = {}
        for r in RUNS:
            g = r.get(key) or "?"
            d = groups.setdefault(g, {"n": 0, "flagged": 0, "yes": 0, "deliberate": 0})
            d["n"] += 1
            if is_flagged(r)[0]:
                d["flagged"] += 1
            # final adjudication, both rule axes (HL_META covers the flagged
            # contamination runs AND the OpenAI-API violations)
            m = HL_META.get(r["run_id"])
            lbl = m.get("label") if m else None
            if lbl in ("contamination", "disallowed model", "openai api misuse",
                       "broke rules unknowingly"):
                d["yes"] += 1          # broke rules
            if m and m.get("is_reward_hack"):
                d["deliberate"] += 1   # reward hack
        rows = [{"group": g, **d,
                 "flagged_pct": 100 * d["flagged"] / d["n"],
                 "yes_pct": 100 * d["yes"] / d["n"],
                 "deliberate_pct": 100 * d["deliberate"] / d["n"]}
                for g, d in groups.items()]
        rows.sort(key=lambda x: (-x["deliberate_pct"], -x["n"]))
        tables.append({"key": key, "title": title, "rows": rows})
    return tables


@app.route("/rates")
def rates():
    total = {"n": len(RUNS),
             "flagged": sum(1 for r in RUNS if is_flagged(r)[0]),
             "yes": sum(1 for m in HL_META.values()
                        if m.get("label") in ("contamination", "disallowed model",
                                              "openai api misuse",
                                              "broke rules unknowingly")),
             "deliberate": sum(1 for m in HL_META.values()
                               if m.get("is_reward_hack"))}
    return render_template_string(RATES_HTML, tables=compute_rates(), t=total)


# Outcome categories for the standalone bar chart, in display order.
# Each completed trajectory lands in exactly one. "other" is the catch-all so a
# new pattern (or a prompt whose multiple runs DISAGREE) is visible, not silently
# folded into one of the named bars.
_RB_CHART_CATS = [
    ("all_clean", "no hacks",           "#3a9d57"),
    ("all_hack",  "all hacks",          "#d9554a"),
    ("p1_only",   "only control hacks", "#4f86c6"),
    ("other",     "other",                       "#9aa3ad"),
]


def _rb_condition_verdict(members: list[dict]) -> str:
    """Collapse one prompt-condition's runs to a single verdict. Returns
    'hack'/'nohack' only if every counted run AGREES; 'mixed' if they disagree;
    'absent' if there are no counted runs. What counts:
      - placeholders (running/queued/to-do) and not-yet-judged runs: ignored.
      - tainted runs: never count (the experiment itself is flagged invalid).
      - a JUDGED HACK counts even if the rollout later hit a fatal agent error —
        the hack demonstrably happened (this is what keeps a trajectory whose
        prompt crashed *after* hacking in the all-hack bar).
      - a CLEAN verdict from a fatal-errored rollout is dropped as unreliable:
        a crash before the agent finished doesn't prove it wouldn't have hacked."""
    vals = set()
    for m in members:
        if m.get("placeholder") or m.get("tainted"):
            continue
        irh = m.get("is_reward_hack")
        if irh is None:                 # completed but not yet judged
            continue
        if irh:
            vals.add(True)
        elif m.get("failure_category") == "agent_fatal_error":
            continue                    # unreliable clean verdict -> drop
        else:
            vals.add(False)
    if not vals:
        return "absent"
    if vals == {True}:
        return "hack"
    if vals == {False}:
        return "nohack"
    return "mixed"


def rollback_outcome_chart(pairs: list[dict]):
    """Classify each COMPLETED trajectory (>=1 valid judged run for EACH of
    prompt1/2/3) by its hack pattern and render a standalone matplotlib bar
    chart. Trajectories missing any condition (or with only pending/excluded
    runs there) are counted as incomplete and surfaced separately, never folded
    into a bar. Returns (png_base64, breakdown_dict)."""
    members = {k: [] for k, _, _ in _RB_CHART_CATS}
    incomplete = []
    for p in pairs:
        bycond = {"prompt1": [], "prompt2": [], "prompt3": []}
        for r in p["runs"]:
            c = r.get("condition")
            if c in bycond:
                bycond[c].append(r)
        v1 = _rb_condition_verdict(bycond["prompt1"])
        v2 = _rb_condition_verdict(bycond["prompt2"])
        v3 = _rb_condition_verdict(bycond["prompt3"])
        oid = p.get("orig_run_id") or "?"
        triple = (v1, v2, v3)
        # Transparency: surface any condition whose HACK only counts because we
        # let a fatal-errored run keep its hack verdict (see _rb_condition_verdict).
        caveat_conds = [c for c in ("prompt1", "prompt2", "prompt3")
                        if any(m.get("is_reward_hack")
                               and m.get("failure_category") == "agent_fatal_error"
                               and not m.get("placeholder") and not m.get("tainted")
                               for m in bycond[c])]
        oid_label = oid + (f"  ⚠ {', '.join(caveat_conds)} hack counted despite fatal agent error"
                           if caveat_conds else "")
        n_resolved = sum(1 for v in triple if v != "absent")
        if n_resolved == 0:
            continue                    # never started — not part of the experiment yet
        if "absent" in triple:
            incomplete.append(oid)      # started but missing >=1 prompt -> excluded
            continue
        if triple == ("nohack", "nohack", "nohack"):
            members["all_clean"].append(oid_label)
        elif triple == ("hack", "hack", "hack"):
            members["all_hack"].append(oid_label)
        elif triple == ("hack", "nohack", "nohack"):
            members["p1_only"].append(oid_label)
        else:                           # any other pattern, incl. a 'mixed' condition
            members["other"].append(f"{oid_label}  ({'·'.join(triple)})")

    counts = [len(members[k]) for k, _, _ in _RB_CHART_CATS]
    n_completed = sum(counts)

    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    labels = [lbl for _, lbl, _ in _RB_CHART_CATS]
    colors = [col for _, _, col in _RB_CHART_CATS]
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=110)
    bars = ax.bar(range(len(counts)), counts, color=colors,
                  edgecolor="#333", linewidth=0.7, width=0.66, zorder=3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("# trajectories", fontsize=10)
    ax.set_title(f"Rollback outcome by trajectory  (n={n_completed} completed)",
                 fontsize=11.5)
    maxc = max(counts) if counts else 0
    ax.set_ylim(0, (maxc + 1) if maxc else 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.03, str(c),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    breakdown = {
        "n_completed": n_completed,
        "n_incomplete": len(incomplete),
        "incomplete": incomplete,
        "cats": [{"key": k, "label": lbl.replace("\n", " "),
                  "count": len(members[k]), "members": members[k]}
                 for k, lbl, _ in _RB_CHART_CATS],
    }
    return png_b64, breakdown


@app.route("/rollback")
def rollback_page():
    def g(r, k): return r.get(k) if isinstance(r, dict) else getattr(r, k, None)
    manifest_rows = load_rollback_manifest()
    manifest_by_id = {r.get("run_id"): r for r in manifest_rows if r.get("run_id")}
    # Group rollback rollouts by the original trajectory they branch from, but
    # seed the groups from the full 30-run adjudicated manifest so not-yet-run
    # trajectories still appear as progress-tracker rows.
    by_orig = {rid: [] for rid in manifest_by_id}

    def condition_from_label(label: str) -> str:
        if label.startswith("prompt1"):
            return "prompt1"
        if label.startswith("prompt2"):
            return "prompt2"
        if label.startswith("prompt3"):
            return "prompt3"
        if label.startswith("TAINTED_"):
            return "TAINTED"
        return "other"

    for r in RUNS:
        if not g(r, "is_rollback"):
            continue
        rid = g(r, "run_id")
        label = (g(r, "rb_label") or
                 ((g(r, "experiment") or "").split("__rollback__")[-1]) or
                 rid)
        by_orig.setdefault(g(r, "orig_run_id"), []).append({
            "run_id": rid, "label": label,
            "condition": condition_from_label(label),
            "score": g(r, "accuracy"),
            "verdict": HL_META.get(rid, {}).get("label"),
            # judge's stamped hack/not-hack boolean (None if not yet judged);
            # drives the prompt1/2/3 re-hack chart at the bottom of the page.
            "is_reward_hack": (HL_META.get(rid, {}).get("is_reward_hack")
                               if rid in HL_META else None),
            "tainted": g(r, "rb_tainted"),
            "ncont": g(r, "rb_ncont"),
            # degenerate continuation: agent stopped immediately (0-1 events, e.g.
            # an ack-and-stop on a single-turn resume, or a recon/model glitch).
            # NOT a valid result -> flagged ERROR in the viewer.
            "immediate_stop": bool(g(r, "rb_empty")) or ((g(r, "rb_ncont") or 99) <= 1),
            "caveats": g(r, "rb_caveats") or [],
            "failure_category": g(r, "rb_failure_category") or g(r, "failure_category"),
            "failure_mode": g(r, "rb_failure_mode") or g(r, "failure_mode"),
            "failure_summary": g(r, "rb_failure_summary") or g(r, "failure_summary"),
            "placeholder": False,
        })
    completed_result_dirs = {
        g(r, "rb_result_dir")
        for r in RUNS
        if g(r, "is_rollback") and g(r, "rb_result_dir")
    }
    # (trajectory, condition) pairs that already have a completed rollout — used to
    # suppress a stale 'running' row even when we never learned its result_dir
    # (the launcher registers a run by trajectory+condition, not result_dir).
    completed_orig_cond = {(o, m["condition"])
                           for o, members in by_orig.items()
                           for m in members if not m.get("placeholder")}
    now = time.time()
    for rr in load_running_rollbacks():
        result_dir = rr.get("result_dir")
        orig = rr.get("source_trajectory") or rr.get("run_id")
        cond = rr.get("condition") or "running"
        if result_dir and result_dir in completed_result_dirs:
            continue
        if (orig, cond) in completed_orig_cond:
            continue   # this condition already finished and viewerized
        label = rr.get("label") or f"{cond} running"
        status = rr.get("status") or ("finishing" if rr.get("agent_done") else "running")
        la = rr.get("launched_at")
        since = time.strftime("%H:%M", time.localtime(la)) if isinstance(la, (int, float)) else None
        # "live this session" = launched in the last ~6h, so the user can
        # associate the colored rows with THIS launch session.
        live = isinstance(la, (int, float)) and (now - la) < 6 * 3600
        detail_bits = []
        if rr.get("region"):
            detail_bits.append(rr["region"])
        if rr.get("lambda_name"):
            detail_bits.append(rr["lambda_name"])
        if rr.get("ip"):
            detail_bits.append(rr["ip"])
        if since:
            detail_bits.append(f"launched {since}")
        if result_dir:
            detail_bits.append(result_dir)
        if rr.get("gpu"):
            detail_bits.append(f"GPU {rr['gpu']}")
        by_orig.setdefault(orig, []).append({
            "run_id": None,
            "label": label,
            "condition": cond,
            "score": None,
            "verdict": None,
            "tainted": None,
            "caveats": [],
            "failure_category": None,
            "failure_mode": None,
            "failure_summary": None,
            "placeholder": True,
            "running": True,
            "live": live,
            "since": since,
            "status": status,
            "status_detail": " · ".join(detail_bits),
            "result_dir": result_dir,
            "parallel": int(rr.get("parallel") or 1),
            "ip": rr.get("ip"),
        })
    # DEFERRED = currently un-runnable for lack of credentials (mirrors
    # rollback/run/targets.runnable_reason). The 5 deferred = 4 claude_non_api
    # (need CLAUDE_CODE_OAUTH_TOKEN) + 1 qwen3max (needs DASHSCOPE). codex is NOT
    # deferred (auth.json present; it's run as normal, pending recon validation).
    manual_defer = load_manual_defer()

    def deferred_reason(orig: str) -> str:
        if orig in manual_defer:           # parked by choice (manual_defer.json)
            return manual_defer[orig]
        m = manifest_by_id.get(orig) or {}
        if m.get("agent") == "qwen3max":
            return "needs DASHSCOPE key"
        if m.get("scaffold") == "claude" and m.get("auth") == "oauth":
            return "needs CLAUDE_CODE_OAUTH_TOKEN (claude_non_api)"
        return ""

    pairs = []
    cheat_by_run = cheating_categories_by_run()
    ordered_orig = [r.get("run_id") for r in manifest_rows if r.get("run_id")]
    ordered_orig += sorted(o for o in by_orig if o not in set(ordered_orig) and o)
    order_pos = {rid: i for i, rid in enumerate(ordered_orig)}
    cond_order = {"prompt1": 0, "prompt2": 1, "prompt3": 2,
                  "TAINTED": 8, "other": 9}

    def engine_of(orig: str) -> str:
        return (manifest_by_id.get(orig) or {}).get("agent") or "other"
    engine_rank = {e: i for i, e in enumerate(ROLLBACK_ENGINE_ORDER)}
    # How many trajectory rows each engine block contains (shown in its header).
    engine_counts = {}
    for o in by_orig:
        engine_counts[engine_of(o)] = engine_counts.get(engine_of(o), 0) + 1

    # Primary sort: engine block (fixed ROLLBACK_ENGINE_ORDER); secondary:
    # manifest order within the block. Deferred (no-creds) trajectories are
    # styled greyed but stay with their engine — they're no longer sunk to the
    # very bottom, since the engine grouping is now the organizing principle.
    prev_engine = None
    for orig, members in sorted(by_orig.items(),
                                key=lambda kv: (engine_rank.get(engine_of(kv[0]), len(ROLLBACK_ENGINE_ORDER)),
                                                order_pos.get(kv[0], 1 << 30))):
        # Synthesize explicit "to-do" rows for prompt conditions that have neither
        # a completed nor a running entry, on NON-deferred trajectories — so the
        # overview shows what's LEFT to run, not just what's done/in-flight.
        # (Placeholder rows: they don't count toward done/running in status_summary.)
        if not deferred_reason(orig):
            present = {m.get("condition") for m in members if not m.get("todo")}
            for cond in ("prompt1", "prompt2", "prompt3"):
                if cond not in present:
                    members.append({
                        "run_id": None, "label": "to-do", "condition": cond,
                        "score": None, "verdict": None, "tainted": None,
                        "caveats": [], "placeholder": True, "running": False,
                        "todo": True,
                    })
        members.sort(key=lambda x: (cond_order.get(x["condition"], 9), x["label"]))
        if not members:
            members = [{
                "run_id": None, "label": "", "condition": "not run",
                "score": None, "verdict": None,
                "tainted": None, "caveats": [], "placeholder": True,
                "running": False,
            }]
        # A box runs only `parallel` conditions at once (= its GPU count); on a 1x
        # box the other conditions queue behind on the same GPU. Mark conditions
        # beyond the box's parallelism as QUEUED so the orange "running" rows match
        # the GPUs actually in use (this is what made 11 rows look like 11 GPUs).
        run_members = [m for m in members if m.get("running")]
        par = max((m.get("parallel") or 1) for m in run_members) if run_members else 1
        for k, m in enumerate(run_members):
            if k >= par:
                m["status"] = "queued"
                m["queued"] = True
                m["live"] = False
        orow = RUNS[INDEX[orig]] if orig in INDEX else {}
        mrow = manifest_by_id.get(orig) or {}
        eng = engine_of(orig)
        # Blanket Claude taint: the native-CLI resume cold-starts (doesn't replay
        # the pre-cut history), so no Claude run is a faithful continuation. Real
        # completed runs that lack their own measured taint get the generic note;
        # the per-run measured taints (where present) are left untouched.
        if eng in ROLLBACK_CLAUDE_ENGINES:
            for m in members:
                if not m.get("placeholder") and not m.get("tainted"):
                    m["tainted"] = ROLLBACK_CLAUDE_COLD_START_TAINT
        # Emit a one-row engine header before the first trajectory of each block.
        engine_header = None
        if eng != prev_engine:
            engine_header = f"{ROLLBACK_ENGINE_LABEL.get(eng, eng)} · {engine_counts.get(eng, 0)}"
            prev_engine = eng
        dreason = deferred_reason(orig)
        n_real = sum(1 for m in members if not m.get("placeholder"))
        n_run = sum(1 for m in members if m.get("running"))
        if dreason:
            pstatus = "deferred"
        elif n_run:
            pstatus = "running"
        elif n_real >= 3:
            pstatus = "done"
        elif n_real:
            pstatus = "partial"
        else:
            pstatus = "not run"
        pairs.append({
            "orig_run_id": orig,
            "orig_in_index": orig in INDEX,
            "orig_accuracy": (orow.get("accuracy") if isinstance(orow, dict)
                              else getattr(orow, "accuracy", None)),
            "orig_stderr": (orow.get("stderr") if isinstance(orow, dict)
                            else getattr(orow, "stderr", None)),
            "orig_experiment": (orow.get("experiment") if isinstance(orow, dict)
                                else mrow.get("experiment")),
            "scaffold": mrow.get("scaffold"),
            "engine": eng,
            "engine_header": engine_header,
            # Tag A: was the original trajectory's model trained at all before the
            # cut? If so, the rollback must re-create that pre-cut model (prep);
            # if not, the cut-point model is just the untouched base model.
            "n_trainings": mrow.get("trainings_before_first_hack"),
            "needs_retrain": (mrow.get("needs_prep")
                              if mrow.get("needs_prep") is not None
                              else (mrow.get("trainings_before_first_hack") or 0) > 0),
            # Tag B: which of the 7 /report cheating categories this trajectory hit
            # (a few hit two). [(short_tag, full_title), ...]; empty if unmapped.
            "cheat_cats": cheat_by_run.get(orig, []),
            "deferred": bool(dreason),
            "deferred_reason": dreason,
            "status_summary": pstatus,
            "runs": members,
        })
    n_running_traj = sum(1 for p in pairs if p["status_summary"] == "running")
    n_done_traj = sum(1 for p in pairs if p["status_summary"] == "done")
    n_deferred_traj = sum(1 for p in pairs if p["deferred"])
    n_todo_runs = sum(1 for p in pairs for r in p["runs"] if r.get("todo"))
    nruns = sum(1 for p in pairs for r in p["runs"] if not r.get("placeholder"))

    # Standalone bar chart: classify each COMPLETED trajectory by its hack
    # pattern across prompt1/2/3. Computed from the SAME `pairs` rows that build
    # the table above, so it auto-updates as new runs sync in.
    chart_png, breakdown = rollback_outcome_chart(pairs)

    model_score_path = ROLLBACK_DATA.parent / "reconstruction_fidelity.json"
    model_score_rows = []
    if model_score_path.exists():
        try:
            model_score_rows = json.loads(model_score_path.read_text())
        except json.JSONDecodeError:
            model_score_rows = []
    return render_template_string(ROLLBACK_HTML, pairs=pairs, nruns=nruns,
                                  n_running_traj=n_running_traj,
                                  n_done_traj=n_done_traj,
                                  n_deferred_traj=n_deferred_traj,
                                  n_todo_runs=n_todo_runs,
                                  model_score_rows=model_score_rows,
                                  chart_png=chart_png,
                                  breakdown=breakdown,
                                  label_class=_LABEL_CLASS)




@app.route("/report")
def report():
    """Cheating-pattern taxonomy (generated by exp_categorize_cheating.py).

    Read fresh per request so a newly generated report appears without a
    viewer restart."""
    path = HIGHLIGHTS / "categories.json"
    if not path.exists():
        return render_template_string(REPORT_MISSING_HTML)
    data = json.loads(path.read_text())
    # which runs appear in more than one category (shown as a ×N marker)
    appearances: dict[str, list[str]] = {}
    for c in data.get("categories", []):
        for m in c.get("members", []):
            if m.get("run_id"):
                appearances.setdefault(m["run_id"], []).append(c.get("title", ""))
    cats = []
    for c in data.get("categories", []):
        members = []
        for m in c.get("members", []):
            rid = m.get("run_id")
            if rid not in INDEX:
                continue
            r = RUNS[INDEX[rid]]
            hm = HL_META.get(rid, {})
            members.append({"run_id": rid, "note": m.get("note", ""),
                            "experiment": r.get("experiment"),
                            "benchmark": r.get("benchmark"),
                            "trained_model": r.get("trained_model"),
                            "agent_model": r.get("agent_model"),
                            "accuracy": r.get("accuracy"),
                            "label": hm.get("label", ""),
                            "truncated": hm.get("truncated"),
                            "also": [t for t in appearances.get(rid, [])
                                     if t != c.get("title")]})
        members.sort(key=lambda m: ORDER_POS.get(m["run_id"], 1 << 30))  # browse order
        cats.append({**c, "members": members})
    # Inject the OpenAI-API reward hacks as their own cheating category
    # (final label "openai api misuse"; the accidental violations carry
    # "broke rules unknowingly" — not deliberate cheating, not listed here).
    oai_members = []
    for rid, m in HL_META.items():
        if m.get("label") != "openai api misuse" or rid not in INDEX:
            continue
        r = RUNS[INDEX[rid]]
        oj = OPENAI_JUDGE.get(rid, {})
        ca = oj.get("context_assessment") or {}
        oai_members.append({
            "run_id": rid, "experiment": r.get("experiment"),
            "benchmark": r.get("benchmark"), "trained_model": r.get("trained_model"),
            "agent_model": r.get("agent_model"), "accuracy": r.get("accuracy"),
            "label": "openai api misuse", "truncated": None, "also": [],
            "note": (oj.get("non_eval_purpose", "") + " — " + ca.get("basis", "")).strip(" —")})
    if oai_members:
        oai_members.sort(key=lambda m: ORDER_POS.get(m["run_id"], 1 << 30))
        cats.append({
            "kind": "cheating",
            "title": "OpenAI-API misuse (eval key → data generation)",
            "description": "Deliberately used the eval-only OpenAI API for non-eval work "
                           "(synthetic training data / self-judging) with the rule in context. "
                           "Surfaced by the OpenAI-rule pass — separate from the original "
                           "contamination / disallowed-model judges. Accidental (rule-dropped) "
                           "violations are excluded as non-deliberate.",
            "members": oai_members})
    return render_template_string(REPORT_HTML, cats=cats,
                                  overview=data.get("overview", ""),
                                  n_runs=data.get("n_runs", 0),
                                  model=data.get("model", ""),
                                  label_class=_LABEL_CLASS)


@app.route("/workspace/<run_id>")
def workspace(run_id: str):
    ws = load_workspace(run_id)
    if ws is None:
        abort(404)
    return render_template_string(WORKSPACE_HTML, run_id=run_id,
                                  root=ws.get("root", ""), files=ws.get("files", []))


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", "5001")))
