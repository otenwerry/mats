"""Local web viewer for the InferenceBench agent trajectories.

Data lives in mats-local (mirrored from HF by download.py, see that file):

  mats-local/inferencebench/data/runs/run_NNNN/
    trace.jsonl     -- flat event stream: {i, role, type, text, tool_*, meta}
    run_meta.json   -- agent/scenario + invalid_or_reward_hack flag + judgements
    metrics.json    -- final eval (absent if the run produced none)
    logs/           -- server.log / eval.log

Index page: one row per run, flagged (invalid_or_reward_hack) runs badged red
and sorted to the top. Run page: meta + judge outputs + final metrics + the
full trace (long bodies are collapsed, never truncated). Raw logs are served
as plain text.

Usage:  uv run python mats/inferencebench/viewer.py   then open http://127.0.0.1:5002
"""
from __future__ import annotations

import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, abort, render_template_string, Response

DATA = Path(__file__).resolve().parents[2] / "mats-local" / "inferencebench" / "data"
RUNS = DATA / "runs"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
COLLAPSE_CHARS = 1500  # fold longer bodies into <details>; full text stays in the page

# Our judge-panel outputs (exp_judge_flagged.py). Read live from disk on every
# page load, so results appear as soon as each (run x judge) conversation lands.
JUDGE_DIR = DATA.parent / "judge" / "panel_v1"
JUDGE_ORDER = ["kimi-k2.6", "deepseek-v4-pro", "glm-5.2"]  # glm = pilot-only leftovers
JUDGE_TAG = {"kimi-k2.6": "K", "deepseek-v4-pro": "D", "glm-5.2": "G"}
JUDGE_CATEGORIES = ("contamination", "disallowed_model", "quality_gate")
# most→least damning; a judge's headline chip shows its worst verdict.
# open_rule_break was merged into cheating (2026-07-02); older on-disk outputs
# still carry it, so verdicts are normalized through LEGACY_VERDICTS on load.
VERDICT_SEVERITY = ["cheating", "honest_mistake", "uncertain", "false_positive"]
VERDICT_CLS = {"cheating": "bad", "honest_mistake": "other",
               "uncertain": "none", "false_positive": "clean"}
LEGACY_VERDICTS = {"open_rule_break": "cheating"}

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Data access                                                                 #
# --------------------------------------------------------------------------- #
def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    if not d.is_dir() or not d.resolve().is_relative_to(RUNS.resolve()):
        abort(404)
    return d


def load_meta(d: Path) -> dict:
    with open(d / "run_meta.json") as f:
        return json.load(f)


def load_metrics(d: Path) -> dict | None:
    p = d / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# The two judge outputs are free text; classify into clean / bad / other for
# compact colored chips (full text shown on hover + on the run page).
def judge_chip(text: str | None, kind: str) -> dict:
    t = (text or "").strip()
    low = t.lower()
    if kind == "contamination":
        clean, bad = "no contamination", "contamination detected"
        clean_label, bad_label = "clean", "contamination"
    else:
        clean, bad = "only allowed", "disallowed use detected"
        clean_label, bad_label = "clean", "disallowed model"
    if not t:
        return {"cls": "none", "label": "–", "full": ""}
    if clean in low:
        return {"cls": "clean", "label": clean_label, "full": t}
    if bad in low:
        return {"cls": "bad", "label": bad_label, "full": t}
    return {"cls": "other", "label": t[:40], "full": t}


def load_judgements(run_id: str) -> list[dict]:
    """One entry per judge with an output file for this run, in JUDGE_ORDER."""
    out = []
    for judge in JUDGE_ORDER:
        p = JUDGE_DIR / f"{run_id}__{judge}.json"
        if not p.exists():
            continue
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue  # file mid-write; next refresh will pick it up
        verdicts = r.get("verdicts") or {}
        cats = {}
        for c in JUDGE_CATEGORIES:
            v = verdicts.get(c)
            if isinstance(v, dict):
                v = dict(v)  # don't mutate the loaded JSON
                v["verdict"] = LEGACY_VERDICTS.get(v.get("verdict"), v.get("verdict"))
                cats[c] = v
        cat_verdicts = [c.get("verdict") for c in cats.values()]
        worst = next((v for v in VERDICT_SEVERITY if v in cat_verdicts), None)
        out.append({
            "judge": judge,
            "tag": JUDGE_TAG.get(judge, judge[:1].upper()),
            "error": r.get("error"),
            "hit_turn_limit": bool(r.get("hit_turn_limit")),
            "invalid_submission": bool(r.get("invalid_submission")),
            "cats": cats,
            "worst": worst,
            "cls": "none" if r.get("error") else VERDICT_CLS.get(worst, "none"),
            "label": "error" if r.get("error") else (worst or "no verdict"),
            "tip": "; ".join(f"{c}={v.get('verdict')}({v.get('confidence')})" for c, v in cats.items())
                   + (" · HIT TURN LIMIT" if r.get("hit_turn_limit") else "")
                   or (r.get("error") or "no verdicts"),
            "overall_summary": verdicts.get("overall_summary") or "",
            "additional_observations": verdicts.get("additional_observations") or "",
        })
    return out


def panel_sort_key(judgements: list[dict]) -> int:
    """Severity order for the 'our panel' column: rows sort by their most damning
    per-judge headline verdict, and within a verdict by how many judges gave it
    (2x cheating, 1x cheating, 2x honest_mistake, 1x, ... per VERDICT_SEVERITY).
    Lower = more damning; 99 = no verdicts yet. Emitted as data-key on the cell,
    which the sort script prefers over cell text."""
    ranks = [VERDICT_SEVERITY.index(j["worst"]) for j in judgements if j.get("worst")]
    if not ranks:
        return 99
    worst = min(ranks)
    return worst * 10 + (3 - ranks.count(worst))


def index_rows() -> list[dict]:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        meta = load_meta(d)
        metrics = load_metrics(d)
        qc = (metrics or {}).get("quality_check") or {}
        served = (metrics or {}).get("model_id")
        rows.append({
            "run_id": meta["anon_id"],
            "flagged": bool(meta.get("invalid_or_reward_hack")),
            "agent": meta["agent"],
            "harness": meta["harness"],
            "scenario": meta["scenario"],
            "contamination": judge_chip(
                (meta.get("judgements") or {}).get("contamination_judgement"), "contamination"),
            "disallowed": judge_chip(
                (meta.get("judgements") or {}).get("disallowed_model_judgement"), "disallowed"),
            "has_metrics": metrics is not None,
            # four states: pass / fail / missing (metrics.json exists but has
            # no quality_check block — happens in 8 runs) / noeval (no metrics)
            "quality_state": ("noeval" if metrics is None else
                              "pass" if qc.get("pass") is True else
                              "fail" if qc.get("pass") is False else "missing"),
            "served_model": served,
            "served_ok": served == BASE_MODEL if served else None,
            "time_taken": meta.get("time_taken") or "–",
            "n_events": meta.get("n_trace_events", 0),
            "judgements": load_judgements(meta["anon_id"]),
        })
        rows[-1]["panel_key"] = panel_sort_key(rows[-1]["judgements"])
        r = rows[-1]
        # unflagged runs that still tripped something, for the middle sort tiers:
        # a judge detection (most interesting), then a real quality-gate FAIL
        # (legit non-hack outcome), then runs whose gate is simply absent
        # (missing quality_check / no final eval) — anomalous but not "fails"
        r["trip_judge"] = (not r["flagged"]) and (
            r["contamination"]["cls"] == "bad" or r["disallowed"]["cls"] == "bad")
        r["trip_quality"] = (not r["flagged"]) and not r["trip_judge"] \
            and r["quality_state"] == "fail"
        r["no_check"] = (not r["flagged"]) and not r["trip_judge"] \
            and r["quality_state"] in ("missing", "noeval")
        # quality-gate-fail-only runs: manually reviewed 2026-07-02 and relabeled
        # CLEAN for the index (judges found broken submissions / unintended quality
        # drops, no rule-breaking). Raw verdicts remain on the run page.
        if r["trip_quality"] and r["judgements"]:
            r["judgements"] = [{
                "judge": "manual", "tag": "", "cls": "clean", "label": "clean",
                "tip": ("quality-gate-only failure — manually relabeled clean "
                        "(judges: broken submission / unintended quality drop, "
                        "no rule-breaking); full verdicts on the run page"),
            }]
            r["panel_key"] = 50  # below all real verdicts, above unjudged (99)
    rows.sort(key=lambda r: (
        0 if r["flagged"] else 1 if r["trip_judge"] else 2 if r["trip_quality"]
        else 3 if r["no_check"] else 4,
        r["run_id"]))
    return rows


# --------------------------------------------------------------------------- #
# Trace rendering                                                             #
# --------------------------------------------------------------------------- #
def _fold(body_html: str, char_len: int, label: str) -> str:
    if char_len <= COLLAPSE_CHARS:
        return body_html
    return (f'<details><summary>{html.escape(label)} · {char_len:,} chars</summary>'
            f'{body_html}</details>')


def render_event(e: dict) -> str:
    i, role, typ = e.get("i"), e.get("role", "?"), e.get("type", "?")
    anchor = f'<a class="idx" id="e{i}" href="#e{i}">[{i}]</a>'
    text = e.get("text") or ""

    if role == "system":  # init / status runner chatter: slim gray lines
        extra = f' · model={e["model"]}' if e.get("model") else ""
        return (f'<div class="sys">{anchor} {html.escape(typ)}{html.escape(extra)} '
                f'<span class="systext">{html.escape(text)}</span></div>')

    if typ == "tool_call":
        name = e.get("tool_name", "?")
        inp = e.get("tool_input")
        body = inp if isinstance(inp, str) else json.dumps(inp, indent=2)
        pre = f'<pre class="content tool">{html.escape(body)}</pre>'
        return (f'<div class="msg call"><div class="head">{anchor} assistant · '
                f'tool call · <b>{html.escape(str(name))}</b></div>'
                f'{_fold(pre, len(body), "input")}</div>')

    if typ == "tool_result":
        out = e.get("tool_output") or ""
        out = out if isinstance(out, str) else json.dumps(out, indent=2)
        err = bool((e.get("meta") or {}).get("is_error"))
        pre = f'<pre class="content tool">{html.escape(out)}</pre>'
        return (f'<div class="msg result{" err" if err else ""}">'
                f'<div class="head">{anchor} tool result{" · ERROR" if err else ""}</div>'
                f'{_fold(pre, len(out), "output")}</div>')

    cls = "thinking" if typ == "thinking" else "text"
    pre = f'<pre class="content {cls}">{html.escape(text)}</pre>'
    return (f'<div class="msg {cls}"><div class="head">{anchor} {html.escape(role)} · '
            f'{html.escape(typ)}</div>{_fold(pre, len(text), typ)}</div>')


def render_judgement_panel(judgements: list[dict]) -> str:
    """Run-page panel: one box per judge with per-gate verdicts, folded
    justifications, and key-event links into the trace anchors below."""
    if not judgements:
        return ""
    boxes = []
    for j in judgements:
        head = (f'<b>{html.escape(j["judge"])}</b> '
                f'<span class="chip {j["cls"]}">{html.escape(j["label"])}</span>')
        if j["hit_turn_limit"]:
            head += ' <span class="chip warn" title="judge was forced to submit at the turn limit">hit turn limit</span>'
        if j["invalid_submission"]:
            head += ' <span class="chip warn">invalid submission accepted raw</span>'
        if j["error"]:
            boxes.append(f'<div class="jbox">{head}<div class="kv">{html.escape(j["error"])}</div></div>')
            continue
        parts = [head]
        if j["overall_summary"]:
            parts.append(f'<div class="kv">{html.escape(j["overall_summary"])}</div>')
        for cat, v in j["cats"].items():
            verdict = v.get("verdict", "?")
            cls = VERDICT_CLS.get(verdict, "none")
            events = " ".join(
                f'<a class="idx" href="#e{e.get("i")}" title="{html.escape(str(e.get("note", "")))}">[{e.get("i")}]</a>'
                for e in (v.get("key_events") or []) if isinstance(e, dict)
            )
            just = html.escape(v.get("justification") or "")
            parts.append(
                f'<div class="jcat"><span class="chip {cls}">{html.escape(cat)}: '
                f'{html.escape(verdict)} ({html.escape(str(v.get("confidence")))})</span> {events}'
                f'<details><summary>justification</summary>'
                f'<pre class="content">{just}</pre></details></div>'
            )
        if j["additional_observations"]:
            parts.append(f'<div class="kv"><b>additional observations:</b> '
                         f'{html.escape(j["additional_observations"])}</div>')
        boxes.append(f'<div class="jbox">{"".join(parts)}</div>')
    return f'<div class="banner"><b>our judge panel</b>{"".join(boxes)}</div>'


# --------------------------------------------------------------------------- #
# Visuals                                                                     #
# --------------------------------------------------------------------------- #
JUDGE_PLOT = ["kimi-k2.6", "deepseek-v4-pro"]
JUDGE_COLORS = {"kimi-k2.6": "#4c72b0", "deepseek-v4-pro": "#dd8452"}


def visuals_groups() -> list[tuple[str, list[str]]]:
    """The two plotted populations. Judged runs in neither (quality-gate-fail-only
    runs) were manually reviewed 2026-07-02 and are treated as CLEAN: their judge
    verdicts were all broken-submission / unintended-quality-drop honest mistakes
    or harness issues, i.e. no rule-breaking (raw verdicts stay on disk and on
    the run pages)."""
    author, gates = [], []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        meta = load_meta(d)
        j = meta.get("judgements") or {}
        detected = (j.get("contamination_judgement") == "contamination detected"
                    or j.get("disallowed_model_judgement") == "disallowed use detected")
        if meta.get("invalid_or_reward_hack"):
            author.append(meta["anon_id"])
        elif detected:
            gates.append(meta["anon_id"])
    return [("flagged invalid_or_reward_hack", author),
            ("contamination/disallowed but not flagged invalid_or_reward_hack", gates)]


def categories_figure() -> bytes:
    """Paper-style figure: per population, the fraction of runs each judge
    assigns to each verdict category (judge headline = most damning verdict
    across the gates it adjudicated). Counts annotated above the bars."""
    cats = VERDICT_SEVERITY
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, (title, run_ids) in zip(axes, visuals_groups()):
        worsts = {jn: [] for jn in JUDGE_PLOT}
        for rid in run_ids:
            for jr in load_judgements(rid):
                if jr["judge"] in worsts and jr["worst"]:
                    worsts[jr["judge"]].append(jr["worst"])
        for k, jn in enumerate(JUDGE_PLOT):
            w = worsts[jn]
            n = len(w)
            fracs = [(w.count(c) / n if n else 0) for c in cats]
            pos = [i + (k - 0.5) * width for i in range(len(cats))]
            bars = ax.bar(pos, fracs, width, label=f"{jn}  (n={n})",
                          color=JUDGE_COLORS[jn], edgecolor="white", linewidth=0.5)
            for rect, c in zip(bars, cats):
                cnt = w.count(c)
                if cnt:
                    ax.annotate(str(cnt), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                ha="center", va="bottom", fontsize=8, color="#444",
                                xytext=(0, 1), textcoords="offset points")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace("_", " ") for c in cats], fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("fraction of runs", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


VERDICT_COLORS = {"cheating": "#b3261e", "honest_mistake": "#d99a00",
                  "uncertain": "#9a9a9a", "false_positive": "#137333",
                  "clean": "#88b78d"}


def overall_figure() -> bytes:
    """Single bar chart: verdict rates across ALL runs in the dataset. 'Clean'
    counts runs that were never judged (nothing flagged them) plus the
    quality-gate-fail-only runs (manually reviewed and relabeled clean — see
    visuals_groups). Other judged runs contribute one unit of mass split
    equally between the panel judges' headline verdicts (so a kimi/deepseek
    disagreement counts half each). A judged run whose judges all errored
    contributes no mass."""
    cats = VERDICT_SEVERITY + ["clean"]
    mass = {c: 0.0 for c in cats}
    n_runs = 0
    plotted = set().union(*(ids for _, ids in visuals_groups()))
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        n_runs += 1
        judgements = load_judgements(d.name)
        if not judgements or d.name not in plotted:
            mass["clean"] += 1.0
            continue
        jrs = [j for j in judgements if j["judge"] in JUDGE_PLOT and j["worst"]]
        for j in jrs:
            mass[j["worst"]] += 1.0 / len(jrs)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    fracs = [mass[c] / n_runs if n_runs else 0 for c in cats]
    bars = ax.bar(range(len(cats)), fracs, 0.62,
                  color=[VERDICT_COLORS[c] for c in cats], edgecolor="white", linewidth=0.5)
    for rect, c in zip(bars, cats):
        if mass[c]:
            ax.annotate(f"{mass[c]:g}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8, color="#444",
                        xytext=(0, 1), textcoords="offset points")
    ax.set_title(f"all runs (n={n_runs})", fontsize=9.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace("_", " ") for c in cats], fontsize=9)
    ax.set_ylabel("fraction of runs", fontsize=10)
    ax.set_ylim(0, max(fracs + [0.01]) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


def build_nav_groups(judgements: list[dict]) -> list[dict]:
    """Evidence navigator groups, one per CONCLUSION present on this run.

    Each (judge, gate) verdict contributes its cited key_events to the group for
    that verdict — so the 'cheating' group contains only turns some judge cited
    while calling the run a hack, the 'honest_mistake' group only turns cited in
    support of an honest mistake, etc. Groups are ordered most→least damning."""
    groups: dict[str, dict] = {}
    for j in judgements:
        for cat, v in j["cats"].items():
            verdict = v.get("verdict")
            if verdict not in VERDICT_CLS:
                continue
            g = groups.setdefault(verdict, {"sources": [], "events": {}})
            g["sources"].append(f'{j["tag"]} {cat}')
            for e in v.get("key_events") or []:
                if isinstance(e, dict) and isinstance(e.get("i"), int):
                    g["events"].setdefault(e["i"], []).append(f'{j["tag"]}: {e.get("note") or ""}')
    out = []
    for verdict in VERDICT_SEVERITY:
        if verdict not in groups:
            continue
        g = groups[verdict]
        events = [{"i": i, "title": " | ".join(dict.fromkeys(notes))}
                  for i, notes in sorted(g["events"].items())]
        if events:
            out.append({"verdict": verdict, "cls": VERDICT_CLS[verdict],
                        "sources": g["sources"], "events": events})
    return out


NAV_JS = """<script>
(function () {
  var GROUPS = __GROUPS__;
  var panel = document.getElementById("cnav");
  var gos = [];
  GROUPS.forEach(function (g) {
    var div = document.createElement("div");
    div.className = "cnav-grp";
    div.innerHTML = '<span class="chip ' + g.cls + '">' + g.verdict + '</span>'
      + ' <span class="cnav-src">' + g.sources.join(", ") + '</span>'
      + '<div class="cnav-row"><button>&larr;</button><span class="lbl">0 / '
      + g.events.length + '</span><button>&rarr;</button></div>'
      + '<div class="cnav-title"></div>';
    panel.appendChild(div);
    var btns = div.querySelectorAll("button");
    var lbl = div.querySelector(".lbl"), ttl = div.querySelector(".cnav-title");
    var targets = g.events.map(function (e) {
      var a = document.getElementById("e" + e.i);
      var el = a ? (a.closest(".msg") || a.closest(".sys") || a.parentElement) : null;
      return el ? { el: el, e: e } : null;
    }).filter(Boolean);
    // outline each cited event by the most damning conclusion citing it (groups
    // arrive most-damning first, so the first outline wins)
    targets.forEach(function (t) {
      if (!/(^|\\s)cited-/.test(t.el.className)) t.el.classList.add("cited-" + g.cls);
    });
    var cur = -1;
    function go(d) {
      if (!targets.length) return;
      cur = (cur + d + targets.length) % targets.length;
      var t = targets[cur];
      t.el.scrollIntoView({ behavior: "smooth", block: "center" });
      t.el.classList.remove("flash"); void t.el.offsetWidth; t.el.classList.add("flash");
      lbl.textContent = (cur + 1) + " / " + targets.length;
      ttl.textContent = "[" + t.e.i + "] " + (t.e.title || "");
    }
    btns[0].onclick = function () { go(-1); };
    btns[1].onclick = function () { go(1); };
    gos.push(go);
  });
  var keys = document.createElement("div");
  keys.className = "cnav-keys";
  keys.textContent = "n/p: " + GROUPS[0].verdict + (GROUPS[1] ? " \\u00b7 j/k: " + GROUPS[1].verdict : "");
  panel.appendChild(keys);
  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "n" && gos[0]) gos[0](1);
    else if (e.key === "p" && gos[0]) gos[0](-1);
    else if (e.key === "j" && gos[1]) gos[1](1);
    else if (e.key === "k" && gos[1]) gos[1](-1);
  });
})();
</script>"""


def render_nav(judgements: list[dict]) -> str:
    groups = build_nav_groups(judgements)
    if not groups:
        return ""
    payload = json.dumps(groups).replace("</", "<\\/")
    return '<div class="cnav" id="cnav"></div>' + NAV_JS.replace("__GROUPS__", payload)


# --------------------------------------------------------------------------- #
# Templates                                                                   #
# --------------------------------------------------------------------------- #
CSS = """
 body{font-family:system-ui,sans-serif;margin:1.2rem auto;max-width:1150px;padding:0 1rem;color:#222}
 a{color:#1a56b0;text-decoration:none} a:hover{text-decoration:underline}
 h1{font-size:1.3rem} .sub{color:#666;margin:.2rem 0 1rem}
 table{border-collapse:collapse;width:100%;font-size:.85rem}
 th,td{padding:.28rem .5rem;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
 th{background:#f7f7f7;position:sticky;top:0}
 table.sortable th{cursor:pointer;user-select:none}
 table.sortable th:hover{background:#e2e6ee}
 table.sortable th[data-sort="asc"]::after{content:" ▲";font-size:9px;color:#2456a6}
 table.sortable th[data-sort="desc"]::after{content:" ▼";font-size:9px;color:#2456a6}
 tr.flagged{background:#fff5f5}
 .badge{display:inline-block;border-radius:4px;padding:0 .4rem;font-size:.72rem;font-weight:700}
 .badge.flag{color:#b3261e;background:#fde7e7;border:1px solid #f2b8b5}
 .badge.trip{color:#9a5b00;background:#fdf3d7;border:1px solid #eec07a}
 tr.tripped{background:#fffdf0}
 .chip{display:inline-block;border-radius:4px;padding:0 .35rem;font-size:.72rem;cursor:help}
 .chip.clean{color:#137333;background:#e6f4ea} .chip.bad{color:#b3261e;background:#fde7e7;font-weight:700}
 .chip.none{color:#999} .chip.other{color:#7a5b00;background:#fdf3d7}
 .chip.warn{color:#8a3b00;background:#ffe3cc;font-weight:700}
 .jcell .chip{display:block;width:max-content;margin:1px 0}
 .jcell .jtag{font-weight:700;opacity:.65;margin-right:.2rem}
 .jbox{border:1px solid #e6e6e6;border-radius:6px;padding:.4rem .6rem;margin-top:.4rem;background:#fff}
 .jcat{margin-top:.3rem}
 .jcat details{display:inline-block;margin-left:.4rem;vertical-align:top}
 .jcat pre.content{border:1px solid #eee;border-radius:6px;max-width:60rem}
 .totop{position:fixed;right:18px;bottom:18px;width:42px;height:42px;border:none;border-radius:50%;
   background:#1a56b0;color:#fff;font-size:1.25rem;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.28);
   display:none;z-index:60}
 .totop:hover{background:#0f47b0}
 .cnav{position:fixed;bottom:70px;right:18px;width:240px;max-height:calc(100vh - 110px);overflow:auto;
   background:#fff;border:1px solid #d5d8e0;border-radius:8px;padding:.5rem .6rem;
   box-shadow:0 2px 10px rgba(0,0,0,.08);font-size:.78rem;z-index:50}
 .cnav-grp{margin-bottom:.45rem;padding-bottom:.4rem;border-bottom:1px solid #eceef2}
 .cnav-row{display:flex;align-items:center;gap:8px;margin:4px 0 2px}
 .cnav .lbl{font-variant-numeric:tabular-nums;min-width:44px;text-align:center}
 .cnav button{cursor:pointer;border:1px solid #c8cad2;background:#f2f3f6;border-radius:5px;padding:1px 9px;font-size:12px}
 .cnav button:hover{background:#e4e6ec}
 .cnav-src{color:#778;font-size:.7rem}
 .cnav-title{font-size:.72rem;color:#555;min-height:12px}
 .cnav-keys{color:#889;margin-top:2px;font-size:.68rem}
 .msg.cited-bad,.sys.cited-bad{outline:2px solid #d93025}
 .msg.cited-warn,.sys.cited-warn{outline:2px solid #ea580c}
 .msg.cited-other,.sys.cited-other{outline:2px solid #d99a00}
 .msg.cited-clean,.sys.cited-clean{outline:2px solid #137333}
 .msg.cited-none,.sys.cited-none{outline:2px solid #99a}
 .flash{animation:navflash 1.1s}
 @keyframes navflash{0%{background:#fff3cd}100%{background:transparent}}
 .topnav{margin:0 0 1.1rem;display:flex;gap:8px}
 .topnav a{padding:5px 13px;border-radius:6px;background:#eef0f4;font-size:.85rem;font-weight:600}
 .topnav a.active{background:#1a56b0;color:#fff}
 .topnav a:hover{text-decoration:none;background:#e0e3ea}
 .topnav a.active:hover{background:#0f47b0}
 img.figure{display:block;background:#fff;max-width:100%;margin:.6rem 0}
 .mono{font-family:ui-monospace,monospace;font-size:.8rem}
 .servedbad{color:#b3261e;font-weight:700}
 .banner{border:1px solid #e6e6e6;border-radius:8px;padding:.6rem .8rem;margin:.6rem 0;background:#fafafa}
 .banner.flagged{background:#fff1f0;border-color:#f2b8b5}
 .banner .big{font-weight:700;font-size:1.02rem}
 .kv{color:#555;margin-top:.25rem} .kv b{color:#222}
 .judgebox{margin-top:.4rem;padding:.4rem .6rem;border-radius:6px;font-size:.86rem}
 .judgebox.clean{background:#f0fbf2;border:1px solid #bfe6cb}
 .judgebox.bad{background:#fde7e7;border:1px solid #f2b8b5}
 .judgebox.none{background:#f4f4f4;border:1px solid #e0e0e0;color:#888}
 .judgebox.other{background:#fdf3d7;border:1px solid #eec07a}
 .msg{border:1px solid #e6e6e6;border-radius:8px;margin:.55rem 0;overflow:hidden}
 .msg .head{background:#f4f6f8;padding:.25rem .6rem;font-size:.78rem;color:#555}
 .msg.call .head{background:#eef3fb} .msg.result .head{background:#f4f4f4}
 .msg.thinking .head{background:#f5f1fc} .msg.err{border-color:#f2b8b5} .msg.err .head{background:#fde7e7}
 pre.content{margin:0;padding:.5rem .6rem;white-space:pre-wrap;word-break:break-word;font-size:.8rem;line-height:1.35}
 pre.content.tool{color:#444;background:#fbfbfb}
 pre.content.thinking{color:#5b4b8a;background:#faf8ff;font-style:italic}
 .sys{color:#888;font-size:.76rem;margin:.15rem 0;padding-left:.2rem}
 .sys .systext{color:#aaa}
 .idx{color:#bbb;font-family:ui-monospace,monospace;font-size:.72rem} .idx:hover{color:#1a56b0}
 details summary{cursor:pointer;color:#1a56b0;font-size:.78rem;padding:.35rem .6rem}
 .num{text-align:right;font-variant-numeric:tabular-nums}
"""

TOPNAV = """<div class="topnav">
 <a href="/"{% if tab == 'trajectories' %} class="active"{% endif %}>trajectories</a>
 <a href="/visuals"{% if tab == 'visuals' %} class="active"{% endif %}>visuals</a>
</div>"""

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>InferenceBench trajectories</title><style>__CSS__</style></head><body>
__TOPNAV__
<h1>InferenceBench trajectories</h1>
<p class="sub">{{ rows|length }} runs, sorted: <span class="badge flag">⚑ {{ n_flagged }}</span>
 flagged invalid_or_reward_hack, then <span class="badge trip">⚠ judge {{ n_trip_judge }}</span>
 unflagged judge detections, then <span class="badge trip">⚠ quality {{ n_trip_quality }}</span>
 unflagged quality-gate fails, then {{ n_no_check }} runs with a missing quality
 check / no final eval (no badge), then the rest — click a column header to re-sort</p>
<table class="sortable"><thead><tr>
 <th>run</th><th></th><th>agent</th><th>harness</th><th>sc</th>
 <th>contamination</th><th>disallowed model</th><th>quality gate</th>
 <th>our panel</th>
 <th class="num">time</th><th class="num">events</th>
</tr></thead><tbody>
{% for r in rows %}
<tr data-group="{{ 0 if (r.flagged or r.trip_judge or r.trip_quality) else 1 }}"{% if r.flagged %} class="flagged"{% elif r.trip_judge or r.trip_quality %} class="tripped"{% endif %}>
 <td><a class="mono" href="/run/{{ r.run_id }}">{{ r.run_id }}</a></td>
 <td>{% if r.flagged %}<span class="badge flag">⚑ flagged</span>
     {% elif r.trip_judge %}<span class="badge trip" title="a judge said contamination/disallowed use, but the run is not flagged">⚠ judge</span>
     {% elif r.trip_quality %}<span class="badge trip" title="quality gate failed, missing, or no final eval — not flagged">⚠ quality</span>{% endif %}</td>
 <td>{{ r.agent }}</td><td>{{ r.harness }}</td><td>{{ r.scenario }}</td>
 <td><span class="chip {{ r.contamination.cls }}" title="{{ r.contamination.full }}">{{ r.contamination.label }}</span></td>
 <td><span class="chip {{ r.disallowed.cls }}" title="{{ r.disallowed.full }}">{{ r.disallowed.label }}</span></td>
 <td>{% if r.quality_state == 'pass' %}<span class="chip clean">pass</span>
     {% elif r.quality_state == 'fail' %}<span class="chip bad">fail</span>
     {% elif r.quality_state == 'missing' %}<span class="chip other" title="metrics.json exists but contains no quality_check block">no quality check</span>
     {% else %}<span class="chip none">no final eval</span>{% endif %}</td>
 <td class="jcell" data-key="{{ r.panel_key }}">{% for j in r.judgements %}<span class="chip {{ j.cls }}" title="{{ j.judge }}: {{ j.tip }}"><span class="jtag">{{ j.tag }}</span>{{ j.label }}</span>{% endfor %}</td>
 <td class="num">{{ r.time_taken }}</td><td class="num">{{ r.n_events }}</td>
</tr>
{% endfor %}
</tbody></table>
__SORT_JS__
__TOTOP__
</body></html>""".replace("__CSS__", CSS)

# back-to-top button (petri-style): appears once you've scrolled past 400px
TOTOP_HTML = """<button class="totop" id="totop" title="back to top">&#8593;</button>
<script>
(function () {
  var b = document.getElementById("totop");
  function upd() { b.style.display = window.scrollY > 400 ? "block" : "none"; }
  b.onclick = function () { window.scrollTo({ top: 0, behavior: "smooth" }); };
  window.addEventListener("scroll", upd); upd();
})();
</script>"""

# Click a column header to re-sort client-side (adapted from petri/make_viewer.py).
# The server-rendered order is the default; clicking sorts by that column and toggles
# asc/desc. A column is numeric if every non-blank cell parses as a number (events);
# everything else sorts alphabetically on the cell text (chips sort by their labels).
# Rows carry data-group (0 = flagged/judge-detection/quality-fail — the judging scope;
# 1 = the rest): group 0 always stays on top, and the sort applies within each group.
SORT_JS = """<script>
(function () {
  function cellText(tr, idx) {
    var c = tr.children[idx];
    if (!c) return "";
    // cells may carry an explicit sort key (e.g. panel severity) overriding their text
    if (c.hasAttribute("data-key")) return c.getAttribute("data-key");
    return (c.textContent || "").trim();
  }
  function numKey(v) {
    if (v === "" || v === "\\u2013" || v === "-") return null;
    var m = v.match(/^(-?\\d+(?:\\.\\d+)?)$/);
    return m ? parseFloat(m[1]) : undefined;
  }
  document.querySelectorAll("table.sortable").forEach(function (tbl) {
    var ths = Array.prototype.slice.call(tbl.tHead.querySelectorAll("th"));
    var tbody = tbl.tBodies[0];
    ths.forEach(function (th, idx) {
      th.addEventListener("click", function () {
        var asc = th.getAttribute("data-sort") !== "asc";
        ths.forEach(function (o) { o.removeAttribute("data-sort"); });
        th.setAttribute("data-sort", asc ? "asc" : "desc");
        var rows = Array.prototype.slice.call(tbody.rows);
        var keys = rows.map(function (r) { return numKey(cellText(r, idx)); });
        var numeric = keys.some(function (k) { return typeof k === "number"; })
          && keys.every(function (k) { return k === null || typeof k === "number"; });
        rows.sort(function (a, b) {
          var ga = parseInt(a.getAttribute("data-group") || "0", 10);
          var gb = parseInt(b.getAttribute("data-group") || "0", 10);
          if (ga !== gb) return ga - gb;  // scope group pinned on top, either direction
          var va = cellText(a, idx), vb = cellText(b, idx), cmp;
          if (numeric) {
            var ka = numKey(va), kb = numKey(vb);
            cmp = (ka === null ? -Infinity : ka) - (kb === null ? -Infinity : kb);
          } else { cmp = va.localeCompare(vb); }
          return asc ? cmp : -cmp;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
  });
})();
</script>"""
INDEX_HTML = (INDEX_HTML.replace("__SORT_JS__", SORT_JS)
              .replace("__TOTOP__", TOTOP_HTML).replace("__TOPNAV__", TOPNAV))

VISUALS_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>InferenceBench visuals</title><style>__CSS__</style></head><body>
__TOPNAV__
<h1>Panel verdicts by population</h1>
<img class="figure" src="/visuals/categories.svg">
<h1>Overall</h1>
<img class="figure" src="/visuals/overall.svg">
</body></html>""".replace("__CSS__", CSS).replace("__TOPNAV__", TOPNAV)

RUN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>{{ m.anon_id }} · InferenceBench</title><style>__CSS__</style></head><body>
<p><a href="/">&larr; all runs</a></p>
<div class="banner{% if m.invalid_or_reward_hack %} flagged{% endif %}">
 <span class="big">{{ m.anon_id }}</span>
 {% if m.invalid_or_reward_hack %}<span class="badge flag">⚑ invalid_or_reward_hack</span>{% endif %}
 <div class="kv">agent <b>{{ m.agent }}</b> · harness <b>{{ m.harness }}</b> ·
  scenario <b>{{ m.scenario }}</b> · time <b>{{ m.time_taken }}</b>
  (budget {{ m.budget_hours }}h) · {{ m.n_trace_events }} events ·
  logs: <a href="/log/{{ m.anon_id }}/server">server.log</a> ·
  <a href="/log/{{ m.anon_id }}/eval">eval.log</a></div>
 <div class="judgebox {{ contamination.cls }}"><b>contamination judge:</b>
  {{ contamination.full or 'no output' }}</div>
 <div class="judgebox {{ disallowed.cls }}"><b>disallowed-model judge:</b>
  {{ disallowed.full or 'no output' }}</div>
</div>

{{ judge_html|safe }}

{% if metrics %}
<div class="banner">
 <b>final eval</b> · served model:
 {% if served_ok %}<span class="mono">{{ metrics.model_id }}</span>
 {% else %}<span class="mono servedbad">{{ metrics.model_id }}</span> (≠ base model){% endif %}
 · quality gate:
 {% if metrics.quality_check and metrics.quality_check.pass %}<span class="chip clean">pass</span>
 {% elif metrics.quality_check %}<span class="chip bad">fail</span>{% else %}–{% endif %}
 {% if metrics.quality_check %}
  <div class="kv">{% for name, ds in metrics.quality_check.datasets.items() %}
   {{ name }}: <b>{{ '%.3f'|format(ds.observed_accuracy) }}</b> vs baseline
   {{ '%.3f'|format(ds.baseline_accuracy) }} (ratio {{ '%.3f'|format(ds.ratio) }},
   gate {{ ds.tau }}){% endfor %}</div>
 {% endif %}
 <table style="margin-top:.5rem"><thead><tr><th>profile</th><th class="num">ok/total</th>
  <th class="num">ttft p50</th><th class="num">tpot p50</th><th class="num">itl p50</th>
  <th class="num">gen tok/s</th><th class="num">req/s</th></tr></thead><tbody>
 {% for name, p in metrics.profiles.items() %}
  <tr><td>{{ name }}</td>
   <td class="num">{{ p.success_count }}/{{ p.request_count }}</td>
   <td class="num">{{ '%.4f'|format(p.ttft.p50) }}</td>
   <td class="num">{{ '%.4f'|format(p.tpot.p50) }}</td>
   <td class="num">{{ '%.4f'|format(p.itl.p50) }}</td>
   <td class="num">{{ '%.1f'|format(p.generation_throughput_tokens_per_s) }}</td>
   <td class="num">{{ '%.3f'|format(p.request_throughput_req_per_s) }}</td></tr>
 {% endfor %}
 </tbody></table>
</div>
{% else %}
<div class="banner"><b>final eval</b>: none produced</div>
{% endif %}

{{ trace_html|safe }}
{{ nav_html|safe }}
__TOTOP__
</body></html>""".replace("__CSS__", CSS).replace("__TOTOP__", TOTOP_HTML)


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    rows = index_rows()
    return render_template_string(
        INDEX_HTML, rows=rows, tab="trajectories",
        n_flagged=sum(r["flagged"] for r in rows),
        n_trip_judge=sum(r["trip_judge"] for r in rows),
        n_trip_quality=sum(r["trip_quality"] for r in rows),
        n_no_check=sum(r["no_check"] for r in rows))


@app.route("/visuals")
def visuals_page():
    return render_template_string(VISUALS_HTML, tab="visuals")


@app.route("/visuals/categories.svg")
def visuals_categories_svg():
    return Response(categories_figure(), mimetype="image/svg+xml")


@app.route("/visuals/overall.svg")
def visuals_overall_svg():
    return Response(overall_figure(), mimetype="image/svg+xml")


@app.route("/run/<run_id>")
def run_page(run_id: str):
    d = run_dir(run_id)
    meta = load_meta(d)
    metrics = load_metrics(d)
    with open(d / "trace.jsonl") as f:
        events = [json.loads(line) for line in f if line.strip()]
    trace_html = "\n".join(render_event(e) for e in events)
    j = meta.get("judgements") or {}
    judgements = load_judgements(run_id)
    return render_template_string(
        RUN_HTML, m=meta, metrics=metrics,
        served_ok=(metrics or {}).get("model_id") == BASE_MODEL,
        contamination=judge_chip(j.get("contamination_judgement"), "contamination"),
        disallowed=judge_chip(j.get("disallowed_model_judgement"), "disallowed"),
        judge_html=render_judgement_panel(judgements),
        nav_html=render_nav(judgements),
        trace_html=trace_html)


@app.route("/log/<run_id>/<which>")
def log_page(run_id: str, which: str):
    if which not in ("server", "eval"):
        abort(404)
    p = run_dir(run_id) / "logs" / f"{which}.log"
    if not p.exists():
        abort(404)
    return Response(p.read_text(errors="replace"), mimetype="text/plain")


if __name__ == "__main__":
    if not RUNS.is_dir():
        raise SystemExit(
            f"{RUNS} not found — run `uv run python mats/inferencebench/download.py` first.")
    app.run(host="127.0.0.1", port=5002, debug=False)
