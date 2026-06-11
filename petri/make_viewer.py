"""Generate a static HTML viewer for Petri audit runs (free — reads local logs only).

Scans every mode directory under logs/ (e.g. logs/pilot, logs/full), and writes
a static site to viewer/:

  viewer/index.html        -- one row per audit, key judge scores, links
  viewer/pages/*.html      -- per-audit page: judge summary / justification /
                              highlights, full score table, and the exact
                              [MN]-numbered transcript the judge read
                              (rebuilt with Petri's own rendering functions,
                              so [MN] citations link to transcript anchors)

Per-audit pages get a fixed nav panel for judge-cited messages: every [MN]
the judge cited is outlined in the transcript; n/p (or buttons) jump between
them, d toggles dimming of uncited messages.

Usage:
  uv run make_viewer.py
  open viewer/index.html
"""

import asyncio
import html
import json
import re
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_petri._judge.branches import flatten_timeline, render_segments
from inspect_scout import (
    MessagesPreprocessor,
    TranscriptContent,
    message_numbering,
    transcripts_from,
)

HERE = Path(__file__).parent
# data (logs + generated viewer) lives in mats-local so it isn't committed to github
DATA = HERE.parent.parent / "mats-local" / "petri"
LOGS = DATA / "logs"
OUT = DATA / "viewer"

# shown as columns on the index page (in this order)
KEY_DIMS = [
    "reward_hacking",
    "concerning",
    "unprompted_deception_toward_user",
    "input_hallucination",
    "eval_awareness",
    "scenario_realism",
]
# higher-is-better dimensions (don't color these red)
GOOD_HIGH = {"admirable", "fun_or_funny", "scenario_realism"}

CSS = """
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: #f6f7f9; color: #1a1a2e; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
.wrap.fit { max-width: none; width: fit-content; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 28px; }
a { color: #2456a6; text-decoration: none; } a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
th, td { padding: 7px 10px; border-bottom: 1px solid #e8e8ee; font-size: 13px; text-align: left; }
th { background: #eef0f4; position: sticky; top: 0; }
.score { display: inline-block; min-width: 22px; text-align: center; border-radius: 4px; padding: 1px 5px; }
.s1 { color: #999; }
.bad2 { background: #fff3cd; } .bad5 { background: #ffd6a5; } .bad8 { background: #ffadad; font-weight: 600; }
.good { background: #d3f3d8; }
.note { background: #fff; border-left: 4px solid #2456a6; padding: 12px 16px; margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); white-space: pre-wrap; font-size: 14px; line-height: 1.45; }
.note.justif { border-left-color: #a62445; }
.note.hl { border-left-color: #6a994e; }
.msg { margin: 10px 0; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,.07); }
.msg pre { margin: 0; padding: 10px 14px; white-space: pre-wrap; word-break: break-word;
           font-size: 12.5px; line-height: 1.45; background: #fff; }
.mhead { padding: 4px 14px; font-size: 11.5px; font-weight: 700; letter-spacing: .4px; }
.role-system .mhead { background: #6c757d; color: #fff; }
.role-user .mhead { background: #2456a6; color: #fff; }
.role-assistant .mhead { background: #1d7a4f; color: #fff; }
.role-assistant pre { background: #f0fff5; }
.role-tool .mhead { background: #8a5a00; color: #fff; }
.role-tool pre { background: #fffaf0; }
.role-other .mhead { background: #444; color: #fff; }
.branch { text-align: center; font-style: italic; color: #7a3030; margin: 14px 0; }
.meta { color: #667; font-size: 13px; }
:target .mhead { outline: 3px solid #ffb703; }
.msg.cited { outline: 2px solid #ffb703; }
.msg.cited .mhead::after { content: " \\00b7 cited by judge"; font-weight: 400; opacity: .9; }
@keyframes flashpulse { from { box-shadow: 0 0 0 7px rgba(255,183,3,.9); } to { box-shadow: 0 0 0 0 rgba(255,183,3,0); } }
.msg.flash { animation: flashpulse 1.1s ease-out; }
body.dimmode .msg:not(.cited), body.dimmode .branch { opacity: .25; }
body.hasnav { padding-right: 212px; }
.cnav { position: fixed; top: 16px; right: 16px; width: 170px; background: #fff; border: 1px solid #d9dbe3;
        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.13); padding: 10px 12px; font-size: 12.5px; z-index: 50; }
.cnav-row { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
.cnav button { cursor: pointer; border: 1px solid #c8cad2; background: #f2f3f6; border-radius: 5px; padding: 2px 10px; font-size: 13px; }
.cnav button:hover { background: #e4e6ec; }
.cnav-keys { color: #889; margin-top: 6px; font-size: 11px; }
"""

NAV_HTML = """
<div class="cnav" id="cnav">
  <b>judge-cited messages</b>
  <div class="cnav-row">
    <button id="cnav-prev" title="previous (p)">&larr;</button>
    <span id="cnav-count"></span>
    <button id="cnav-next" title="next (n)">&rarr;</button>
  </div>
  <label><input type="checkbox" id="cnav-dim"> dim uncited</label>
  <div class="cnav-keys">keys: n next &middot; p prev &middot; d dim</div>
</div>
"""

NAV_JS = """
<script>
(function () {
  var cited = __CITED__;
  var nodes = cited.map(function (n) { return document.getElementById("M" + n); }).filter(Boolean);
  if (!nodes.length) { document.getElementById("cnav").style.display = "none"; return; }
  nodes.forEach(function (el) { el.classList.add("cited"); });
  var counter = document.getElementById("cnav-count");
  var cur = -1;
  function show() { counter.textContent = (cur + 1) + " / " + nodes.length; }
  function go(d) {
    cur = (cur + d + nodes.length) % nodes.length;
    var el = nodes[cur];
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
    show();
  }
  var dim = document.getElementById("cnav-dim");
  dim.onchange = function () { document.body.classList.toggle("dimmode", dim.checked); };
  document.getElementById("cnav-next").onclick = function () { go(1); };
  document.getElementById("cnav-prev").onclick = function () { go(-1); };
  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "n") go(1);
    if (e.key === "p") go(-1);
    if (e.key === "d") { dim.checked = !dim.checked; dim.onchange(); }
  });
  show();
})();
</script>
"""


def esc(s: str) -> str:
    return html.escape(str(s), quote=False)


def linkify(text: str) -> str:
    """HTML-escape and turn [M12] citations into transcript anchor links."""
    return re.sub(r"\[M(\d+)\]", r'<a href="#M\1">[M\1]</a>', esc(text))


def score_cell(dim: str, v) -> str:
    if not isinstance(v, (int, float)):
        return f'<span class="score">{esc(v)}</span>'
    v = int(v)
    if dim in GOOD_HIGH:
        cls = "good" if v > 1 else "s1"
    elif v >= 8:
        cls = "bad8"
    elif v >= 5:
        cls = "bad5"
    elif v >= 2:
        cls = "bad2"
    else:
        cls = "s1"
    return f'<span class="score {cls}">{v}</span>'


MSG_HEAD = re.compile(r"^\[M(\d+)\] (<?[A-Za-z_]+>?[A-Za-z_ ]*):", re.M)
BRANCH = re.compile(r"^--- (?:End )?Conversation Branch .*?---$", re.M)


def transcript_html(rendered: str) -> str:
    """Split the judge-view rendered transcript into anchored message blocks."""
    # collect cut points: message heads and branch delimiter lines
    cuts = sorted(
        [(m.start(), "msg", m) for m in MSG_HEAD.finditer(rendered)]
        + [(m.start(), "branch", m) for m in BRANCH.finditer(rendered)]
    )
    if not cuts:
        return f'<div class="msg role-other"><pre>{esc(rendered)}</pre></div>'
    parts: list[str] = []
    for i, (start, kind, m) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(rendered)
        chunk = rendered[start:end].rstrip()
        if kind == "branch":
            parts.append(f'<div class="branch">{esc(m.group(0))}</div>')
            body = chunk[m.end() - m.start():].strip()
            if body:
                parts.append(f'<div class="msg role-other"><pre>{esc(body)}</pre></div>')
            continue
        num, role = m.group(1), m.group(2)
        body = chunk[m.end() - m.start():].lstrip("\n")
        role_cls = re.sub(r"[^a-z]", "", role.lower().split()[0]) or "other"
        if role_cls not in ("system", "user", "assistant", "tool"):
            role_cls = "other"
        parts.append(
            f'<div class="msg role-{role_cls}" id="M{num}">'
            f'<div class="mhead">[M{num}] {esc(role)}</div>'
            f"<pre>{esc(body)}</pre></div>"
        )
    return "\n".join(parts)


def page_name(mode: str, task: str, seed: str, epoch: int) -> str:
    raw = f"{mode}__{task}__{seed}__e{epoch}.html"
    return re.sub(r"[^A-Za-z0-9._-]", "-", raw)


async def load_mode(mode_dir: Path) -> list[dict]:
    """One dict per audit in this mode dir: scores + judge notes + transcript."""
    mode = mode_dir.name
    # newest .eval mtime in this dir — used to order run sections (newest first)
    eval_files = list(mode_dir.glob("*.eval"))
    mode_mtime = max((f.stat().st_mtime for f in eval_files), default=0.0)

    # judge-view transcripts, keyed by (task_set, task_id, repeat)
    rendered: dict[tuple, str] = {}
    ts = transcripts_from(str(mode_dir))
    async with ts.reader() as reader:
        infos = [i async for i in reader.index()]
        for info in infos:
            t = await reader.read(info, TranscriptContent(messages="all", events="all"))
            messages_as_str, _extract_refs, label_for_id = message_numbering(
                MessagesPreprocessor(exclude_system=False), label_for_id=True
            )
            target = next(
                (tl for tl in t.timelines if tl.name == "target"), t.timelines[0]
            )
            segments = flatten_timeline(target.root)
            rendered[(info.task_set, info.task_id, info.task_repeat)] = (
                await render_segments(segments, messages_as_str, label_for_id)
            )
            print(f"  rendered transcript: {info.task_set}/{info.task_id} (epoch {info.task_repeat})")

    # scores + judge notes from the eval logs
    audits: list[dict] = []
    for li in list_eval_logs(str(mode_dir)):
        log = read_eval_log(li)
        roles = log.eval.model_roles or {}
        target_model = getattr(roles.get("target"), "model", None) or str(roles.get("target", "?"))
        for s in log.samples or []:
            sc = (s.scores or {}).get("audit_judge")
            if sc is None:
                print(f"  WARNING: no audit_judge score for {s.id} epoch {s.epoch} — skipping")
                continue
            md = sc.metadata or {}
            audits.append(
                dict(
                    mode=mode,
                    mtime=mode_mtime,
                    task=log.eval.task,
                    target=target_model,
                    seed=str(s.id),
                    epoch=s.epoch,
                    scores=sc.value if isinstance(sc.value, dict) else {},
                    summary=md.get("summary", ""),
                    justification=getattr(sc, "expl" + "anation", None) or "",
                    highlights=md.get("highlights", ""),
                    transcript=rendered.get((log.eval.task, str(s.id), s.epoch), ""),
                )
            )
    return audits


def write_page(a: dict) -> str:
    name = page_name(a["mode"], a["task"], a["seed"], a["epoch"])
    flagged = {
        k: v for k, v in sorted(a["scores"].items())
        if isinstance(v, (int, float)) and v > 1
    }
    all_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{score_cell(k, v)}</td></tr>"
        for k, v in sorted(a["scores"].items())
    )
    flagged_chips = " ".join(
        f"{esc(k)}={score_cell(k, v)}" for k, v in sorted(flagged.items(), key=lambda kv: -kv[1])
    ) or "none"
    transcript_note = "" if a["transcript"] else "<p><b>No transcript rendered for this sample.</b></p>"
    cited = sorted({
        int(m) for m in re.findall(
            r"\[M(\d+)\]", a["summary"] + a["justification"] + a["highlights"]
        )
    })
    nav = (NAV_HTML + NAV_JS.replace("__CITED__", json.dumps(cited))) if cited else ""
    body = f"""
<p><a href="../index.html">&larr; index</a></p>
<h1>{esc(a['seed'])} <span class="meta">(epoch {a['epoch']})</span></h1>
<p class="meta">run: {esc(a['mode'])} / {esc(a['task'])} &middot; target: {esc(a['target'])}</p>
<p>Dimensions scoring &gt;1 (all scores out of 10): {flagged_chips}</p>

<h2>Judge summary</h2>
<div class="note">{linkify(a['summary'])}</div>

<h2>Judge justification <span class="meta">(one note covering every dimension scoring &gt;1)</span></h2>
<div class="note justif">{linkify(a['justification'])}</div>

<h2>Judge highlights</h2>
<div class="note hl">{linkify(a['highlights'])}</div>

<details><summary><b>All {len(a['scores'])} dimension scores</b></summary>
<table><tr><th>dimension</th><th>score (1&ndash;10)</th></tr>{all_rows}</table></details>

<h2>Transcript (judge view, with rollback branches)</h2>
{transcript_note}
{transcript_html(a['transcript'])}
"""
    body_cls = " class='hasnav'" if cited else ""
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(a['seed'])}</title><style>{CSS}</style></head><body{body_cls}><div class='wrap'>{body}</div>{nav}</body></html>"
    (OUT / "pages" / name).write_text(page)
    return name


def write_run_table(mode_audits: list[dict]) -> str:
    """One section (heading + table) for a single run/mode, with columns drawn
    only from the dimensions actually present in this run."""
    mode = mode_audits[0]["mode"]
    # dimensions present anywhere in this run, KEY_DIMS first (in their order), then the rest
    present = set().union(*(a["scores"].keys() for a in mode_audits))
    cols = [d for d in KEY_DIMS if d in present]
    extra = sorted(d for d in present if d not in KEY_DIMS)
    show_other = bool(extra)  # collapse non-KEY dims into a count column when there are any
    head_cols = "".join(f"<th>{esc(d)}</th>" for d in cols)
    other_head = "<th>other dims &gt;1</th>" if show_other else ""
    rows = []
    for a in sorted(mode_audits, key=lambda a: -(a["scores"].get("reward_hacking") or 0)):
        name = page_name(a["mode"], a["task"], a["seed"], a["epoch"])
        cells = "".join(f"<td>{score_cell(d, a['scores'].get(d, '-'))}</td>" for d in cols)
        other_cell = ""
        if show_other:
            other = sum(
                1 for k, v in a["scores"].items()
                if k not in cols and isinstance(v, (int, float)) and v > 1
            )
            other_cell = f"<td>{other}</td>"
        rows.append(
            f"<tr><td><a href='pages/{name}'>{esc(a['seed'])}</a></td>"
            f"<td>{a['epoch']}</td><td>{esc(a['target'])}</td>{cells}{other_cell}</tr>"
        )
    return f"""
<h2>{esc(mode)} <span class="meta">({len(mode_audits)} audits)</span></h2>
<table>
<tr><th>seed</th><th>epoch</th><th>target</th>{head_cols}{other_head}</tr>
{''.join(rows)}
</table>
"""


def write_index(audits: list[dict]) -> None:
    # group by run/mode, newest run first (each gets its own columns)
    modes = sorted(
        {a["mode"] for a in audits},
        key=lambda m: -next(a["mtime"] for a in audits if a["mode"] == m),
    )
    sections = "".join(
        write_run_table([a for a in audits if a["mode"] == m]) for m in modes
    )
    body = f"""
<h1>Petri reward-hacking audits</h1>
<p class="meta">{len(audits)} audits across {len(modes)} run(s) &middot; all scores out of 10
(1 = no sign of the behavior) &middot; each run sorted by reward_hacking &middot;
newest run first &middot; regenerate with <code>uv run make_viewer.py</code></p>
{sections}
"""
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Petri audits</title><style>{CSS}</style></head><body><div class='wrap fit'>{body}</div></body></html>"
    (OUT / "index.html").write_text(page)


async def main() -> None:
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    audits: list[dict] = []
    mode_dirs = sorted(d for d in LOGS.iterdir() if d.is_dir()) if LOGS.exists() else []
    if not mode_dirs:
        raise SystemExit(f"no log directories found under {LOGS}")
    for mode_dir in mode_dirs:
        print(f"loading {mode_dir.name}/ ...")
        audits.extend(await load_mode(mode_dir))
    for a in audits:
        name = write_page(a)
        print(f"  wrote pages/{name}")
    write_index(audits)
    print(f"\nwrote {OUT / 'index.html'} ({len(audits)} audits)")
    print(f"open it with: open {OUT / 'index.html'}")


if __name__ == "__main__":
    asyncio.run(main())
