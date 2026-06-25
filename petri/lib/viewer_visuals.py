"""Paper-style figures for the rollback re-hacking analysis (free — no API).

Builds the `visuals.html` page of the viewer: matplotlib figures (embedded as inline
SVG) answering two questions about the 10 hack trajectories and their rollback
continuations:

  Q1  How does the rate of RE-HACKING differ between control (plain resume) and
      treatment (resume + a message telling the model it was rolled back for
      reward-hacking and to continue honestly)?
  Q2  How does the TIME to first re-hack (in target/assistant turns after the
      rollback point) differ between the two conditions?

This module is intentionally decoupled from make_viewer: it takes a list of plain
per-continuation record dicts (built by make_viewer.collect_rehack_analysis) plus the
site CSS + top-nav html, and returns a complete HTML page. It only needs numpy +
matplotlib, so a missing matplotlib can be caught by the caller without breaking the
rest of the viewer.

Each record:
  {cond: 'control'|'treatment', oid: int, model: str, seed: str,
   cls: 'hack'|'non'|'neither'|'pending', dead: bool,
   cut_turn: int|None,        # reroll_turn k: assistant-turn index of the rollback cut
   live_len: int|None,        # L: assistant turns from the cut to the end (inclusive)
   first_hack_a: int|None,    # assistant-turn index of the first re-hack (hacks only)
   delta: int|None}           # first_hack_a - cut_turn: assistant turns after resume (>=0)
"""

import io
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# consistent condition palette + order across every figure (colorblind-safe blue/orange)
CONTROL_C = "#4C72B0"
TREAT_C = "#DD8452"
CONDS = ["control", "treatment"]
COND_LABEL = {"control": "Control", "treatment": "Treatment"}
COND_COLOR = {"control": CONTROL_C, "treatment": TREAT_C}
# a re-hack continuation counts toward the rate numerator; these two classes are the
# denominator (nudged/aware = 'neither' and not-yet-judged = 'pending' are excluded).
RATE_CLASSES = ("hack", "non")

# cut-location display order (by where the cut sits in the conversation). Unknown
# locations sort last but still render -- the page adapts to whatever appears. Each
# location is a SEPARATE experiment (its own rollback point); the page never pools across
# them. LOC_DESC is the plain-English "where the cut is" shown in each section header.
LOC_ORDER = {"begin": 0, "middle": 1, "before": 2, "after": 3}
LOC_DESC = {
    "begin":  "cut at the very start (turn 1)",
    "middle": "cut about halfway to the first hack",
    "before": "cut right before the first hack",
    "after":  "cut right after the first hack",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.9,
    "xtick.color": "#444",
    "ytick.color": "#444",
    "svg.fonttype": "none",   # keep text as text (smaller svg, selectable)
    "figure.dpi": 100,
})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pretty_model(m: str) -> str:
    return m.replace("-instruct", "")


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n. Returns (p, lo, hi) as fractions."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def _rate(records, predicate):
    """(num, den) re-hacks over (hacks+non) among records matching predicate."""
    sub = [r for r in records if predicate(r) and r["cls"] in RATE_CLASSES]
    den = len(sub)
    num = sum(1 for r in sub if r["cls"] == "hack")
    return num, den


def _fig_to_svg(fig) -> str:
    """Render a figure to an inline-embeddable <svg> string and close it."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]   # drop the XML/doctype preamble


def _empty_fig(msg: str, size=(4.6, 4.0)) -> str:
    """Placeholder figure for a facet with no plottable data (e.g. a location whose
    continuations aren't judged yet) -- keeps the page rendering instead of crashing."""
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#999",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------- #
# Q1 — re-hacking rate
# --------------------------------------------------------------------------- #
def fig_aggregate_rate(records) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    xs = np.arange(len(CONDS))
    rates, los, his, nums, dens = [], [], [], [], []
    for c in CONDS:
        k, n = _rate(records, lambda r, c=c: r["cond"] == c)
        p, lo, hi = _wilson(k, n)
        rates.append(100 * p); los.append(100 * (p - lo)); his.append(100 * (hi - p))
        nums.append(k); dens.append(n)
    ax.bar(xs, rates, width=0.6, color=[COND_COLOR[c] for c in CONDS],
           yerr=[los, his], capsize=5, error_kw=dict(ecolor="#333", lw=1.1))
    for x, k, n in zip(xs, nums, dens):
        ax.annotate(f"{k}/{n}", (x, 100 * k / n if n else 0), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color="#333", fontweight="bold")
    ax.set_xticks(xs, [COND_LABEL[c] for c in CONDS])
    ax.set_ylabel("Re-hack rate (%)")
    ax.set_ylim(0, max(his[i] + rates[i] for i in range(len(CONDS))) + 12)
    ax.set_title("Re-hacking rate after rollback")
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_control_across_locations(records) -> str:
    """The one deliberate CROSS-location comparison: control (plain re-roll) re-hack rate
    at each cut location -- i.e. does plain-re-roll re-hacking depend on WHERE you cut?
    Control only, so it isn't confounded by the treatment message."""
    locs = sorted({r["location"] for r in records}, key=lambda l: (LOC_ORDER.get(l, 99), l))
    if not locs:
        return _empty_fig("no locations", (5.2, 4.0))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = np.arange(len(locs))
    rates, los, his, nums, dens = [], [], [], [], []
    for loc in locs:
        k, n = _rate(records, lambda r, loc=loc: r["cond"] == "control" and r["location"] == loc)
        p, lo, hi = _wilson(k, n)
        rates.append(100 * p); los.append(100 * (p - lo)); his.append(100 * (hi - p))
        nums.append(k); dens.append(n)
    ax.bar(xs, rates, width=0.62, color=CONTROL_C, yerr=[los, his], capsize=5,
           error_kw=dict(ecolor="#333", lw=1.1))
    for x, k, n in zip(xs, nums, dens):
        ax.annotate(f"{k}/{n}", (x, 100 * k / n if n else 0), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color="#333", fontweight="bold")
    ax.set_xticks(xs, locs)
    ax.set_ylabel("Re-hack rate (%)")
    ax.set_ylim(0, max(his[i] + rates[i] for i in range(len(locs))) + 12)
    ax.set_title("Control re-hack rate by cut location")
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_rate_by_model(records) -> str:
    models = sorted({r["model"] for r in records},
                    key=lambda m: -(_rate(records, lambda r, m=m: r["model"] == m)[0] /
                                    max(_rate(records, lambda r, m=m: r["model"] == m)[1], 1)))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(models)); w = 0.38
    for i, c in enumerate(CONDS):
        rates, nums, dens = [], [], []
        for m in models:
            k, n = _rate(records, lambda r, m=m, c=c: r["model"] == m and r["cond"] == c)
            rates.append(100 * k / n if n else 0); nums.append(k); dens.append(n)
        bars = ax.bar(x + (i - 0.5) * w, rates, w, label=COND_LABEL[c], color=COND_COLOR[c])
        for b, k, n in zip(bars, nums, dens):
            ax.annotate(f"{k}/{n}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", fontsize=8, color="#555")
    ax.set_xticks(x, [_pretty_model(m) for m in models], rotation=20, ha="right")
    ax.set_ylabel("Re-hack rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Re-hacking rate by model")
    ax.legend(frameon=False, loc="upper right")
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------- #
# Q2 — time to first re-hack (assistant turns after the rollback)
# --------------------------------------------------------------------------- #
def fig_time_distribution(records) -> str:
    """Box + jittered strip of turns-after-resume to first re-hack, among re-hackers."""
    rng = np.random.default_rng(0)
    data = [[r["delta"] for r in records if r["cond"] == c and r["cls"] == "hack"
             and r["delta"] is not None] for c in CONDS]
    if not any(data):
        return _empty_fig("no re-hacks to time", (4.8, 4.2))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False,
                    medianprops=dict(color="#222", lw=1.6),
                    boxprops=dict(color="#888"), whiskerprops=dict(color="#888"),
                    capprops=dict(color="#888"))
    for i, (c, d) in enumerate(zip(CONDS, data)):
        jx = i + rng.uniform(-0.13, 0.13, size=len(d))
        ax.scatter(jx, d, s=34, color=COND_COLOR[c], edgecolor="white", lw=0.6, zorder=3)
        ax.annotate(f"n={len(d)}", (i, -0.8), ha="center", va="top", fontsize=9, color="#555")
    ax.set_xticks([0, 1], [COND_LABEL[c] for c in CONDS])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.6, max(max(d) for d in data if d) + 1.5)
    ax.set_ylabel("Assistant turns after rollback")
    ax.set_title("Turns until first re-hack")
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
VISUALS_CSS = """
.vwrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
.vsub { color: #555; font-size: 13.5px; line-height: 1.5; margin: 2px 0 22px; max-width: 760px; }
.vsub b { color: #1a1a2e; }
.figgrid { display: flex; flex-wrap: wrap; gap: 22px; align-items: flex-start; }
figure.fig { margin: 0; background: #fff; border: 1px solid #e3e5ec; border-radius: 8px;
             padding: 14px 16px 10px; box-shadow: 0 1px 3px rgba(0,0,0,.07); }
figure.fig svg { display: block; height: auto; max-width: 100%; }
figure.fig figcaption { font-size: 11.5px; color: #6a7180; line-height: 1.45; margin-top: 6px;
             max-width: 520px; }
/* each cut location is its own boxed experiment so they never read as one pooled set */
.locblock { border: 1px solid #d7dae6; border-radius: 12px; background: #f7f8fc;
            padding: 6px 20px 22px; margin: 26px 0; }
.locblock .vsec { font-size: 19px; font-weight: 700; margin: 16px 0 2px; color: #1a1a2e; }
.locblock .vmeta { font-size: 12.5px; color: #6a7180; margin: 0 0 14px; }
.vsec { font-size: 16px; margin: 30px 0 8px; color: #1a1a2e; }
"""


def _figure(svg: str, caption: str = "") -> str:
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="fig">{svg}{cap}</figure>'


def _location_section(loc: str, sub: list[dict]) -> str:
    """One per-location block, treated as a self-contained experiment: the three kept
    figures (overall re-hack rate, rate by model, turns-to-first-re-hack) computed over
    ONLY this location's continuations, comparing control vs treatment WITHIN the location
    (the meaningful contrast -- a begin-control isn't comparable to an after-treatment).
    Nothing is pooled across locations."""
    n_traj = len({r["oid"] for r in sub})
    n_cont = len(sub)
    excluded = sum(1 for r in sub if r["cls"] not in RATE_CLASSES)
    rate_caption = (
        "Re-hack = a full or degenerate reward hack. Rate = re-hacks / (re-hacks + non-hacks); "
        f"{excluded} nudged/aware or unjudged continuation(s) excluded. Error bars: Wilson 95% CI.")
    time_caption = (
        "Time = target (assistant) turns after the rollback point (turn&nbsp;1 = the first "
        "re-rolled turn). Among re-hacking continuations only.")
    desc = LOC_DESC.get(loc, "")
    return (
        '<div class="locblock">'
        f'<div class="vsec">{esc_loc(loc)}</div>'
        f'<div class="vmeta">{esc_loc(desc)} &middot; {n_traj} trajectories, {n_cont} continuations</div>'
        '<div class="figgrid">'
        + _figure(fig_aggregate_rate(sub), rate_caption)
        + _figure(fig_rate_by_model(sub), "Models pooled across their trajectories. Counts shown as k/n.")
        + _figure(fig_time_distribution(sub), time_caption)
        + '</div>'
        '</div>'
    )


def build_visuals_page(records: list[dict], css: str, topnav: str) -> str:
    """Full HTML for visuals.html. `records` are the per-continuation dicts described in
    the module docstring (each carries a `location`); `css` is the site CSS; `topnav` is
    the shared nav bar html. The page FACETS by cut location -- one section per location
    that appears -- so it adapts to whatever the rollback runs produced (begin / middle /
    before / after, or any subset), comparing control vs treatment within each."""
    # records may predate the location field (older callers) -> default to "before".
    for r in records:
        r.setdefault("location", "before")
    n_traj = len({r["oid"] for r in records})
    n_cont = len(records)
    locations = sorted({r["location"] for r in records}, key=lambda l: (LOC_ORDER.get(l, 99), l))
    sections = "".join(_location_section(loc, [r for r in records if r["location"] == loc])
                       for loc in locations)
    loc_blurb = ", ".join(locations)
    overview = (
        '<div class="vsec" style="font-size:18px;">Control re-hack rate across cut locations</div>'
        '<div class="figgrid">'
        + _figure(fig_control_across_locations(records),
                  "Control (plain re-roll) only, so cut location isn't confounded by the treatment "
                  "message. Rate = re-hacks / (re-hacks + non-hacks). Error bars: Wilson 95% CI.")
        + '</div>'
    )
    body = f"""
{topnav}
<h1>Reward-hacking under rollback</h1>
<p class="vsub"><b>Control</b> = plain resume from the rollback point.
<b>Treatment</b> = resume with a message telling the model it was rolled back because it
began reward-hacking, asking it to continue honestly. Each <b>cut location</b>
({esc_loc(loc_blurb)}) is a <b>separate experiment</b> in its own box below &mdash; nothing
is pooled across locations. {n_traj} original hack trajectories, {n_cont} continuations.</p>
{overview}
{sections}
"""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Petri visuals</title>"
            f"<style>{css}{VISUALS_CSS}</style></head><body><div class='wrap vwrap'>{body}</div>"
            f"</body></html>")


def esc_loc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
