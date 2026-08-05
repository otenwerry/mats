"""Environment-owned visuals: matplotlib figures embedded as inline SVG (free — no API).

The page structure mirrors the Petri visuals: an Included/Excluded integrity toggle at
the top, then two underlined sub-tabs per view — "base rates" (outcome composition by
target model, in counts and per-seed shares) and "cost" (all-in cost per trajectory by
model, spend by role, and the per-trajectory cost spread). Figures are ported from
petri/lib/viewer_visuals.py so the two viewers read the same way.

Historical 1–10 judgments are never thresholded into the new categories: a legacy or
missing judgment lands in the "awaiting current judgment" bucket, not in hack/clean.
"""

from __future__ import annotations

import html
import io
import textwrap
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from model_catalog import TARGET_CHOICES, route as route_model  # noqa: E402


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


TARGET_LABELS = {
    route_model(slug): name for name, slug in TARGET_CHOICES.items()
}


def target_label(target: object) -> str:
    """Pretty catalog name for a target slug; the raw slug when unregistered."""
    raw = str(target or "unknown")
    return TARGET_LABELS.get(route_model(raw), raw)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def _dimension(audit: dict, key: str) -> dict | None:
    judgment = audit.get("judgment") or {}
    if judgment.get("format") != "structured":
        return None
    return next(
        (
            dimension for dimension in judgment.get("dimensions") or []
            if dimension.get("key") == key
        ),
        None,
    )


def trajectory_category(audit: dict, *, respect_exclusion: bool = True) -> str:
    """The one outcome bucket a trajectory belongs to (index sections + visuals)."""
    if audit.get("judge_failure"):
        return "unjudged"
    if respect_exclusion and audit.get("integrity_status") == "excluded":
        return "excluded"
    judgment = audit.get("judgment") or {}
    if judgment.get("format") != "structured":
        return "awaiting"
    reward = _dimension(audit, "reward_hacking")
    if not reward or reward.get("status") != "ok":
        return "invalid"
    if reward.get("requires_review"):
        return "review"
    if reward.get("is_hack"):
        return "hack"
    values = reward.get("value") or []
    if isinstance(values, list) and any(str(value) != "clean" for value in values):
        return "notable"
    return "clean"


# Outcome buckets in stack order, with the Petri outcome palette: reward hacks keep the
# established RH red, review takes the reversed orange, notable takes the interesting
# purple, clean stays green, and the not-usable buckets stay visually recessive.
OUTCOME_ORDER = (
    ("hack", "reward hack", "#C44E52"),
    ("review", "needs review", "#DD8452"),
    ("notable", "notable behavior", "#8172B3"),
    ("clean", "clean", "#55A868"),
    ("unjudged", "not judged", "#b9bcc6"),
    ("invalid", "invalid judgment", "#cdd0d9"),
    ("awaiting", "awaiting current judgment", "#a7adba"),
    ("excluded", "excluded", "#8C8C8C"),
)
OUTCOME_COLORS = {key: color for key, _label, color in OUTCOME_ORDER}
# count labels drawn in white on the saturated segments, dark on the light ones
_WHITE_TEXT_KEYS = {"hack", "review", "notable"}


def _fig_to_svg(fig) -> str:
    """Render a figure to an inline-embeddable <svg> string and close it."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]   # drop the XML/doctype preamble


def _empty_fig(msg: str, size=(4.6, 4.0)) -> str:
    """Placeholder figure for a view with no plottable data — keeps the page rendering."""
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#999",
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _fig_to_svg(fig)


def _figure(svg: str, caption: str = "") -> str:
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="fig">{svg}{cap}</figure>'


def _stack(*svgs: str) -> str:
    """Lay figures out ONE PER ROW so nothing gets squeezed side-by-side."""
    return "".join(f'<div class="figgrid">{_figure(s)}</div>' for s in svgs if s)


# --------------------------------------------------------------------------- #
# base rates — outcome composition by target model
# --------------------------------------------------------------------------- #
def outcome_data(audits: list[dict], *, respect_exclusion: bool = True) -> dict:
    """Rows/categories/seeds for the outcome figures, in the Petri data shape."""
    by_model: dict[str, Counter] = {}
    by_model_seed: dict[str, dict[str, Counter]] = {}
    totals: Counter = Counter()
    for audit in audits:
        model = target_label(audit.get("target"))
        seed = str(audit.get("seed") or "unknown")
        category = trajectory_category(audit, respect_exclusion=respect_exclusion)
        totals[category] += 1
        by_model.setdefault(model, Counter())[category] += 1
        by_model_seed.setdefault(model, {}).setdefault(seed, Counter())[category] += 1
    categories = [
        (key, label) for key, label, _color in OUTCOME_ORDER if totals.get(key)
    ]
    seeds = sorted({
        seed for cells in by_model_seed.values() for seed in cells
    })
    rows = []
    for model in sorted(by_model, key=lambda m: (-sum(by_model[m].values()), m)):
        counts = by_model[model]
        rows.append({
            "model": model,
            "n": sum(counts.values()),
            "counts": dict(counts),
            "by_seed": [
                {
                    "seed": seed,
                    "n": sum(cell.values()),
                    "counts": dict(cell),
                }
                for seed, cell in sorted(by_model_seed[model].items())
            ],
        })
    return {"rows": rows, "categories": categories, "seeds": seeds}


def fig_model_outcomes(data: dict) -> str:
    """Count-stacked outcome composition, one bar per target model."""
    rows = [r for r in data.get("rows", []) if r.get("n")]
    categories = data.get("categories", [])
    if not rows or not categories:
        return _empty_fig("no trajectories")
    fig, ax = plt.subplots(figsize=(max(6.2, 1.35 * len(rows)), 4.8))
    xs = np.arange(len(rows))
    bottoms = np.zeros(len(rows))
    for key, label in categories:
        raw = np.array([r["counts"].get(key, 0) for r in rows])
        color = OUTCOME_COLORS.get(key, "#8C8C8C")
        ax.bar(xs, raw, bottom=bottoms, width=0.66, color=color, label=label,
               edgecolor="white", lw=0.5)
        for x, count, bottom in zip(xs, raw, bottoms):
            if count:
                ax.annotate(str(count), (x, bottom + count / 2), ha="center",
                            va="center", fontsize=8,
                            color="white" if key in _WHITE_TEXT_KEYS else "#222",
                            fontweight="bold")
        bottoms += raw
    ax.set_xticks(xs, [r["model"] for r in rows],
                  rotation=16, ha="right", fontsize=8.5)
    ax.set_ylabel("Trajectories")
    ax.set_ylim(0, max(r["n"] for r in rows) * 1.05)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=3)
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_model_seed_outcomes(data: dict) -> str:
    """Outcome-composition small multiples: seed rows x model columns.

    Each cell is one 100%-stacked bar using the exact same outcome buckets and colors
    as ``fig_model_outcomes``. Normalizing within a cell makes differently sized
    model/seed groups comparable; the stored denominator and every non-zero segment
    count remain printed on the graph.
    """
    rows = [r for r in data.get("rows", []) if r.get("n")]
    seeds = data.get("seeds", [])
    categories = data.get("categories", [])
    if not rows or not seeds or not categories:
        return _empty_fig("no model-by-seed trajectories")

    nrows, ncols = len(seeds), len(rows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(6.8, 2.15 * ncols), max(4.8, 1.35 * nrows + 1.25)),
        squeeze=False,
        sharey=True,
    )
    for col_idx, row in enumerate(rows):
        by_seed = {cell["seed"]: cell for cell in row.get("by_seed", [])}
        for row_idx, seed in enumerate(seeds):
            ax = axes[row_idx][col_idx]
            cell = by_seed.get(seed) or {"n": 0, "counts": {}}
            n = cell.get("n", 0)
            bottom = 0.0
            if n:
                for key, _ in categories:
                    count = cell.get("counts", {}).get(key, 0)
                    if not count:
                        continue
                    height = 100 * count / n
                    ax.bar(
                        [0], [height], bottom=[bottom], width=0.72,
                        color=OUTCOME_COLORS.get(key, "#8C8C8C"),
                        edgecolor="white", linewidth=0.5,
                    )
                    ax.annotate(
                        str(count), (0, bottom + height / 2),
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if key in _WHITE_TEXT_KEYS else "#222",
                    )
                    bottom += height
                ax.text(0, 102.5, f"n={n}", ha="center", va="bottom",
                        fontsize=7.5, color="#555")
            else:
                ax.set_facecolor("#f2f2f5")
                ax.text(0, 50, "no runs", ha="center", va="center",
                        fontsize=7.5, color="#999")
            ax.set_xlim(-0.62, 0.62)
            ax.set_ylim(0, 111)
            ax.set_xticks([])
            ax.yaxis.grid(True, color="#e6e6ee", linewidth=0.6)
            ax.set_axisbelow(True)
            if row_idx == 0:
                ax.set_title(row["model"], fontsize=8.5, pad=8)
            if col_idx == 0:
                ax.set_ylabel(
                    "\n".join(textwrap.wrap(seed.replace("_", " "), 18)),
                    rotation=0, ha="right", va="center",
                    labelpad=34, fontsize=8.5, fontweight="bold",
                )
                ax.set_yticks([0, 50, 100], ["0", "50", "100"], fontsize=7.5)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)

    handles = [
        Patch(facecolor=OUTCOME_COLORS.get(key, "#8C8C8C"), label=label)
        for key, label in categories
    ]
    fig.suptitle("Outcomes by seed and model", fontsize=12, fontweight="bold", y=0.995)
    fig.supylabel("Share of trajectories (%)", fontsize=9, x=0.01)
    fig.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), ncol=3)
    fig.tight_layout(rect=(0.035, 0.11, 1, 0.96), h_pad=0.8, w_pad=0.8)
    return _fig_to_svg(fig)


def _base_rates_section(audits: list[dict], *, respect_exclusion: bool = True) -> str:
    data = outcome_data(audits, respect_exclusion=respect_exclusion)
    return (
        '<h2>Outcomes by model</h2><div class="figgrid outcomegrid">'
        + _figure(fig_model_outcomes(data))
        + _figure(fig_model_seed_outcomes(data))
        + "</div>"
    )


# --------------------------------------------------------------------------- #
# cost — what each trajectory costs and where the money goes
# --------------------------------------------------------------------------- #
ROLE_ORDER = ("target", "gate", "judge")
# Display names only; stored role keys stay gate/judge. The gate is the stage-one
# judge call at the first submission, the judge role is the fresh final call.
ROLE_LABEL = {"target": "Target", "gate": "First judge", "judge": "Second judge"}
# the Petri role palette (orange target, green judge); gate takes the blue slot
ROLE_COLOR = {"target": "#DD8452", "gate": "#4C72B0", "judge": "#55A868"}
VM_KEY = "vm"
VM_COLOR = "#8172B3"
_EXTRA_ROLE_COLORS = ("#937860", "#DA8BC3", "#CCB974", "#64B5CD")


def _usd(x: float) -> str:
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    if x >= 0.01:
        return f"${x:.3f}"
    if x > 0:
        return f"${x:.4f}"
    return "$0"


def recorded_cost_summary(audits: list[dict]) -> dict:
    """Sum stored LLM costs and post-run AWS VM estimates."""
    by_role: Counter = Counter()
    missing = 0
    vm_total = 0.0
    vm_count = 0
    vm_exclusions = Counter()
    for audit in audits:
        role_usage = audit.get("role_usage") or {}
        saw_cost = False
        for role, usage in role_usage.items():
            if not isinstance(usage, dict) or usage.get("total_cost") is None:
                continue
            by_role[str(role)] += float(usage["total_cost"])
            saw_cost = True
        if not audit.get("retrospective_rejudge"):
            compute = ((audit.get("real_env") or {}).get("compute") or {})
            vm_cost = compute.get("estimated_vm_cost_usd")
            if isinstance(vm_cost, (int, float)):
                vm_total += float(vm_cost)
                vm_count += 1
                saw_cost = True
                for key in (
                    "s3_cost_excluded",
                    "ebs_cost_excluded",
                    "public_ipv4_cost_excluded",
                    "internet_data_transfer_cost_excluded",
                    "shared_runtime_cost_excluded",
                ):
                    if compute.get(key) is True:
                        vm_exclusions[key] += 1
        if not saw_cost:
            missing += 1
    return {
        "by_role": dict(by_role),
        "vm_total": vm_total,
        "vm_count": vm_count,
        "vm_exclusions": dict(vm_exclusions),
        "total": sum(by_role.values()) + vm_total,
        "trajectories_without_recorded_cost": missing,
    }


def cost_data(audits: list[dict]) -> dict:
    """Per-model and per-trajectory cost rows for the cost figures.

    Component attribution follows recorded_cost_summary exactly: stored per-role LLM
    costs, plus the AWS VM estimate on original runs only (a rejudge never re-counts
    its source's VM cost).
    """
    by_model: dict[str, dict] = {}
    per_traj: list[tuple[str, float]] = []
    roles_present: set[str] = set()
    for audit in audits:
        components: dict[str, float] = {}
        for role, usage in (audit.get("role_usage") or {}).items():
            if not isinstance(usage, dict) or usage.get("total_cost") is None:
                continue
            components[str(role)] = (
                components.get(str(role), 0.0) + float(usage["total_cost"])
            )
        if not audit.get("retrospective_rejudge"):
            compute = ((audit.get("real_env") or {}).get("compute") or {})
            vm_cost = compute.get("estimated_vm_cost_usd")
            if isinstance(vm_cost, (int, float)):
                components[VM_KEY] = components.get(VM_KEY, 0.0) + float(vm_cost)
        if not components:
            continue
        roles_present.update(key for key in components if key != VM_KEY)
        total = sum(components.values())
        model = target_label(audit.get("target"))
        per_traj.append((model, total))
        row = by_model.setdefault(
            model, {"model": model, "n": 0, "components": Counter(), "total": 0.0}
        )
        row["n"] += 1
        row["components"].update(components)
        row["total"] += total
    rows = []
    for row in sorted(
        by_model.values(), key=lambda r: (-r["total"] / r["n"], r["model"])
    ):
        rows.append({
            "model": row["model"],
            "n": row["n"],
            "components": dict(row["components"]),
            "mean": row["total"] / row["n"],
        })
    component_order = [role for role in ROLE_ORDER if role in roles_present]
    component_order += sorted(roles_present - set(ROLE_ORDER))
    return {"by_model": rows, "per_traj": per_traj, "roles": component_order}


def _component_color(key: str, extra_index: dict[str, str]) -> str:
    if key == VM_KEY:
        return VM_COLOR
    if key in ROLE_COLOR:
        return ROLE_COLOR[key]
    if key not in extra_index:
        extra_index[key] = _EXTRA_ROLE_COLORS[len(extra_index) % len(_EXTRA_ROLE_COLORS)]
    return extra_index[key]


def _component_label(key: str) -> str:
    return "VM estimate" if key == VM_KEY else ROLE_LABEL.get(key, key)


def fig_all_in_cost_by_model(cost: dict) -> str:
    """Average all-in trajectory cost by target model, stacked by recorded component."""
    rows = [r for r in cost.get("by_model", []) if r.get("n")]
    if not rows:
        return _empty_fig("no per-trajectory cost data")
    components = [
        key for key in (*cost.get("roles", ()), VM_KEY)
        if any(r["components"].get(key) for r in rows)
    ]
    extra_index: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(rows) + 1.1), 4.4))
    xs = np.arange(len(rows))
    bottoms = np.zeros(len(rows))
    for key in components:
        vals = np.array([r["components"].get(key, 0.0) / r["n"] for r in rows])
        ax.bar(xs, vals, bottom=bottoms, width=0.64,
               color=_component_color(key, extra_index),
               label=_component_label(key), edgecolor="white", lw=0.4)
        bottoms += vals
    for x, r in zip(xs, rows):
        ax.annotate(_usd(r["mean"]), (x, r["mean"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color="#333", fontweight="bold")
    ax.set_xticks(xs, [f"{r['model']}\n(n={r['n']})" for r in rows],
                  rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Average all-in cost per trajectory ($)")
    ax.set_ylim(0, max(r["mean"] for r in rows) * 1.20)
    ax.set_title("All-in cost per trajectory by model")
    if len(components) > 1:
        ax.legend(frameon=False, fontsize=8.5)
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_cost_by_role(summary: dict, roles: list[str]) -> str:
    """Total recorded spend split into the stored roles plus the VM estimate."""
    by_component = {
        role: float(summary.get("by_role", {}).get(role, 0.0)) for role in roles
    }
    if summary.get("vm_count"):
        by_component[VM_KEY] = float(summary.get("vm_total", 0.0))
    total = sum(by_component.values())
    if total <= 0:
        return _empty_fig("no cost data", (5.0, 4.0))
    components = [key for key, value in by_component.items() if value > 0]
    extra_index: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    xs = np.arange(len(components))
    vals = [by_component[key] for key in components]
    ax.bar(xs, vals, width=0.6,
           color=[_component_color(key, extra_index) for key in components])
    for x, key in zip(xs, components):
        value = by_component[key]
        ax.annotate(f"{_usd(value)}\n{100 * value / total:.0f}%", (x, value),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, color="#333", fontweight="bold")
    ax.set_xticks(xs, [_component_label(key) for key in components], fontsize=9)
    ax.set_ylabel("Total spend ($)")
    ax.set_ylim(0, max(vals) * 1.20)
    ax.set_title(f"Where the budget goes — {_usd(total)} total")
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_cost_distribution(cost: dict) -> str:
    """Box + jittered strip of the TOTAL cost of each individual trajectory, grouped by
    target model — shows the spread (a few long runs cost far more than the median)."""
    groups: dict[str, list] = {}
    for model, total in cost.get("per_traj", []):
        groups.setdefault(model, []).append(total)
    order = sorted(groups, key=lambda m: -float(np.median(groups[m])))
    if not order:
        return _empty_fig("no cost data")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.25 * len(order) + 1.0), 4.2))
    data = [groups[m] for m in order]
    pos = np.arange(len(order))
    ax.boxplot(data, positions=pos, widths=0.5, showfliers=False,
               medianprops=dict(color="#222", lw=1.6), boxprops=dict(color="#888"),
               whiskerprops=dict(color="#888"), capprops=dict(color="#888"))
    for i, m in enumerate(order):
        d = groups[m]
        jx = i + rng.uniform(-0.13, 0.13, size=len(d))
        ax.scatter(jx, d, s=26, color="#DD8452", edgecolor="white", lw=0.5, zorder=3)
    ax.set_xticks(pos, [f"{m}\n(n={len(groups[m])})" for m in order],
                  rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Cost per trajectory ($)")
    ax.set_ylim(0, max(max(d) for d in data) * 1.12)
    ax.set_title("Per-trajectory cost spread")
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def _cost_section(audits: list[dict]) -> str:
    """The 'Cost' tab: total-spend headline, visible gaps, and the three cost figures."""
    summary = recorded_cost_summary(audits)
    cost = cost_data(audits)
    headline = (
        '<div class="costtotal"><span>total recorded spend</span>'
        f'<b>{_usd(summary["total"])}</b></div>'
    )
    gaps = ""
    missing = summary["trajectories_without_recorded_cost"]
    if missing:
        gaps += (
            f'<p class="costgap">&#9888; {missing} trajectory(ies) have no recorded '
            'cost; totals and averages are partial.</p>'
        )
    if summary["vm_exclusions"]:
        gaps += (
            '<p class="costgap">&#9888; AWS VM estimates exclude the stored S3, EBS, '
            'public IPv4, internet-transfer, and shared-runtime cost categories.</p>'
        )
    return (
        "<h2>Cost</h2>"
        + headline
        + gaps
        + _stack(
            fig_all_in_cost_by_model(cost),
            fig_cost_by_role(summary, cost["roles"]),
            fig_cost_distribution(cost),
        )
    )


# --------------------------------------------------------------------------- #
# page assembly — integrity toggle wrapping the two sub-tabs
# --------------------------------------------------------------------------- #
def _tab_layout(tabs: list[tuple]) -> str:
    """Client-side sub-tab layout: `tabs` = [(key, label, html)], already filtered to
    non-empty panels, in display order. The first tab shows by default."""
    if not tabs:
        return ""
    bar = "".join(
        f'<button class="vsubtab{" active" if i == 0 else ""}" '
        f'data-vtab="{key}">{_esc(label)}</button>'
        for i, (key, label, _) in enumerate(tabs)
    )
    panels = "".join(
        f'<div class="vpanel{" active" if i == 0 else ""}" data-vpanel="{key}">{body}</div>'
        for i, (key, _, body) in enumerate(tabs)
    )
    script = (
        "<script>(function(root){if(!root)return;"
        "var ts=root.querySelectorAll('.vsubtab'),ps=root.querySelectorAll('.vpanel');"
        "ts.forEach(function(t){t.addEventListener('click',function(){"
        "ts.forEach(function(x){x.classList.remove('active')});"
        "ps.forEach(function(p){p.classList.remove('active')});"
        "t.classList.add('active');"
        "var key=t.getAttribute('data-vtab');"
        "ps.forEach(function(p){if(p.getAttribute('data-vpanel')===key)"
        "p.classList.add('active')});"
        "});});})(document.currentScript.previousElementSibling);</script>"
    )
    return f'<div class="vtablayout"><div class="vsubtabs">{bar}</div>{panels}</div>{script}'


def _visual_subset(title: str, audits: list[dict], *, respect_exclusion: bool) -> str:
    tabs = _tab_layout([
        (
            "base_rates",
            "base rates",
            _base_rates_section(audits, respect_exclusion=respect_exclusion),
        ),
        ("cost", "cost", _cost_section(audits)),
    ])
    return f'<p class="vsub">{_esc(title)}</p>{tabs}'


def render_visuals(audits: list[dict]) -> str:
    """Render separate included and excluded aggregates behind one visible toggle."""

    included = [
        audit for audit in audits if audit.get("integrity_status") != "excluded"
    ]
    excluded = [
        audit for audit in audits if audit.get("integrity_status") == "excluded"
    ]
    return (
        '<div class="visual-toggle" role="group" aria-label="Aggregate inclusion">'
        '<button type="button" class="active" data-visual-target="included">Included</button>'
        '<button type="button" data-visual-target="excluded">Excluded</button>'
        '</div>'
        '<div class="visual-view active" data-visual-view="included">'
        + _visual_subset(
            f"Included trajectories — {len(included)}", included,
            respect_exclusion=True,
        )
        + '</div><div class="visual-view" data-visual-view="excluded">'
        + _visual_subset(
            f"Excluded trajectories — {len(excluded)}", excluded,
            respect_exclusion=False,
        )
        + '</div>'
    )


VISUALS_CSS = """
.vsubtabs{margin:2px 0 20px;display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #dcdfe6}
.vsubtab{background:none;border:none;cursor:pointer;font-family:inherit;padding:3px 1px 8px;font-size:12.5px;font-weight:600;color:#6b7280;border-bottom:2px solid transparent;margin-bottom:-1px}
.vsubtab.active{color:#1558d6;border-bottom-color:#1558d6}.vsubtab:hover{color:#1558d6}
.vpanel{display:none}.vpanel.active{display:block}
.vsub{color:#555;font-size:13.5px;line-height:1.5;margin:2px 0 14px;max-width:760px}
.figgrid{display:flex;flex-wrap:wrap;gap:22px;align-items:flex-start;margin:0 0 22px}
.outcomegrid>figure.fig{flex:1 1 500px;min-width:0}
figure.fig{margin:0;background:#fff;border:1px solid #e3e5ec;border-radius:8px;padding:14px 16px 10px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
figure.fig svg{display:block;height:auto;max-width:100%}
figure.fig figcaption{font-size:11.5px;color:#6a7180;line-height:1.45;margin-top:6px;max-width:520px}
.costtotal{display:inline-flex;flex-direction:column;gap:2px;padding:10px 14px;margin:0 0 12px;border:1px solid #d9dce6;border-radius:8px;background:#f8f9fc}
.costtotal span{color:#687080;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.costtotal b{color:#1a1a2e;font-size:22px}
.costgap{color:#9b4b00;font-size:12.5px;margin:6px 0 14px}
.visual-toggle{display:inline-flex;background:#e8eaf0;border-radius:7px;padding:2px;margin:0 0 14px}.visual-toggle button{border:0;background:transparent;color:#596170;padding:5px 14px;border-radius:5px;font:600 12px inherit;cursor:pointer}.visual-toggle button.active{background:#fff;color:#1558d6;box-shadow:0 1px 3px rgba(0,0,0,.12)}.visual-view{display:none}.visual-view.active{display:block}
"""


VISUALS_JS = r"""<script>
(function () {
  var buttons = Array.prototype.slice.call(
    document.querySelectorAll("[data-visual-target]")
  );
  var views = Array.prototype.slice.call(
    document.querySelectorAll("[data-visual-view]")
  );
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      var target = button.getAttribute("data-visual-target");
      buttons.forEach(function (item) {
        item.classList.toggle("active", item === button);
      });
      views.forEach(function (view) {
        view.classList.toggle(
          "active", view.getAttribute("data-visual-view") === target
        );
      });
    });
  });
})();
</script>"""
