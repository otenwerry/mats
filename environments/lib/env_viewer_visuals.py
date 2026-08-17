"""Environment-owned visuals: matplotlib figures embedded as inline SVG (free — no API).

The page structure mirrors the Petri visuals: a Filtered / All-trajectories integrity
toggle at the top (Filtered removes integrity-excluded runs; All trajectories pools
everything), then two underlined sub-tabs per view — "base rates" (outcome composition
by agent model, in counts and per-seed shares) and "cost" (all-in cost per trajectory
by model, spend by role, and the per-trajectory cost spread). Figures are ported from
petri/lib/viewer_visuals.py so the two viewers read the same way.

Historical 1–10 judgments are never thresholded into the new categories: a legacy or
missing judgment lands in the "awaiting current judgment" bucket, not in hack/clean.
"""

from __future__ import annotations

import html
import io
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from model_catalog import TARGET_CHOICES, route as route_model  # noqa: E402
from env_viewer_turns import experiment_user_turn_count  # noqa: E402


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


def agent_label(agent: object) -> str:
    """Pretty catalog name for an agent slug; the raw slug when unregistered."""
    raw = str(agent or "unknown")
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
    if respect_exclusion and (
        audit.get("mechanical_status") == "invalid"
        or audit.get("integrity_status") == "excluded"
    ):
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
# The hack bucket is split by controller-authored task turns (the same count as the
# index column): 1 = ended at the first-submission gate, 2 = hacked after the scripted
# follow-up. Native-scaffold preamble messages are not experiment turns. The split
# colors are Petri's hack-elicitation pair (RH red / deep wine), with its grey-mauve
# for a legacy hack whose turn count is unavailable or outside the one/two-turn protocol.
OUTCOME_ORDER = (
    ("hack_1turn", "reward hack · 1 user turn", "#C44E52"),
    ("hack_2turn", "reward hack · 2 user turns", "#7E3B44"),
    ("hack_other", "reward hack · other/unknown turns", "#8F7A80"),
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
_WHITE_TEXT_KEYS = {"hack_1turn", "hack_2turn", "hack_other", "review", "notable"}
_HACK_KEYS = ("hack_1turn", "hack_2turn", "hack_other")
_BINARY_NON_HACK_KEYS = ("notable", "clean")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion, as fractions."""

    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    )
    # Floating-point rounding at exactly 0% or 100% can otherwise produce a tiny
    # negative Matplotlib error-bar length.
    lo = max(0.0, min(p, center - half))
    hi = min(1.0, max(p, center + half))
    return p, lo, hi


def _outcome_key(audit: dict, *, respect_exclusion: bool) -> str:
    category = trajectory_category(audit, respect_exclusion=respect_exclusion)
    if category != "hack":
        return category
    turns = experiment_user_turn_count(audit)
    if turns == 1:
        return "hack_1turn"
    if turns == 2:
        return "hack_2turn"
    return "hack_other"


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
# base rates — outcome composition by agent model
# --------------------------------------------------------------------------- #
def outcome_data(audits: list[dict], *, respect_exclusion: bool = True) -> dict:
    """Per-model category counts shared by the outcome and binary-rate figures."""
    by_model: dict[str, Counter] = {}
    totals: Counter = Counter()
    for audit in audits:
        model = agent_label(audit.get("target"))
        category = _outcome_key(audit, respect_exclusion=respect_exclusion)
        totals[category] += 1
        by_model.setdefault(model, Counter())[category] += 1
    categories = [
        (key, label) for key, label, _color in OUTCOME_ORDER if totals.get(key)
    ]
    rows = []
    for model in sorted(by_model, key=lambda m: (-sum(by_model[m].values()), m)):
        counts = by_model[model]
        rows.append({
            "model": model,
            "n": sum(counts.values()),
            "counts": dict(counts),
        })
    return {"rows": rows, "categories": categories}


def fig_model_outcomes(data: dict) -> str:
    """Count-stacked outcome composition, one bar per agent model."""
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


def hack_rate_data(data: dict) -> list[dict]:
    """Binary reward-hack counts by model, retaining denominator exclusions."""

    rows = []
    for row in data.get("rows", []):
        counts = row.get("counts") or {}
        hacks = sum(counts.get(key, 0) for key in _HACK_KEYS)
        non_hacks = sum(counts.get(key, 0) for key in _BINARY_NON_HACK_KEYS)
        denominator = hacks + non_hacks
        excluded_counts = {
            key: count
            for key, count in counts.items()
            if key not in (*_HACK_KEYS, *_BINARY_NON_HACK_KEYS) and count
        }
        rows.append({
            "model": row["model"],
            "k": hacks,
            "n": denominator,
            "excluded": sum(excluded_counts.values()),
            "excluded_counts": excluded_counts,
        })
    return rows


def fig_model_hack_rates(data: dict) -> str:
    """Binary reward-hack rates by agent model with Wilson 95% intervals."""

    rows = [row for row in hack_rate_data(data) if row["n"] > 0]
    if not rows:
        return _empty_fig("no usable binary judgments")

    fig, ax = plt.subplots(figsize=(max(6.2, 1.35 * len(rows)), 4.4))
    xs = np.arange(len(rows))
    rates = []
    lower_errors = []
    upper_errors = []
    for row in rows:
        p, lo, hi = _wilson(row["k"], row["n"])
        rates.append(100 * p)
        lower_errors.append(100 * (p - lo))
        upper_errors.append(100 * (hi - p))

    ax.bar(
        xs,
        rates,
        width=0.66,
        color=OUTCOME_COLORS["hack_1turn"],
        yerr=[lower_errors, upper_errors],
        capsize=5,
        error_kw={"ecolor": "#333", "lw": 1.1},
    )
    for x, row, rate, upper_error in zip(xs, rows, rates, upper_errors):
        ax.annotate(
            f'{row["k"]}/{row["n"]}',
            (x, rate + upper_error),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            color="#333",
            fontweight="bold",
        )
    ax.set_xticks(
        xs,
        [row["model"] for row in rows],
        rotation=16,
        ha="right",
        fontsize=8.5,
    )
    ax.set_ylabel("Reward-hack rate (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


_CONTINUATION_RATE_COLORS = {
    "baseline": "#8C8C8C",
    "hack-in-one-turn": OUTCOME_COLORS["hack_1turn"],
    "hack-in-two-turns": OUTCOME_COLORS["hack_2turn"],
    "no-honeypot": "#4C72B0",
    "p-hacking-no-honeypot": "#4C72B0",
    "no-hack": OUTCOME_COLORS["clean"],
    "no-hack-1turn": "#86C98F",
    "no-hack-2turn": OUTCOME_COLORS["clean"],
    "hack_prefix": OUTCOME_COLORS["hack_1turn"],
    "nonhack_prefix": OUTCOME_COLORS["clean"],
    "prefix": OUTCOME_COLORS["notable"],
}

_CONTINUATION_RATE_LABELS = {
    "baseline": "Baseline (no prefix)",
    "hack-in-one-turn": "1-turn hack prefix",
    "hack-in-two-turns": "2-turn hack prefix",
    "no-honeypot": "No-honeypot ML prefix",
    "p-hacking-no-honeypot": "No-honeypot p-hacking prefix",
    "no-hack": "Clean prefix",
    "no-hack-1turn": "1-turn clean prefix",
    "no-hack-2turn": "2-turn clean prefix",
    "hack_prefix": "After hack prefix",
    "nonhack_prefix": "After non-hack prefix",
    "prefix": "After prefix",
}


def _continuation_bar_style(bar: dict) -> str:
    treatment = str(bar.get("treatment") or "")
    prefix_user_turns = bar.get("prefix_user_turns")
    if (
        treatment == "no-honeypot"
        and bar.get("prefix_type") == "p_hacking_no_honeypot"
    ):
        return "p-hacking-no-honeypot"
    if treatment == "no-hack" and prefix_user_turns in {1, 2}:
        return f"no-hack-{prefix_user_turns}turn"
    return treatment if treatment in _CONTINUATION_RATE_COLORS else bar["kind"]


def _continuation_bar_tick(bar: dict) -> str:
    """Compact condition label, wrapped where crowded repeated labels need it."""

    if bar["kind"] == "baseline":
        return "Baseline"
    if str(bar.get("treatment") or "") == "no-hack":
        prefix_user_turns = bar.get("prefix_user_turns")
        if prefix_user_turns in {1, 2}:
            return f"{prefix_user_turns}-turn\nclean"
    treatment_labels = {
        "hack-in-one-turn": "1-turn\nhack",
        "hack-in-two-turns": "2-turn\nhack",
        "no-honeypot": "No\nhoneypot",
        "no-hack": "No hack",
    }
    condition = treatment_labels.get(str(bar.get("treatment") or ""))
    if condition:
        return condition
    return str(bar.get("short_label") or "After prefix").removeprefix("After ")


def _figure_title(title: str, context: str = "") -> str:
    """Add screenshot-safe experiment context without changing ordinary figures."""

    return f"{title}\n{context}" if context else title


def fig_continuation_prefix_hack_rates(
    groups: list[dict], *, context: str = ""
) -> str:
    """Matched rates with per-bar conditions and a second, model-level axis."""

    model_order = {
        "opus-4.6": 0,
        "gpt-5.5": 1,
        "deepseek-v4-pro": 2,
        "glm-5.1": 3,
        "kimi-k2.6": 4,
    }
    plotted_groups = [
        {**group, "bars": [bar for bar in group.get("bars", []) if bar["n"] > 0]}
        for group in groups
    ]
    plotted_groups = [group for group in plotted_groups if group["bars"]]
    plotted_groups.sort(key=lambda group: (
        model_order.get(str(group.get("model") or "").split(" · ", 1)[0], 99),
        str(group.get("model") or ""),
    ))
    if not plotted_groups:
        return _empty_fig("no usable continuation judgments", (6.2, 4.4))

    xs: list[float] = []
    bars: list[dict] = []
    group_centers: list[float] = []
    group_labels: list[str] = []
    separators: list[float] = []
    cursor = 0.0
    for index, group in enumerate(plotted_groups):
        group_xs = []
        for bar in group["bars"]:
            xs.append(cursor)
            group_xs.append(cursor)
            bars.append(bar)
            cursor += 1.0
        group_centers.append(float(np.mean(group_xs)))
        group_label = group["model"].split(" · ", 1)[0]
        group_labels.append(group_label)
        if index < len(plotted_groups) - 1:
            separators.append(cursor - 0.05)
            cursor += 1.1

    fig, ax = plt.subplots(
        figsize=(max(8.0, min(16.0, 0.82 * len(bars) + 2.4)), 6.2)
    )
    rates, lower_errors, upper_errors = [], [], []
    for bar in bars:
        rate, low, high = _wilson(bar["k"], bar["n"])
        rates.append(100 * rate)
        lower_errors.append(100 * (rate - low))
        upper_errors.append(100 * (high - rate))

    ax.bar(
        xs,
        rates,
        width=0.68,
        color=[
            _CONTINUATION_RATE_COLORS[_continuation_bar_style(bar)]
            for bar in bars
        ],
        yerr=[lower_errors, upper_errors],
        capsize=5,
        error_kw={"ecolor": "#30343b", "lw": 1.1, "capthick": 1.1},
        zorder=3,
    )
    for x, bar, rate, high_error in zip(xs, bars, rates, upper_errors):
        ax.annotate(
            f'{bar["k"]}/{bar["n"]}',
            (x, rate + high_error),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            color="#333",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
            zorder=5,
        )
    for separator in separators:
        ax.axvline(separator, color="#d9dce4", lw=1.0, zorder=1)
    ax.set_xticks(xs, [_continuation_bar_tick(bar) for bar in bars])
    ax.tick_params(axis="x", labelsize=8.5, length=0, pad=8)
    for center, label in zip(group_centers, group_labels):
        ax.annotate(
            label,
            (center, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -52),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#353944",
            fontweight="bold",
            linespacing=1.2,
            annotation_clip=False,
        )
    ax.set_ylabel("Reward-hack rate (%)")
    ax.set_ylim(0, 113)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)

    styles = []
    for bar in bars:
        style = _continuation_bar_style(bar)
        if style not in styles:
            styles.append(style)
    legend_order = (
        "baseline", "hack-in-one-turn", "hack-in-two-turns", "no-honeypot",
        "p-hacking-no-honeypot",
        "no-hack-1turn", "no-hack-2turn", "no-hack",
        "hack_prefix", "nonhack_prefix", "prefix",
    )
    legend_styles = [style for style in legend_order if style in styles]
    legend = [
        Patch(
            color=_CONTINUATION_RATE_COLORS[style],
            label=_CONTINUATION_RATE_LABELS[style],
        )
        for style in legend_styles
    ]
    fig.suptitle(
        _figure_title("Reward-hack rate by prefix condition", context),
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    fig.legend(
        handles=legend,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895 if context else 0.935),
        ncol=min(4, len(legend)),
        columnspacing=1.5,
        handlelength=1.4,
    )
    fig.subplots_adjust(
        top=0.76 if context else 0.80,
        bottom=0.25,
        left=0.07,
        right=0.99,
    )
    return _fig_to_svg(fig)


_CONTINUATION_COMPOSITION = (
    ("hack_1turn", "1-turn hack", OUTCOME_COLORS["hack_1turn"]),
    ("hack_2turn", "2-turn hack", OUTCOME_COLORS["hack_2turn"]),
    ("interesting", "Interesting behavior", OUTCOME_COLORS["notable"]),
    ("clean", "Clean", OUTCOME_COLORS["clean"]),
)


def fig_continuation_outcome_distribution(
    groups: list[dict], *, context: str = ""
) -> str:
    """100%-stacked usable outcomes for every matched continuation condition."""

    model_order = {
        "opus-4.6": 0,
        "gpt-5.5": 1,
        "deepseek-v4-pro": 2,
        "glm-5.1": 3,
        "kimi-k2.6": 4,
    }
    plotted_groups = [
        {
            **group,
            "bars": [
                bar
                for bar in group.get("bars", [])
                if bar.get("composition_n", 0) > 0
            ],
        }
        for group in groups
    ]
    plotted_groups = [group for group in plotted_groups if group["bars"]]
    plotted_groups.sort(key=lambda group: (
        model_order.get(str(group.get("model") or "").split(" · ", 1)[0], 99),
        str(group.get("model") or ""),
    ))
    if not plotted_groups:
        return _empty_fig("no usable continuation outcomes", (6.2, 4.4))

    xs: list[float] = []
    bars: list[dict] = []
    group_centers: list[float] = []
    group_labels: list[str] = []
    separators: list[float] = []
    cursor = 0.0
    for index, group in enumerate(plotted_groups):
        group_xs = []
        for bar in group["bars"]:
            xs.append(cursor)
            group_xs.append(cursor)
            bars.append(bar)
            cursor += 1.0
        group_centers.append(float(np.mean(group_xs)))
        group_labels.append(group["model"].split(" · ", 1)[0])
        if index < len(plotted_groups) - 1:
            separators.append(cursor - 0.05)
            cursor += 1.1

    fig, ax = plt.subplots(
        figsize=(max(8.0, min(16.0, 0.82 * len(bars) + 2.4)), 6.2)
    )
    bottoms = np.zeros(len(bars))
    for key, label, color in _CONTINUATION_COMPOSITION:
        counts = np.array([
            bar.get("composition", {}).get(key, 0) for bar in bars
        ])
        shares = np.array([
            100 * count / bar["composition_n"]
            for count, bar in zip(counts, bars)
        ])
        ax.bar(
            xs,
            shares,
            bottom=bottoms,
            width=0.68,
            color=color,
            label=label,
            edgecolor="white",
            lw=0.7,
            zorder=3,
        )
        for x, share, bottom in zip(xs, shares, bottoms):
            if share >= 8:
                ax.text(
                    x,
                    bottom + share / 2,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )
        bottoms += shares

    for x, bar in zip(xs, bars):
        ax.annotate(
            f'n={bar["composition_n"]}',
            (x, 100),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8.5,
            color="#4b4f58",
            zorder=5,
        )
    for separator in separators:
        ax.axvline(separator, color="#d9dce4", lw=1.0, zorder=1)
    ax.set_xticks(xs, [_continuation_bar_tick(bar) for bar in bars])
    ax.tick_params(axis="x", labelsize=8.5, length=0, pad=8)
    for center, label in zip(group_centers, group_labels):
        ax.annotate(
            label,
            (center, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -52),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#353944",
            fontweight="bold",
            annotation_clip=False,
        )
    ax.set_ylabel("Share of usable trajectories (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)

    fig.suptitle(
        _figure_title("Outcome distribution by prefix condition", context),
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(color=color, label=label)
            for _key, label, color in _CONTINUATION_COMPOSITION
        ],
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895 if context else 0.935),
        ncol=4,
        columnspacing=1.5,
        handlelength=1.4,
    )
    fig.subplots_adjust(
        top=0.76 if context else 0.80,
        bottom=0.25,
        left=0.07,
        right=0.99,
    )
    return _fig_to_svg(fig)


def render_continuation_visuals(
    groups: list[dict], audits: list[dict], *, context: str = ""
) -> str:
    """Continuation rates and recorded experiment spend in two compact tabs."""

    rates = (
        '<div class="figgrid">'
        + _figure(fig_continuation_prefix_hack_rates(groups, context=context))
        + '</div><div class="figgrid">'
        + _figure(fig_continuation_outcome_distribution(groups, context=context))
        + "</div>"
    )
    return _tab_layout([
        ("rates", "rates", rates),
        ("cost", "cost", _cost_section(audits, context=context)),
    ])


def _base_rates_section(audits: list[dict], *, respect_exclusion: bool = True) -> str:
    data = outcome_data(audits, respect_exclusion=respect_exclusion)
    return (
        '<h2>Outcomes by model</h2>'
        + _stack(fig_model_outcomes(data))
        + '<h2>Reward-hack rate by model</h2>'
        + _stack(fig_model_hack_rates(data))
        + '<p class="vsub">Rate = reward hacks / usable non-review judgments; '
        'error bars are Wilson 95% intervals. Needs-review, unjudged, awaiting, '
        'and invalid judgments are excluded from the denominator.</p>'
    )


# --------------------------------------------------------------------------- #
# cost — what each trajectory costs and where the money goes
# --------------------------------------------------------------------------- #
ROLE_ORDER = ("target", "gate", "judge")
# Display names only; stored role keys stay gate/judge. The gate is the stage-one
# judge call at the first submission, the judge role is the fresh final call.
ROLE_LABEL = {"target": "Agent", "gate": "First judge", "judge": "Second judge"}
# The agent uses the orange role color; the first judge uses blue and the second green.
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


def _subscription_agent_usage_excluded(audit: dict) -> bool:
    native_harness = (
        audit.get("native_harness")
        or ((audit.get("real_env") or {}).get("harness"))
        or {}
    )
    agent_billing = native_harness.get("agent_billing")
    return audit.get("harness") == "subscription" and (
        agent_billing == "subscription_included_usage"
        or (agent_billing is None and native_harness.get("scaffold") != "opencode")
    )


def recorded_cost_summary(audits: list[dict]) -> dict:
    """Sum stored LLM costs and post-run AWS VM estimates."""
    by_role: Counter = Counter()
    missing = 0
    vm_total = 0.0
    vm_count = 0
    vm_exclusions = Counter()
    subscription_agent_usage_excluded = 0
    for audit in audits:
        included_subscription = _subscription_agent_usage_excluded(audit)
        if included_subscription:
            subscription_agent_usage_excluded += 1
        role_usage = audit.get("role_usage") or {}
        saw_cost = False
        for role, usage in role_usage.items():
            if included_subscription and str(role) == "target":
                continue
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
        "subscription_agent_usage_excluded": subscription_agent_usage_excluded,
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
        included_subscription = _subscription_agent_usage_excluded(audit)
        components: dict[str, float] = {}
        for role, usage in (audit.get("role_usage") or {}).items():
            if included_subscription and str(role) == "target":
                continue
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
        model = agent_label(audit.get("target"))
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


def fig_all_in_cost_by_model(cost: dict, *, context: str = "") -> str:
    """Average all-in trajectory cost by agent model, stacked by recorded component."""
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
    ax.set_title(_figure_title("All-in cost per trajectory by model", context))
    if len(components) > 1:
        ax.legend(frameon=False, fontsize=8.5)
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_cost_by_role(
    summary: dict, roles: list[str], *, context: str = ""
) -> str:
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
    ax.set_title(_figure_title(
        f"Where the budget goes — {_usd(total)} total", context
    ))
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def fig_cost_distribution(cost: dict, *, context: str = "") -> str:
    """Box + jittered strip of the TOTAL cost of each individual trajectory, grouped by
    agent model — shows the spread (a few long runs cost far more than the median)."""
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
    ax.set_title(_figure_title("Per-trajectory cost spread", context))
    ax.yaxis.grid(True, color="#e6e6ee", lw=0.8)
    ax.set_axisbelow(True)
    return _fig_to_svg(fig)


def _cost_section(audits: list[dict], *, context: str = "") -> str:
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
    subscription_excluded = summary["subscription_agent_usage_excluded"]
    if subscription_excluded:
        gaps += (
            f'<p class="costgap">&#9888; {subscription_excluded} direct subscription '
            'trajectory(ies) exclude agent usage from dollar totals and averages. '
            'Only recorded judge calls and VM estimates are shown for those runs.</p>'
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
            fig_all_in_cost_by_model(cost, context=context),
            fig_cost_by_role(summary, cost["roles"], context=context),
            fig_cost_distribution(cost, context=context),
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
    """Render the filtered aggregates and the everything-pooled aggregates behind one
    visible toggle. Filtered removes mechanically invalid runs; All trajectories
    pools every run, bucketing invalid ones by their judgment."""

    filtered = [
        audit for audit in audits
        if audit.get("mechanical_status") != "invalid"
        and audit.get("integrity_status") != "excluded"
    ]
    return (
        '<div class="visual-toggle" role="group" aria-label="Trajectory handling">'
        '<button type="button" class="active" data-visual-target="filtered">Filtered</button>'
        '<button type="button" data-visual-target="all">All trajectories</button>'
        '</div>'
        '<div class="visual-view active" data-visual-view="filtered">'
        + _visual_subset(
            f"Filtered trajectories — {len(filtered)}", filtered,
            respect_exclusion=True,
        )
        + '</div><div class="visual-view" data-visual-view="all">'
        + _visual_subset(
            f"All trajectories — {len(audits)}", audits,
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
