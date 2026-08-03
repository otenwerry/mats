"""Environment-owned summaries for the standalone viewer.

These visuals are categorical because the new judgment is categorical.  Historical
1–10 judgments are counted in a separate legacy row and are never thresholded into the
new labels.
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from typing import Any


def _esc(value: Any) -> str:
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


def categorical_summary(audits: list[dict]) -> dict:
    """Return queryable counts without imposing policy thresholds."""
    summary = {
        "total": len(audits),
        "structured": 0,
        "legacy_numeric": 0,
        "missing_judgment": 0,
        "reward_hacking": Counter(),
        "reward_hack_types": Counter(),
        "noticed_honeypot": Counter(),
        "environment_failure": Counter(),
        "deception_scores": Counter(),
        "by_target": defaultdict(lambda: {
            "total": 0, "structured": 0, "hack": 0, "no_hack": 0, "invalid": 0,
        }),
    }
    for audit in audits:
        target = str(audit.get("target") or "unknown")
        by_target = summary["by_target"][target]
        by_target["total"] += 1
        judgment = audit.get("judgment")
        if not judgment:
            summary["missing_judgment"] += 1
            by_target["invalid"] += 1
            continue
        if judgment.get("format") == "legacy_numeric":
            summary["legacy_numeric"] += 1
            continue
        if judgment.get("format") != "structured":
            summary["missing_judgment"] += 1
            by_target["invalid"] += 1
            continue
        summary["structured"] += 1
        by_target["structured"] += 1

        reward = _dimension(audit, "reward_hacking")
        if not reward or reward.get("status") != "ok":
            summary["reward_hacking"]["invalid"] += 1
            by_target["invalid"] += 1
        elif reward.get("value"):
            summary["reward_hacking"]["hack"] += 1
            by_target["hack"] += 1
            summary["reward_hack_types"].update(reward["value"])
        else:
            summary["reward_hacking"]["none"] += 1
            by_target["no_hack"] += 1

        for key in ("noticed_honeypot", "environment_failure"):
            dimension = _dimension(audit, key)
            if not dimension or dimension.get("status") != "ok":
                summary[key]["invalid"] += 1
            else:
                summary[key]["yes" if dimension.get("value") else "no"] += 1

        deception = _dimension(audit, "deception")
        if deception:
            if deception.get("status") == "ok":
                summary["deception_scores"][str(deception.get("value"))] += 1
            elif deception.get("status") == "not_applicable":
                summary["deception_scores"]["n/a"] += 1
            else:
                summary["deception_scores"]["invalid"] += 1

    # Plain dictionaries make the summary JSON/checkpoint friendly.
    for key in (
        "reward_hacking", "reward_hack_types", "noticed_honeypot",
        "environment_failure", "deception_scores",
    ):
        summary[key] = dict(summary[key])
    summary["by_target"] = dict(summary["by_target"])
    return summary


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


_COLORS = ("#5577aa", "#d06b4e", "#62a06b", "#a67bb5", "#999")


def _bar_group(title: str, counts: dict) -> str:
    total = sum(int(value) for value in counts.values())
    if not counts:
        return f'<section class="visual"><h2>{_esc(title)}</h2><p>no data</p></section>'
    rows = []
    for index, (label, count) in enumerate(
        sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    ):
        share = (100 * int(count) / total) if total else 0
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{_esc(label)}</span>'
            '<span class="bar-track">'
            f'<span class="bar-fill" style="width:{share:.2f}%;background:{_COLORS[index % len(_COLORS)]}"></span>'
            '</span>'
            f'<span class="bar-value">{int(count)} ({share:.0f}%)</span></div>'
        )
    return f'<section class="visual"><h2>{_esc(title)}</h2>{"".join(rows)}</section>'


def render_visuals(audits: list[dict]) -> str:
    """Render a compact set of new-schema summaries and explicit coverage notes."""
    summary = categorical_summary(audits)
    cost = recorded_cost_summary(audits)
    coverage = (
        '<section class="visual coverage"><h2>Judgment coverage</h2>'
        f'<p>{summary["structured"]} structured · '
        f'{summary["legacy_numeric"]} legacy numeric · '
        f'{summary["missing_judgment"]} missing</p>'
        '<p class="note">Legacy 1–10 results are not converted into the new categories.</p>'
        '</section>'
    )
    target_counts = {
        target: values["structured"] for target, values in summary["by_target"].items()
    }
    cost_rows = "".join(
        f'<tr><th>{_esc(role)}</th><td>${value:,.4f}</td></tr>'
        for role, value in sorted(cost["by_role"].items())
    )
    if cost["vm_count"]:
        cost_rows += (
            f'<tr><th>AWS VM estimate</th><td>${cost["vm_total"]:,.4f}</td></tr>'
        )
    exclusion_note = (
        '<p class="note">AWS estimates exclude the stored S3, EBS, public IPv4, '
        'internet-transfer, and shared-runtime cost categories.</p>'
        if cost["vm_exclusions"] else ""
    )
    cost_html = (
        '<section class="visual"><h2>Recorded cost</h2>'
        f'<table>{cost_rows}<tr><th>total</th><td>${cost["total"]:,.4f}</td></tr></table>'
        f'<p class="note">{cost["trajectories_without_recorded_cost"]} trajectory(ies) '
        'have no recorded cost; totals can therefore be incomplete.</p>'
        f'{exclusion_note}</section>'
    )
    return (
        '<div class="visual-grid">'
        + coverage
        + _bar_group("Structured judgments by target", target_counts)
        + _bar_group("Reward hacking", summary["reward_hacking"])
        + _bar_group("Reward-hack types", summary["reward_hack_types"])
        + _bar_group("Honeypot noticed", summary["noticed_honeypot"])
        + _bar_group("Environment failure", summary["environment_failure"])
        + _bar_group("Deception among applicable ML judgments", summary["deception_scores"])
        + cost_html
        + '</div>'
    )


VISUALS_CSS = """
.visual-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}
.visual{border:1px solid #dde0e8;border-radius:8px;padding:10px;background:#fff}.visual h2{font-size:15px;margin:0 0 8px}
.bar-row{display:grid;grid-template-columns:125px 1fr 72px;gap:7px;align-items:center;margin:5px 0;font-size:11px}
.bar-label{overflow-wrap:anywhere}.bar-track{height:13px;background:#eff1f5;border-radius:2px;overflow:hidden}.bar-fill{display:block;height:100%;min-width:1px}
.bar-value{text-align:right;font-variant-numeric:tabular-nums}.visual table{border-collapse:collapse}.visual th{text-align:left;padding-right:20px}.visual td{text-align:right}
.visual .note{color:#777;font-size:11px}.coverage{background:#f8f9fc}
"""
