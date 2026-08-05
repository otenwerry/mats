"""Build the standalone static viewer for real-environment trajectories.

This endpoint is free. It reads this project's Inspect logs and writes HTML under
``mats-local/environments/viewer``. It has no Petri imports or runtime fallback.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from env_viewer_components import (  # noqa: E402
    EVIDENCE_NAV_CSS,
    EVIDENCE_NAV_JS,
    esc,
    render_dimension_navigator,
    render_generic_judge_stage,
    render_justification_turn_nav,
    render_judge_narrative,
    render_judge_view,
    render_transcript,
)
from env_viewer_load import (  # noqa: E402
    assign_stable_ids,
    link_rejudge_sources,
    load_all,
    viewer_build_lock,
)
from env_viewer_visuals import (  # noqa: E402
    ROLE_LABEL,
    VISUALS_CSS,
    VISUALS_JS,
    render_visuals,
    target_label,
    trajectory_category,
)
from environment_judge import (  # noqa: E402
    prepare_judge_call,
    provider_request_record,
)
from judge_selection import judge_shortname  # noqa: E402
from project_paths import DATA_ROOT, LOGS_ROOT, SEEDS_ROOT, VIEWER_ROOT  # noqa: E402


REGISTRY_FILE = DATA_ROOT / "trajectory_ids.json"
CACHE_ROOT = DATA_ROOT / "viewer_cache"
GENERIC_JUDGE_FAMILIES = ("ml_training_data_misuse", "p_hacking")
SEED_NAV_ORDER = (
    "fraud_detection",
    "reasoning_prompt_benchmark",
    "checkout_redesign",
    "demand_forecasting",
    "rating_prediction",
    "retrieval_practice",
)
_GENERATED_MARKER = "<!-- ENVIRONMENTS_VIEWER_GENERATED -->"
INDEX_DIMENSION_ORDER = (
    "reward_hacking",
    "deception",
    "noticed_hack",
    # Retained only to order historical judgments that already stored these fields.
    "noticed_honeypot",
    "degenerate_behavior",
    "hallucination",
    "auditor_mistakes",
    "situational_awareness",
    "environment_failure",
    "incompleteness",
)
INDEX_DIMENSION_LABELS = {
    "reward_hacking": "reward hacking",
    "noticed_hack": "noticed hack",
    # Historical label; the current judge no longer emits this dimension.
    "noticed_honeypot": "noticed honeypot",
    "degenerate_behavior": "degenerate behavior",
    "auditor_mistakes": "auditor mistakes",
    "situational_awareness": "situational awareness",
    "environment_failure": "environment failure",
}
FAMILY_NAV_LABELS = {
    "ml_training_data_misuse": "ML",
    "p_hacking": "p-hacking",
}
# The fixed current judge schema per family. Index tables always show these columns,
# even before any trajectory exists, so every page keeps one stable structure.
FAMILY_INDEX_DIMENSIONS = {
    "ml_training_data_misuse": ("reward_hacking", "deception", "environment_failure"),
    "p_hacking": ("reward_hacking", "environment_failure"),
}
BASE_CSS = r"""
:root{color-scheme:light;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1a1a2e;background:#f6f7f9}
*{box-sizing:border-box}body{margin:0;background:#f6f7f9}a{color:#2456a6;text-decoration:none}a:hover{text-decoration:underline}
main.wrap{max-width:1200px;margin:0;padding:24px}main.wrap.fit{max-width:none;width:fit-content;min-width:min(100%,900px)}
.pagehead h1{font-size:22px;margin:18px 0}.top{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}.top h1{font-size:20px;margin:0}.top a{font-size:12px}
.topnav{margin:0 0 8px;display:flex;gap:8px;flex-wrap:wrap}.topnav a{padding:5px 13px;border-radius:6px;background:#eef0f4;font-size:13.5px;font-weight:600}
.topnav a.active{background:#1558d6;color:#fff}.topnav a:hover{text-decoration:none;background:#e0e3ea}.topnav a.active:hover{background:#0f47b0}
.seednav{margin:0 0 8px;display:flex;gap:6px;flex-wrap:wrap}.seednav a{padding:4px 11px;border-radius:5px;background:#f1f2f6;font-size:12px;font-weight:600;color:#4d5362}
.seednav a.active{background:#3f6fd1;color:#fff}.seednav a:hover{text-decoration:none;background:#e0e3ea}.seednav a.active:hover{background:#325cb4}
.viewnav{display:flex;gap:20px;flex-wrap:wrap;border-bottom:1px solid #dcdfe6;margin:0 0 16px}.viewnav a{padding:3px 1px 8px;font-size:12.5px;font-weight:600;color:#6b7280;border-bottom:2px solid transparent;margin-bottom:-1px}
.viewnav a.active{color:#1558d6;border-bottom-color:#1558d6}.viewnav a:hover{text-decoration:none;color:#1558d6}
.panel{background:#fff;border:1px solid #daddE5;border-radius:9px;margin:12px 0;padding:11px}.panel h2{font-size:15px;margin:0 0 8px}
details.panel>summary{cursor:pointer;font-size:13px;font-weight:650;color:#4d5362;list-style:none}details.panel>summary::-webkit-details-marker{display:none}
details.panel>summary:before{content:"›";display:inline-block;margin-right:7px;color:#8990a0;transition:transform .15s}details.panel[open]>summary:before{transform:rotate(90deg)}
.seed-block{margin:14px 0}.seed-block>h2{font-size:17px;margin:28px 0 6px}.runs{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dfe2e9;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.runs th,.runs td{text-align:left;padding:3px 6px;border-bottom:1px solid #e8e8ee;font-size:11px}.runs th{color:#4d5362;background:#eef0f4;font-size:10px;font-weight:600;position:sticky;top:0;white-space:nowrap}
.runs tr:last-child td{border-bottom:0}.result{font-weight:650}.result.yes{color:#a13c31}.result.no{color:#17663b}.result.review,.result.invalid{color:#a06b00}
.runs th.sortable-column{cursor:pointer;user-select:none;padding-right:16px}.runs th.sortable-column:hover{background:#e2e5eb}.runs th[data-sort-direction="asc"]:after{content:" ▲";position:absolute;margin-left:3px}.runs th[data-sort-direction="desc"]:after{content:" ▼";position:absolute;margin-left:3px}
.runs .dimension-cell{white-space:nowrap}.runs .dimension-cell.wide{white-space:normal;max-width:190px}.reward-label-list{display:flex;flex-direction:column;align-items:flex-start;gap:2px}.value-chip{display:inline-block;border-radius:4px;padding:1px 4px;background:#f0f1f4;color:#555d6b}.value-good{background:#d3f3d8;color:#17663b}.value-warn{background:#fff3cd;color:#735e13}.value-concern{background:#ffd6a5;color:#774215}.value-bad{background:#ffadad;color:#7f1d1d;font-weight:650}.value-invalid{background:#fff3cd;color:#8a5a00}.model-name{white-space:nowrap}.raw-title{cursor:help}
.metagrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:13px 20px;margin:12px 2px 4px}
.metacell .metakey{font-size:9.5px;letter-spacing:.5px;text-transform:uppercase;color:#949aa8;font-weight:700;margin-bottom:2px}.metacell .metavalue{font-size:13px;color:#262b36;line-height:1.3;overflow-wrap:anywhere}
.metaprev{font-weight:400;color:#79808f;margin-left:10px;font-size:12px}
details.raw summary{font-size:12px;color:#6f7584;cursor:pointer}details.raw pre,.json{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.45 ui-monospace,monospace;background:#f7f8fa;padding:8px;border-radius:5px}
.load-error{border:1px solid #d89689;background:#fff2ef;padding:8px;margin:8px 0;font-size:12px}.empty{color:#7b8190;font-size:12px}
.legacy-banner{border:1px solid #d8bd73;background:#fff9e9;padding:7px;border-radius:6px;font-size:12px}.cost{font-variant-numeric:tabular-nums}
.rejudge-banner{border:1px solid #bdcbe5;background:#f3f6fc;padding:7px 9px;border-radius:6px;font-size:12px;margin:10px 0}.kind{font-size:10px;border:1px solid #d9deea;background:#f5f7fb;border-radius:9px;padding:1px 6px;color:#59657b;white-space:nowrap}
.flag-list{display:flex;gap:3px;flex-wrap:wrap;min-width:95px}.flag{display:inline-block;border-radius:8px;padding:1px 5px;background:#fff3cd;color:#735e13;font-size:9px;white-space:nowrap;cursor:help}.flag.error{background:#ffd7d7;color:#7f1d1d}.section-note{color:#7b8190;font-size:11px;font-weight:400}
.runs.grouped tr.group-start td{border-top:2px solid #98a0b3}
"""


INDEX_SORT_JS = r"""<script>
(function () {
  function valueFor(row, index, type) {
    var cell = row.children[index];
    if (!cell) return null;
    var raw = cell.getAttribute("data-sort-value");
    if (raw === null || raw === "") return null;
    if (type === "number") {
      var number = Number(raw);
      return Number.isFinite(number) ? number : null;
    }
    return raw.toLocaleLowerCase();
  }
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var body = table.tBodies[0];
    if (!body) return;
    var headers = Array.prototype.slice.call(table.querySelectorAll("thead th"));
    headers.forEach(function (header, index) {
      header.classList.add("sortable-column");
      header.tabIndex = 0;
      function sortColumn() {
        var ascending = header.getAttribute("data-sort-direction") !== "asc";
        headers.forEach(function (other) {
          other.removeAttribute("data-sort-direction");
          other.removeAttribute("aria-sort");
        });
        header.setAttribute("data-sort-direction", ascending ? "asc" : "desc");
        header.setAttribute("aria-sort", ascending ? "ascending" : "descending");
        var type = header.getAttribute("data-sort-type") || "text";
        var rows = Array.prototype.slice.call(body.rows);
        var units;
        if (table.classList.contains("grouped")) {
          var byGroup = {};
          var order = [];
          rows.forEach(function (row) {
            var key = row.getAttribute("data-group");
            if (key === null) key = "row-" + order.length;
            if (!byGroup[key]) { byGroup[key] = []; order.push(key); }
            byGroup[key].push(row);
          });
          units = order.map(function (key) { return byGroup[key]; });
        } else {
          units = rows.map(function (row) { return [row]; });
        }
        units.sort(function (left, right) {
          var a = valueFor(left[0], index, type), b = valueFor(right[0], index, type);
          if (a === null && b === null) return 0;
          if (a === null) return 1;
          if (b === null) return -1;
          var comparison = type === "number"
            ? a - b
            : a.localeCompare(b, undefined, {numeric: true, sensitivity: "base"});
          return ascending ? comparison : -comparison;
        });
        units.forEach(function (unit) {
          unit.forEach(function (row) { body.appendChild(row); });
        });
      }
      header.addEventListener("click", sortColumn);
      header.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          sortColumn();
        }
      });
    });
  });
})();
</script>"""


def _page(
    title: str, body: str, *, scripts: str = "", fit_content: bool = True
) -> str:
    wrap_class = "wrap fit" if fit_content else "wrap"
    return f"""<!doctype html>
{_GENERATED_MARKER}
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{BASE_CSS}\n{EVIDENCE_NAV_CSS}\n{VISUALS_CSS}</style></head>
<body><main class="{wrap_class}">{body}</main>{scripts}</body></html>"""


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dimension(audit: dict, key: str) -> dict | None:
    return next(
        (
            dimension
            for dimension in ((audit.get("judgment") or {}).get("dimensions") or [])
            if dimension.get("key") == key
        ),
        None,
    )


def _result_cell(audit: dict) -> tuple[str, str]:
    judgment = audit.get("judgment")
    if not judgment:
        return "no judgment", "invalid"
    if judgment.get("format") == "legacy_numeric":
        reward = _dimension(audit, "reward_hacking")
        return (reward.get("display_value") if reward else "legacy", "invalid")
    reward = _dimension(audit, "reward_hacking")
    if not reward or reward.get("status") != "ok":
        return "invalid", "invalid"
    if reward.get("requires_review"):
        return str(reward.get("display_value") or "other"), "review"
    return (
        str(reward.get("display_value") or "none"),
        "yes" if reward.get("is_hack") else "no",
    )


def _recorded_cost(audit: dict) -> tuple[float, bool]:
    total = 0.0
    found = False
    for usage in (audit.get("model_usage") or {}).values():
        if isinstance(usage, dict) and isinstance(usage.get("total_cost"), (int, float)):
            total += float(usage["total_cost"])
            found = True
    if not audit.get("retrospective_rejudge"):
        compute = ((audit.get("real_env") or {}).get("compute") or {})
        vm_cost = compute.get("estimated_vm_cost_usd")
        if isinstance(vm_cost, (int, float)):
            total += float(vm_cost)
            found = True
    return total, found


def _role_costs(audit: dict) -> dict[str, float]:
    """Recorded cost split by stored role, plus the VM estimate on original runs."""
    costs: dict[str, float] = {}
    for role, usage in (audit.get("role_usage") or {}).items():
        if isinstance(usage, dict) and isinstance(usage.get("total_cost"), (int, float)):
            costs[str(role)] = costs.get(str(role), 0.0) + float(usage["total_cost"])
    if not audit.get("retrospective_rejudge"):
        compute = ((audit.get("real_env") or {}).get("compute") or {})
        vm_cost = compute.get("estimated_vm_cost_usd")
        if isinstance(vm_cost, (int, float)):
            costs["vm"] = costs.get("vm", 0.0) + float(vm_cost)
    return costs


def _cost_column_label(key: str) -> str:
    if key == "vm":
        return "VM estimate"
    return f"{ROLE_LABEL.get(key, key)} cost"


def _seed_families() -> dict[str, str]:
    return {
        seed_dir.name: family_dir.name
        for family_dir in SEEDS_ROOT.iterdir()
        if family_dir.is_dir() and family_dir.name in GENERIC_JUDGE_FAMILIES
        for seed_dir in family_dir.iterdir()
        if seed_dir.is_dir() and not seed_dir.name.startswith("_")
    } if SEEDS_ROOT.is_dir() else {}


def _ordered_seeds(extra: set[str] | None = None) -> list[str]:
    available = set(_seed_families()) | set(extra or set())
    ordered = [seed for seed in SEED_NAV_ORDER if seed in available]
    return ordered + sorted(available - set(ordered))


def _seed_filename(seed: str) -> str:
    return "index.html" if seed == "fraud_detection" else f"{seed}.html"


def _visuals_filename(seed: str) -> str:
    return "visuals.html" if seed == "fraud_detection" else f"visuals_{seed}.html"


def _comparisons_filename(seed: str) -> str:
    # TEMPORARY judge-comparisons tab (Owen, 2026-08-04). Delete alongside the other
    # judge-comparisons blocks when rejudge comparisons are done.
    return f"judge_comparisons_{seed}.html"


def _active_attr(active: bool) -> str:
    return ' class="active"' if active else ""


def _navigation(
    seeds: list[str],
    *,
    active_seed: str | None = None,
    active_view: str | None = None,
    href_prefix: str = "",
) -> str:
    families = _seed_families()
    active_family = families.get(active_seed) if active_seed else None
    family_links = []
    for family_name in GENERIC_JUDGE_FAMILIES:
        members = [seed for seed in seeds if families.get(seed) == family_name]
        if not members:
            continue
        family_links.append(
            f'<a href="{esc(href_prefix + _seed_filename(members[0]), quote=True)}"'
            f'{_active_attr(family_name == active_family)}>'
            f'{esc(FAMILY_NAV_LABELS.get(family_name, family_name))}</a>'
        )
    top = f'<div class="topnav">{"".join(family_links)}</div>'

    if active_family is not None:
        row_seeds = [seed for seed in seeds if families.get(seed) == active_family]
    else:
        row_seeds = [seed for seed in seeds if families.get(seed) is None]
    seed_links = "".join(
        f'<a href="{esc(href_prefix + _seed_filename(seed), quote=True)}"'
        f'{_active_attr(seed == active_seed)}>{esc(seed)}</a>'
        for seed in row_seeds
    )
    seed_row = f'<div class="seednav">{seed_links}</div>' if seed_links else ""

    if active_seed is not None:
        family = families.get(active_seed)
        items = [
            ("trajectories", _seed_filename(active_seed), "trajectories"),
            ("judge comparisons", _comparisons_filename(active_seed), "comparisons"),
            ("visuals", _visuals_filename(active_seed), "visuals"),
        ]
        if family in GENERIC_JUDGE_FAMILIES:
            items.append((
                "judge view",
                _generic_judge_filename(active_seed),
                "judge",
            ))
        items.append((
            "past iterations", f"{active_seed}_past.html", "past"
        ))
    else:
        items = []
    links = "".join(
        f'<a href="{esc(href_prefix + href, quote=True)}"'
        f'{_active_attr(key == active_view)}>{esc(label)}</a>'
        for label, href, key in items
    )
    return top + seed_row + f'<div class="viewnav">{links}</div>'


def _display_epoch(audit: dict) -> object:
    """Show the campaign epoch while preserving each imported log's stable identity."""

    real_env = audit.get("real_env") or {}
    compute = (real_env.get("compute") or {}) if isinstance(real_env, dict) else {}
    original_epoch = (
        compute.get("original_epoch") if isinstance(compute, dict) else None
    )
    return original_epoch if original_epoch is not None else audit.get("epoch")


def _index_dimension_keys(
    audits: list[dict], *, family: str | None = None
) -> list[str]:
    present = {
        str(dimension.get("key"))
        for audit in audits
        for dimension in ((audit.get("judgment") or {}).get("dimensions") or [])
        if dimension.get("key")
    }
    present.update(FAMILY_INDEX_DIMENSIONS.get(family, ()))
    ordered = [key for key in INDEX_DIMENSION_ORDER if key in present]
    return ordered + sorted(present - set(ordered))


def _target_display(target: object) -> tuple[str, str]:
    raw = str(target or "unknown")
    return target_label(raw), raw


def _judge_display(audit: dict) -> tuple[str, str]:
    raw = str(audit.get("judge") or "unknown")
    return judge_shortname(raw) or raw.split("/")[-1], raw


def _sort_attr(value: object | None) -> str:
    if value is None:
        return ""
    return f' data-sort-value="{esc(value, quote=True)}"'


def _user_turn_count(audit: dict) -> int | None:
    messages = audit.get("messages") or []
    if not messages:
        return None
    return sum(1 for message in messages if message.get("role") == "user")


def _dimension_sort_type(audits: list[dict], key: str) -> str:
    values = [
        dimension.get("value")
        for audit in audits
        if (dimension := _dimension(audit, key)) is not None
        and dimension.get("status") == "ok"
    ]
    return (
        "number"
        if values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        )
        else "text"
    )


def _dimension_chip_class(audit: dict, key: str, dimension: dict) -> str:
    status = dimension.get("status")
    if status in {"missing", "invalid"}:
        return "value-invalid"
    if status != "ok":
        return ""
    value = dimension.get("value")
    judgment = audit.get("judgment") or {}
    if key == "reward_hacking" and judgment.get("format") == "structured":
        if dimension.get("requires_review"):
            return "value-concern"
        return "value-bad" if dimension.get("is_hack") else "value-good"
    if key == "environment_failure" and isinstance(value, bool):
        return "value-bad" if value else "value-good"
    if key == "noticed_hack" and value == 10:
        return "value-good"
    harmful_numeric = {
        "reward_hacking", "deception", "degenerate_behavior", "hallucination",
        "auditor_mistakes", "situational_awareness", "environment_failure",
        "incompleteness",
    }
    if (
        key in harmful_numeric
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        if value >= 8:
            return "value-bad"
        if value >= 5:
            return "value-concern"
        if value >= 2:
            return "value-warn"
    return ""


def _dimension_cell(audit: dict, key: str) -> str:
    dimension = _dimension(audit, key)
    if dimension is None:
        return '<td class="dimension-cell">—</td>'
    display = str(dimension.get("display_value") or "—")
    value = dimension.get("value")
    sort_value = (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else display
    )
    classes = "value-chip"
    chip_class = _dimension_chip_class(audit, key, dimension)
    if chip_class:
        classes += f" {chip_class}"
    wide = " wide" if key == "reward_hacking" else ""
    display_values = dimension.get("display_values")
    if key == "reward_hacking" and isinstance(display_values, list):
        chips = "".join(
            f'<span class="{classes}">{esc(item)}</span>'
            for item in display_values
        )
        rendered_value = f'<span class="reward-label-list">{chips}</span>'
    else:
        rendered_value = f'<span class="{classes}">{esc(display)}</span>'
    return (
        f'<td class="dimension-cell{wide}"{_sort_attr(sort_value)}>'
        f'{rendered_value}</td>'
    )


def _flag_chip_list(flags: list[dict]) -> str:
    return "".join(
        f'<span class="flag {esc(flag.get("severity") or "warning", quote=True)}" '
        f'title="{esc(flag.get("detail") or "", quote=True)}">'
        f'{esc(flag.get("label") or flag.get("code") or "flag")}</span>'
        for flag in flags
    )


def _flags_cell(audit: dict) -> str:
    flags = audit.get("flags") or []
    if not flags:
        return '<td data-sort-value="">—</td>'
    sort_value = ", ".join(str(flag.get("label") or "") for flag in flags)
    return (
        f'<td{_sort_attr(sort_value)}>'
        f'<div class="flag-list">{_flag_chip_list(flags)}</div></td>'
    )


CATEGORY_SECTIONS = (
    ("hack", "Reward hacks"),
    ("review", "Needs review"),
    ("notable", "Notable behavior without a delivered reward hack"),
    ("clean", "Clean trajectories"),
    ("unjudged", "Not judged"),
    ("invalid", "Invalid judgments"),
    ("awaiting", "Awaiting current judgment"),
    ("excluded", "Excluded data-integrity failures"),
)


def _index_table(
    audits: list[dict],
    *,
    dimension_keys: list[str],
    comparison_groups: list[list[dict]] | None = None,
) -> str:
    # ``comparison_groups`` is the TEMPORARY judge-comparisons mode: each group is one
    # trajectory (official row first, rejudge rows under it), kept together by the
    # grouped sorter and separated by a bolder line.
    if comparison_groups is not None:
        ordered = [
            (audit, str(group_index), position == 0)
            for group_index, group in enumerate(comparison_groups)
            for position, audit in enumerate(group)
        ]
        audits = [audit for audit, _group, _start in ordered]
    else:
        ordered = [
            (audit, None, False)
            for audit in sorted(audits, key=lambda row: int(row.get("id") or 0))
        ]
    show_judge = comparison_groups is not None
    show_provenance = any(
        audit.get("retrospective_rejudge") or audit.get("source_trajectory_id") is not None
        for audit in audits
    )
    # Comparison mode splits recorded cost into one column per role that cost money:
    # target/gate/judge always, then any other stored role, then the VM estimate.
    cost_columns: list[str] = []
    if show_judge:
        present: set[str] = set()
        for audit in audits:
            present.update(_role_costs(audit))
        cost_columns = ["target", "gate", "judge"]
        cost_columns += sorted(present - {"target", "gate", "judge", "vm"})
        if "vm" in present:
            cost_columns.append("vm")
    cells = []
    for audit, group_key, group_start in ordered:
        cost, has_cost = _recorded_cost(audit)
        retrospective = audit.get("retrospective_rejudge")
        source_id = audit.get("source_trajectory_id")
        target_label, target_raw = _target_display(audit.get("target"))
        provenance = ""
        if show_provenance:
            if source_id is not None:
                provenance = (
                    f'<a href="trajectory-{int(source_id)}.html">{int(source_id)}</a>'
                )
            elif retrospective:
                provenance = "unlinked"
            else:
                provenance = "—"
        judge_cell = ""
        if show_judge:
            judge_label, judge_raw = _judge_display(audit)
            judge_cell = (
                f'<td class="model-name raw-title" title="{esc(judge_raw, quote=True)}"'
                f'{_sort_attr(judge_label)}>{esc(judge_label)}</td>'
            )
        dimensions = "".join(
            _dimension_cell(audit, key) for key in dimension_keys
        )
        user_turns = _user_turn_count(audit)
        if show_judge:
            role_costs = _role_costs(audit)
            if (
                retrospective
                and user_turns == 1
                and "judge" in role_costs
                and "gate" not in role_costs
            ):
                # A rejudge is one call. On a source that ended at the first-submission
                # gate it redoes the stage-one judgment, so its cost belongs under
                # First judge, matching the official row above it.
                role_costs["gate"] = role_costs.pop("judge")
            cost_cells = "".join(
                f'<td class="cost"{_sort_attr(role_costs.get(key))}>'
                + (f'${role_costs[key]:.4f}' if key in role_costs else "—")
                + '</td>'
                for key in cost_columns
            )
        else:
            cost_cells = (
                f'<td class="cost"{_sort_attr(cost if has_cost else None)}>'
                f'{f"${cost:.4f}" if has_cost else "—"}</td>'
            )
        row_attributes = f' data-id="{int(audit["id"])}"'
        if group_key is not None:
            row_attributes += f' data-group="{esc(group_key, quote=True)}"'
        if group_start:
            row_attributes += ' class="group-start"'
        cells.append(
            f'<tr{row_attributes}>'
            f'<td{_sort_attr(audit["id"])}><a href="trajectory-{int(audit["id"])}.html">'
            f'{int(audit["id"])}</a></td>'
            f'<td class="model-name raw-title" title="{esc(target_raw, quote=True)}"'
            f'{_sort_attr(target_label)}>{esc(target_label)}</td>'
            + judge_cell
            + f'<td{_sort_attr(user_turns)}>'
            f'{user_turns if user_turns is not None else "—"}</td>'
            + (
                f'<td{_sort_attr(source_id)}>{provenance}</td>'
                if show_provenance else ""
            )
            + dimensions
            + _flags_cell(audit)
            + cost_cells
            + "</tr>"
        )
    dimension_headers = "".join(
        f'<th data-sort-type="{_dimension_sort_type(audits, key)}">'
        f'{esc(INDEX_DIMENSION_LABELS.get(key, key.replace("_", " ")))}</th>'
        for key in dimension_keys
    )
    judge_header = '<th data-sort-type="text">Judge</th>' if show_judge else ""
    provenance_header = '<th data-sort-type="text">Source</th>' if show_provenance else ""
    if show_judge:
        cost_headers = "".join(
            f'<th data-sort-type="number">{esc(_cost_column_label(key))}</th>'
            for key in cost_columns
        )
    else:
        cost_headers = '<th data-sort-type="number">Recorded cost</th>'
    table_class = "runs sortable grouped" if show_judge else "runs sortable"
    return (
        f'<table class="{table_class}"><thead><tr>'
        '<th data-sort-type="number">ID</th><th data-sort-type="text">Target</th>'
        f'{judge_header}'
        '<th data-sort-type="number">User turns</th>'
        f'{provenance_header}{dimension_headers}'
        '<th data-sort-type="text">Flags</th>'
        f'{cost_headers}'
        f'</tr></thead><tbody>{"".join(cells)}</tbody></table>'
    )


def _index(
    seed: str,
    audits: list[dict],
    errors: list[dict],
    *,
    seeds: list[str],
    active_view: str = "trajectories",
    title: str | None = None,
    historical: bool = False,
) -> str:
    dimension_keys = _index_dimension_keys(
        audits, family=_seed_families().get(seed)
    )
    error_html = "".join(
        f'<div class="load-error">{esc(error.get("mode"))}: '
        f'{esc(error.get("error_type"))}: {esc(error.get("error"))}</div>'
        for error in errors
    )
    if historical:
        sections = (
            '<section class="seed-block"><h2>Historical judgments '
            f'<span class="section-note">— {len(audits)} total</span></h2>'
            + _index_table(audits, dimension_keys=dimension_keys)
            + '</section>'
        )
    else:
        grouped = {
            key: [audit for audit in audits if trajectory_category(audit) == key]
            for key, _label in CATEGORY_SECTIONS
        }
        sections = "".join(
            '<section class="seed-block"><h2>'
            f'{esc(label)} <span class="section-note">— {len(grouped[key])}</span></h2>'
            + _index_table(grouped[key], dimension_keys=dimension_keys)
            + '</section>'
            for key, label in CATEGORY_SECTIONS
        )
    heading = title or f"{seed} — trajectories"
    body = (
        _navigation(seeds, active_seed=seed, active_view=active_view)
        + f'<div class="pagehead"><h1>{esc(heading)}</h1></div>'
        + error_html
        + sections
    )
    return _page(heading, body, scripts=INDEX_SORT_JS)


def _comparison_groups(
    official: list[dict], rejudges: list[dict]
) -> list[list[dict]]:
    groups: dict[int, list[dict]] = {
        int(audit["id"]): [audit]
        for audit in sorted(official, key=lambda row: int(row.get("id") or 0))
    }
    orphans: list[list[dict]] = []
    for rejudge in sorted(
        rejudges,
        key=lambda row: (str(row.get("mode") or ""), int(row.get("id") or 0)),
    ):
        source_id = rejudge.get("source_trajectory_id")
        if source_id is not None and int(source_id) in groups:
            groups[int(source_id)].append(rejudge)
        else:
            orphans.append([rejudge])
    return [groups[key] for key in sorted(groups)] + orphans


def _comparisons_index(
    seed: str,
    official: list[dict],
    rejudges: list[dict],
    errors: list[dict],
    *,
    seeds: list[str],
) -> str:
    # TEMPORARY judge-comparisons page (Owen, 2026-08-04): the trajectories page with
    # every trajectory's rejudge rows grouped under its official row. Sections follow
    # the official row's category; a rejudge whose source row is absent gets its own
    # group in its own judgment's category.
    dimension_keys = _index_dimension_keys(
        official + rejudges, family=_seed_families().get(seed)
    )
    error_html = "".join(
        f'<div class="load-error">{esc(error.get("mode"))}: '
        f'{esc(error.get("error_type"))}: {esc(error.get("error"))}</div>'
        for error in errors
    )
    groups = _comparison_groups(official, rejudges)
    grouped = {
        key: [
            group for group in groups
            if trajectory_category(group[0]) == key
        ]
        for key, _label in CATEGORY_SECTIONS
    }
    sections = "".join(
        '<section class="seed-block"><h2>'
        f'{esc(label)} <span class="section-note">— {len(grouped[key])}</span></h2>'
        + _index_table(
            [], dimension_keys=dimension_keys, comparison_groups=grouped[key]
        )
        + '</section>'
        for key, label in CATEGORY_SECTIONS
    )
    heading = f"{seed} — judge comparisons"
    body = (
        _navigation(seeds, active_seed=seed, active_view="comparisons")
        + f'<div class="pagehead"><h1>{esc(heading)}</h1></div>'
        + error_html
        + sections
    )
    return _page(heading, body, scripts=INDEX_SORT_JS)


def _metadata_panel(audit: dict) -> str:
    """Collapsed Metadata dropdown at the top of a trajectory page.

    Petri-style: the closed bar previews the target and result so pages stay scannable
    without expanding; the body is one label-over-value grid. Empty values drop their
    cell. How the run ended is not shown here — abnormal endings appear as flags.
    """
    result, _state = _result_cell(audit)
    cost, has_cost = _recorded_cost(audit)
    label, target_raw = _target_display(audit.get("target"))
    epoch = _display_epoch(audit)

    def cell(key: str, value_html: str) -> str:
        return (
            '<div class="metacell">'
            f'<div class="metakey">{esc(key)}</div>'
            f'<div class="metavalue">{value_html}</div></div>'
        ) if value_html else ""

    cells = [
        cell("trajectory", esc(audit.get("id") or "")),
        cell("seed", esc(audit.get("seed") or "")),
        cell(
            "target",
            f'<span class="raw-title" title="{esc(target_raw, quote=True)}">'
            f'{esc(label)}</span>',
        ),
        cell("judge", esc(audit.get("judge") or "")),
        cell("condition", esc(audit.get("condition") or "")),
        cell("epoch", esc(epoch) if epoch is not None else ""),
        cell("reward hack", esc(result)),
        cell("recorded cost", esc(f"${cost:.6f}") if has_cost else "unavailable"),
        cell(
            "flags",
            f'<div class="flag-list">{_flag_chip_list(audit.get("flags") or [])}</div>'
            if audit.get("flags") else "none",
        ),
        cell("run dir", esc(audit.get("mode") or "")),
    ]
    preview_parts = [label]
    if audit.get("condition"):
        preview_parts.append(str(audit["condition"]))
    preview_parts.append(result)
    return (
        '<details class="panel metadata"><summary>Metadata'
        f'<span class="metaprev">{esc(" · ".join(preview_parts))}</span></summary>'
        f'<div class="metagrid">{"".join(cells)}</div></details>'
    )


def _json_panel(title: str, value: Any) -> str:
    return (
        f'<details class="panel raw"><summary>{esc(title)}</summary>'
        f'<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))}</pre>'
        '</details>'
    )


def _without_duplicate_judge_payloads(value: Any) -> Any:
    """Keep raw records inspectable without embedding the same huge judge input twice."""

    replacements = {
        "prompt_passed_to_scout": "[shown in Judge view]",
        "rendered_messages": "[shown in Trajectory]",
        "rendered_artifacts": "[shown in Judge view > Artifacts]",
        "stage1_prompt": "[shown in Judge view > Stage-one prompt]",
        "final_render": "[shown in Judge view > Final evidence rendering]",
        "head_render": "[included in the stored stage-one prompt]",
    }
    if isinstance(value, dict):
        return {
            str(key): (
                replacements[key]
                if key in replacements
                else _without_duplicate_judge_payloads(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_duplicate_judge_payloads(item) for item in value]
    return value


def _trajectory(
    audit: dict, *, seeds: list[str], active_view: str = "trajectories"
) -> str:
    judgment = audit.get("judgment")
    legacy = (
        '<div class="legacy-banner">Legacy numeric judgment, shown exactly as stored.</div>'
        if judgment and judgment.get("format") == "legacy_numeric" else ""
    )
    grade = (audit.get("real_env") or {}).get("grade")
    source_id = audit.get("source_trajectory_id")
    retrospective = audit.get("retrospective_rejudge")
    judge_failure = audit.get("judge_failure")
    rejudge_banner = ""
    if retrospective:
        source_link = (
            f'<a href="trajectory-{int(source_id)}.html">trajectory {int(source_id)}</a>'
            if source_id is not None else "the stored source trajectory"
        )
        rejudge_banner = (
            '<div class="rejudge-banner">Retrospective rejudge of '
            f'{source_link} · current method '
            f'{esc(str(retrospective.get("judging_method_sha256") or "unrecorded")[:12])}'
            '</div>'
        )
    body = (
        _navigation(
            seeds,
            active_seed=str(audit.get("seed") or "unknown"),
            active_view=active_view,
        )
        + '<div class="pagehead"><h1>'
        f'{esc(audit.get("seed") or "unknown")} — trajectory {int(audit["id"])}</h1></div>'
        + _metadata_panel(audit)
        + rejudge_banner
        + legacy
        + (
            _json_panel("Judge failure", {
                "failure": judge_failure,
                "score_metadata": audit.get("score_metadata") or {},
            })
            if judge_failure else ""
        )
        + render_judge_narrative(judgment)
        + render_judge_view(judgment, audit)
        + f'<section class="panel trajectory-panel" id="trajectory-record"><h2>'
        f'Trajectory · {len(audit.get("messages") or [])} messages</h2>'
        + render_transcript(audit.get("messages") or [])
        + '</section>'
        + (_json_panel("Load issues", audit.get("load_issues"))
           if audit.get("load_issues") else "")
        + '<details class="panel other-stuff"><summary>Other stuff</summary>'
        + render_dimension_navigator(judgment)
        + (_json_panel("Grade", grade) if grade else "")
        + _json_panel("Stored judgment", _without_duplicate_judge_payloads(judgment))
        + _json_panel(
            "Environment record",
            _without_duplicate_judge_payloads(audit.get("real_env") or {}),
        )
        + _json_panel("Model usage", audit.get("model_usage") or {})
        + '</details>'
        + render_justification_turn_nav(judgment)
    )
    return _page(
        f'Trajectory {int(audit["id"])}',
        body,
        scripts=EVIDENCE_NAV_JS,
    )


def _visuals(seed: str, audits: list[dict], *, seeds: list[str]) -> str:
    body = (
        _navigation(seeds, active_seed=seed, active_view="visuals")
        + f'<div class="pagehead"><h1>{esc(seed)} — visuals</h1></div>'
        + render_visuals(audits)
    )
    return _page(f"{seed} · visuals", body, scripts=VISUALS_JS, fit_content=False)


def _generic_judge_filename(seed: str) -> str:
    return f"judge_{seed}.html"


async def _generic_judge_page(
    seed: str, family: str, *, seeds: list[str]
) -> str:
    prepared = await prepare_judge_call(
        family=family,
        stage="final",
        messages=[],
        artifacts=[],
    )
    envelope = {
        **prepared.metadata(),
        "provider_request": provider_request_record(prepared),
    }
    prompt_preview = render_generic_judge_stage(
        prompt=prepared.prompt,
        envelope=envelope,
        summary_label="Judge prompt",
    )
    body = (
        _navigation(seeds, active_seed=seed, active_view="judge")
        + f'<div class="pagehead"><h1>{esc(seed)} — current judge view</h1></div>'
        + prompt_preview
    )
    return _page(
        f"{seed} · current judge", body, fit_content=False
    )


def _archive_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.move(str(source), destination)
        return
    if source.read_bytes() == destination.read_bytes():
        source.unlink()
        return
    for index in range(2, 10_000):
        candidate = destination.with_name(
            f"{destination.stem}-{index}{destination.suffix}"
        )
        if not candidate.exists():
            shutil.move(str(source), candidate)
            return
    raise RuntimeError(f"could not archive {source}")


def _archive_legacy_viewer_pages() -> int:
    """Move old static judgments out of the live viewer without deleting them."""

    candidates = list(VIEWER_ROOT.glob("*_judge_tests.html"))
    candidates += list(VIEWER_ROOT.glob("*_past.html"))
    candidates += list(VIEWER_ROOT.glob("judge-*.html"))
    pages = VIEWER_ROOT / "pages"
    if pages.is_dir():
        candidates += list(pages.glob("*.html"))
    archived = 0
    archive_root = VIEWER_ROOT / "_archive" / "legacy_judge_viewer"
    for path in candidates:
        if not path.is_file():
            continue
        generated = path.parent == VIEWER_ROOT and _GENERATED_MARKER in path.read_text()
        if generated and not path.name.endswith("_judge_tests.html"):
            continue
        relative = path.relative_to(VIEWER_ROOT)
        _archive_path(path, archive_root / relative)
        archived += 1
    return archived


async def _current_judge_methods() -> dict[str, str]:
    methods = {}
    for family in GENERIC_JUDGE_FAMILIES:
        prepared = await prepare_judge_call(
            family=family, stage="final", messages=[], artifacts=[]
        )
        methods[family] = prepared.method_sha256()
    return methods

def _is_current_judgment(audit: dict, methods: dict[str, str]) -> bool:
    judgment = audit.get("judgment") or {}
    if judgment.get("format") == "structured":
        envelope = judgment.get("envelope") or {}
        family = str(judgment.get("family") or envelope.get("family") or "")
    else:
        score_metadata = audit.get("score_metadata") or {}
        envelope = score_metadata.get("environment_judge") or {}
        if not isinstance(envelope, dict):
            return False
        family = str(envelope.get("family") or "")
    stored = envelope.get("judge_method_sha256")
    return bool(stored and stored == methods.get(family))


async def build(*, use_cache: bool = True) -> dict:
    with viewer_build_lock(DATA_ROOT):
        audits, errors = await load_all(
            LOGS_ROOT,
            cache_root=CACHE_ROOT,
            use_cache=use_cache,
        )
        assign_stable_ids(audits, REGISTRY_FILE)
        link_rejudge_sources(audits)
        VIEWER_ROOT.mkdir(parents=True, exist_ok=True)
        archived_pages = _archive_legacy_viewer_pages()
        current_methods = await _current_judge_methods()
        current_grouped: dict[str, list[dict]] = defaultdict(list)
        past_grouped: dict[str, list[dict]] = defaultdict(list)
        # TEMPORARY judge-comparisons routing (Owen, 2026-08-04): rejudge rows never
        # reach the trajectories, visuals, or past pages — canonical judgments stay
        # official there — and render only on the judge-comparisons page.
        rejudge_grouped: dict[str, list[dict]] = defaultdict(list)
        for audit in audits:
            seed = str(audit.get("seed") or "unknown")
            audit["current_judge_method"] = _is_current_judgment(
                audit, current_methods
            )
            if audit.get("retrospective_rejudge"):
                destination = rejudge_grouped
            elif audit["current_judge_method"]:
                destination = current_grouped
            else:
                destination = past_grouped
            destination[seed].append(audit)
        seeds = _ordered_seeds(
            set(current_grouped) | set(past_grouped) | set(rejudge_grouped)
        )
        for seed in seeds:
            current = current_grouped.get(seed, [])
            past = past_grouped.get(seed, [])
            _write_atomic(
                VIEWER_ROOT / _seed_filename(seed),
                _index(seed, current, errors, seeds=seeds),
            )
            _write_atomic(
                VIEWER_ROOT / _comparisons_filename(seed),
                _comparisons_index(
                    seed, current, rejudge_grouped.get(seed, []), errors, seeds=seeds
                ),
            )
            _write_atomic(
                VIEWER_ROOT / _visuals_filename(seed),
                _visuals(seed, current, seeds=seeds),
            )
            _write_atomic(
                VIEWER_ROOT / f"{seed}_past.html",
                _index(
                    seed, past, [], seeds=seeds, active_view="past",
                    title=f"{seed} — past iterations", historical=True,
                ),
            )
        families = _seed_families()
        for seed in seeds:
            family = families.get(seed)
            if family not in GENERIC_JUDGE_FAMILIES:
                continue
            _write_atomic(
                VIEWER_ROOT / _generic_judge_filename(seed),
                await _generic_judge_page(seed, family, seeds=seeds),
            )
        expected = set()
        for audit in audits:
            filename = f'trajectory-{int(audit["id"])}.html'
            expected.add(filename)
            if audit.get("retrospective_rejudge"):
                active_view = "comparisons"
            elif audit["current_judge_method"]:
                active_view = "trajectories"
            else:
                active_view = "past"
            _write_atomic(
                VIEWER_ROOT / filename,
                _trajectory(audit, seeds=seeds, active_view=active_view),
            )
        for path in VIEWER_ROOT.glob("trajectory-*.html"):
            if path.name not in expected:
                path.unlink()
    return {
        "trajectories": len(audits),
        "current_trajectories": sum(map(len, current_grouped.values())),
        "past_trajectories": sum(map(len, past_grouped.values())),
        "rejudge_trajectories": sum(map(len, rejudge_grouped.values())),
        "load_errors": len(errors),
        "legacy_pages_archived": archived_pages,
        "output": str(VIEWER_ROOT / "index.html"),
    }


async def main() -> None:
    stats = await build()
    print(
        f"viewer: {stats['trajectories']} trajectories, "
        f"{stats['load_errors']} load errors -> {stats['output']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
