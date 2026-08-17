"""Build the standalone static viewer for real-environment trajectories.

This endpoint is free. It reads this project's Inspect logs and writes HTML under
``mats-local/environments/viewer``. It has no Petri imports or runtime fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
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
    render_explanation_turn_nav,
    render_jump_to_new_task,
    render_judge_narrative,
    render_judge_view,
    render_transcript,
)
from env_viewer_continuations import (  # noqa: E402
    CONTINUATION_DESTINATIONS,
    CONTINUATION_DIRECTIONS,
    DEFAULT_CONTINUATION_DIRECTION,
    OTHER_CONTINUATION_DIRECTION,
    continuation_condition_context,
    continuation_destination,
    continuation_destination_label,
    continuation_direction,
    continuation_groups,
    continuation_of,
    continuation_prefix_label,
    continuation_prefix_rate_data,
    is_activity_log_context_continuation,
    prefix_of,
    prefix_source_trajectory_id,
    pressure_of,
)
from cost_tracking import CONTEXT_WINDOWS, canonical_slug  # noqa: E402
from env_viewer_load import (  # noqa: E402
    assign_stable_ids,
    link_rejudge_sources,
    load_all,
    normalize_messages,
    promote_rejudge_judgments,
    viewer_build_lock,
)
from env_viewer_cache import trajectory_key  # noqa: E402
from env_viewer_store import (  # noqa: E402
    STORE_FILENAME,
    ViewerAuditStore,
)
from env_viewer_visuals import (  # noqa: E402
    ROLE_LABEL,
    VISUALS_CSS,
    VISUALS_JS,
    render_continuation_visuals,
    render_visuals,
    agent_label,
    trajectory_category,
)
from env_viewer_turns import experiment_user_turn_count  # noqa: E402
from environment_judge import (  # noqa: E402
    prepare_judge_call,
    provider_request_record,
)
from judge_selection import judge_shortname  # noqa: E402
from judgment_semantics import (  # noqa: E402
    FAMILY_INDEX_DIMENSIONS,
    INDEX_DIMENSION_LABELS,
    INDEX_DIMENSION_ORDER,
    REWARD_HACKING_DISPLAY_LABELS,
)
from old_runs import (  # noqa: E402
    old_prefix_files,
    old_run_names,
    old_trajectory_keys,
    promoted_rejudge_run_names,
)
from project_paths import (  # noqa: E402
    CONTINUATION_PREFIXES_ROOT,
    DATA_ROOT,
    LOGS_ROOT,
    SEEDS_ROOT,
    VIEWER_ROOT,
)
from real_integrity import mark_ineligible_prefix_source  # noqa: E402


REGISTRY_FILE = DATA_ROOT / "trajectory_ids.json"
CACHE_ROOT = DATA_ROOT / "viewer_cache"
CONTINUATIONS_FILENAME = "continuations.html"
PREFIXES_FILENAME = "prefixes.html"
MULTI_AGENT_FILENAME = "multi_agent.html"
DEFAULT_PREFIX_TYPE = "natural_questions"
ACTIVITY_LOG_PREFIX_TYPE = "activity_log_context"
PREFIX_SELECTIONS_FORMAT = "environments-prefix-selections-v1"
HARNESS_NAV_MODES = ("simple", "subscription")
CURRENT_PRESSURE_NAV_MODES = ("low", "high")
ALL_PRESSURE_VIEWS = (*CURRENT_PRESSURE_NAV_MODES, "unspecified")
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
FAMILY_NAV_LABELS = {
    "ml_training_data_misuse": "ML",
    "p_hacking": "p-hacking",
}
HARNESS_NAV_LABELS = {
    "simple": "Simple harness",
    "subscription": "Subscription harness",
}
OLD_VIEW_NAMES = frozenset({"comparisons", "past", "prefixes_past"})
HIDDEN_INDEX_DIMENSIONS = frozenset({"situational_awareness"})
PRESSURE_NAV_LABELS = {
    "high": "High pressure",
    "low": "Default",
    "unspecified": "Legacy / unspecified",
}
BASE_CSS = r"""
/* Shared viewer chrome copied from petri/viewer.py. Keep common pages visually and
   behaviorally identical; environment-only widgets add their rules below. */
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: #f6f7f9; color: #1a1a2e; }
.wrap { max-width: 1200px; margin: 0; padding: 24px; }
.wrap.fit { max-width: none; width: fit-content; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 28px; }
h3 { font-size: 14px; color: #3a3a52; }
details.sec { margin-top: 26px; border: 1px solid #d4d9e2; border-radius: 6px;
  background: #eef1f5; }
details.sec > summary { cursor: pointer; user-select: none; background: #e9ecf2;
  border-radius: 5px; padding: 6px 12px; list-style: none; }
details.sec[open] > summary { border-bottom: 1px solid #d4d9e2;
  border-radius: 5px 5px 0 0; }
details.sec > summary:hover { background: #dfe3ec; }
details.sub > summary { cursor: pointer; user-select: none; background: #f0f2f6;
  border: 1px solid #d4d9e2; border-radius: 6px; padding: 4px 10px; list-style: none; }
details.sub[open] > summary { border-radius: 6px 6px 0 0; }
details.sub > summary:hover { background: #e6e9f0; }
details.sec > summary::-webkit-details-marker,
details.sub > summary::-webkit-details-marker { display: none; }
details.sec > summary::marker, details.sub > summary::marker { content: none; }
summary h2, summary h3 { display: inline; margin: 0; }
details.sec > details.sub { margin: 8px 10px; }
details.sec > table { margin: 8px 10px; width: calc(100% - 20px);
  border: 1px solid #d4d9e2; }
details.sub > table { margin-top: 0; border: 1px solid #d4d9e2; border-top: none; }
a { color: #2456a6; text-decoration: none; } a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); }
th, td { padding: 4px 7px; border-bottom: 1px solid #e8e8ee; font-size: 12px; text-align: left; }
th { background: #eef0f4; position: sticky; top: 0; }
tr.cols th { font-size: 11px; font-weight: 600; }
table.sortable th, table.sortable td { padding: 2px 5px; font-size: 11px; }
table.sortable tr.cols th { font-size: 10px; }
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th:hover { background: #e2e6ee; }
.runs th[data-sort-direction="asc"]::after { content: " ▲"; font-size: 9px; color: #2456a6; }
.runs th[data-sort-direction="desc"]::after { content: " ▼"; font-size: 9px; color: #2456a6; }
.note { background: #fff; border-left: 4px solid #2456a6; padding: 12px 16px; margin: 10px 0;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); white-space: pre-wrap; font-size: 14px; line-height: 1.45; }
.note.explanation { border-left-color: #a62445; }
.note.hl { border-left-color: #6a994e; }
details.sec > .note { margin: 8px 10px; }
table.scoretop { width: auto; margin: 10px 0; }
details.sec.metadata > summary { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.metaprev { font-weight: 400; }
.metabody { margin: 8px 10px; background: #fff; border-radius: 5px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08); padding: 14px 16px; }
.mblock + .mblock { margin-top: 18px; padding-top: 16px; border-top: 1px solid #edeff3; }
.mblock-h { font-size: 10px; letter-spacing: .6px; text-transform: uppercase; color: #949aa8;
  font-weight: 700; margin: 0 0 9px; }
.metabody table.scoretop { margin: 0 0 2px; }
.metagrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 13px 20px; }
.metacell .k { display: block; margin-bottom: 2px; font-size: 9.5px; letter-spacing: .5px;
  text-transform: uppercase; color: #949aa8; font-weight: 700; }
.metacell .v { font-size: 13px; color: #262b36; line-height: 1.3; overflow-wrap: anywhere; }
.contextgraph { margin-top: 14px; padding-top: 12px; border-top: 1px solid #edeff3; }
.contextgraph svg { display: block; width: 100%; max-width: 820px; height: auto; }
.contextgraph-note { margin: 7px 0 0; font-size: 11px; color: #9a5b16; }
.window-nav, .scope-nav, .topnav { margin: 0 0 8px; display: flex; gap: 8px; flex-wrap: wrap; }
.window-nav a, .scope-nav a, .topnav a { padding: 5px 13px; border-radius: 6px; background: #eef0f4;
  font-size: 13.5px; font-weight: 600; }
.window-nav a.active, .scope-nav a.active, .topnav a.active { background: #1558d6; color: #fff; }
.window-nav a:hover, .scope-nav a:hover, .topnav a:hover { text-decoration: none; background: #e0e3ea; }
.window-nav a.active:hover, .scope-nav a.active:hover, .topnav a.active:hover { background: #0f47b0; }
.window-nav { margin-bottom: 12px; }
.window-nav a { font-size: 15px; }
.scope-nav { margin-bottom: 12px; }
.scope-nav a { font-size: 14px; }
.seednav, .contextnav, .viewnav, .subnav { display: flex; gap: 20px; flex-wrap: wrap;
  border-bottom: 1px solid #dcdfe6; }
.seednav, .contextnav { margin: 3px 0 4px; }
.viewnav, .subnav { margin: 0 0 16px; }
.seednav a, .contextnav a, .viewnav a, .subnav a { padding: 3px 1px 8px; font-size: 12.5px;
  font-weight: 600; color: #6b7280; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.seednav a.active, .contextnav a.active, .viewnav a.active, .subnav a.active {
  color: #1558d6; border-bottom-color: #1558d6; }
.seednav a:hover, .contextnav a:hover, .viewnav a:hover, .subnav a:hover {
  text-decoration: none; color: #1558d6; }
.continuation-destination-nav { margin: 12px 0 0; padding: 8px 10px 0;
  background: #e9edf5; border: 1px solid #d3d9e5; border-bottom: none;
  border-radius: 8px 8px 0 0; }
.continuation-destination-nav a { font-size: 13.5px; color: #454d5d; }
.continuation-prefix-nav { margin: 0 0 8px; padding: 8px 12px 0 20px;
  background: #fafbfd; border: 1px solid #d3d9e5; border-left: 4px solid #bdc9dc;
  border-radius: 0 0 8px 8px; }
.continuation-prefix-nav a { font-size: 11.5px; font-weight: 550; }
.pagehead { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 10px 0 14px; }
.pagehead h1 { margin: 0; }
.headbtn { padding: 5px 13px; border-radius: 6px; background: #eef0f4; font-size: 13.5px;
  font-weight: 600; white-space: nowrap; }
.headbtn:hover { text-decoration: none; background: #e0e3ea; }
.totop { position: fixed; right: 18px; bottom: 18px; width: 42px; height: 42px; border: none;
  border-radius: 50%; background: #1558d6; color: #fff; font-size: 1.25rem; cursor: pointer;
  box-shadow: 0 2px 10px rgba(0,0,0,.28); display: none; z-index: 50; }
.totop:hover { background: #0f47b0; }
.meta { color: #667; font-size: 13px; }
.all-count { color: #9ca0aa; font-weight: 400; }
.result { font-weight: 650; }.result.yes { color: #a13c31; }.result.no { color: #17663b; }
.result.review, .result.invalid { color: #a06b00; }
.runs .dimension-cell { white-space: nowrap; }.runs .dimension-cell.wide { white-space: normal; max-width: 190px; }
.reward-label-list { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; }
.value-chip { display: inline-block; border-radius: 4px; padding: 1px 4px; background: #f0f1f4; color: #555d6b; }
.value-good { background: #d3f3d8; color: #17663b; }.value-warn { background: #fff3cd; color: #735e13; }
.value-concern { background: #ffd6a5; color: #774215; }.value-bad { background: #ffadad; color: #7f1d1d; font-weight: 650; }
.value-invalid { background: #fff3cd; color: #8a5a00; }.model-name { white-space: nowrap; }.raw-title { cursor: help; }
.panel { background: #fff; border: 1px solid #daddE5; border-radius: 9px; margin: 12px 0; padding: 11px; }
.activity-log { margin: 0; padding: 12px; max-height: 680px; overflow: auto;
  white-space: pre-wrap; overflow-wrap: anywhere; background: #fff; font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.prompt-preview { white-space: pre-wrap; overflow-wrap: anywhere; background: #fff;
  border: 1px solid #d4d9e2; border-radius: 6px; padding: 12px; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
details.panel > summary { cursor: pointer; font-size: 13px; font-weight: 650; color: #4d5362; list-style: none; }
details.panel > summary::-webkit-details-marker { display: none; }
details.panel > summary::before { content: "›"; display: inline-block; margin-right: 7px; color: #8990a0; transition: transform .15s; }
details.panel[open] > summary::before { transform: rotate(90deg); }
details.raw summary { font-size: 12px; color: #6f7584; cursor: pointer; }
details.raw pre, .json { white-space: pre-wrap; overflow-wrap: anywhere; font: 11px/1.45 ui-monospace,monospace;
  background: #f7f8fa; padding: 8px; border-radius: 5px; }
.load-error { border: 1px solid #d89689; background: #fff2ef; padding: 8px; margin: 8px 0; font-size: 12px; }
.empty { color: #7b8190; font-size: 12px; }.legacy-banner { border: 1px solid #d8bd73; background: #fff9e9;
  padding: 7px; border-radius: 6px; font-size: 12px; }.cost { font-variant-numeric: tabular-nums; }
.cost-note { display: block; color: #9b4b00; font-size: 9px; font-variant-numeric: normal; white-space: nowrap; }
.rejudge-banner { border: 1px solid #bdcbe5; background: #f3f6fc; padding: 7px 9px; border-radius: 6px;
  font-size: 12px; margin: 10px 0; }.kind { font-size: 10px; border: 1px solid #d9deea; background: #f5f7fb;
  border-radius: 9px; padding: 1px 6px; color: #59657b; white-space: nowrap; }
.flag-list { display: flex; gap: 3px; flex-wrap: wrap; min-width: 95px; }.flag { display: inline-block;
  border-radius: 8px; padding: 1px 5px; background: #fff3cd; color: #735e13; font-size: 9px;
  white-space: nowrap; cursor: help; }.flag.error { background: #ffd7d7; color: #7f1d1d; }
.status-stack { min-width: 95px; }.status-heading { margin-bottom: 4px; }
.status-reasons { display: flex; gap: 3px; flex-wrap: wrap; }
.flag.status-main { border-radius: 5px; padding: 1px 6px; font-weight: 750;
  border: 1px solid #d9bf5f; background: #f8e7a5; }
.flag.status-main.error { border-color: #d78a8a; background: #f4bcbc; }
.runs tr.integrity-excluded > td { background: #f0f1f3; color: #7c828d; }
.runs tr.integrity-excluded > td a { color: #747d8e; }
.runs tr.integrity-excluded > td .value-chip { background: #dfe1e5; color: #747982; }
.runs.grouped tr.group-start td { border-top: 2px solid #98a0b3; }
.runs tr.prefix-selected > td { background: #edf5ff; }
.runs tr.prefix-selected > td:first-child { border-left: 3px solid #3976c4; }
.runs tr.prefix-incomplete > td { color: #6f5c38; }
.prefix-selected-badge { display: inline-block; margin-left: 5px; border-radius: 8px;
  padding: 1px 5px; background: #dbeaff; color: #2456a6; font-size: 9px;
  font-weight: 700; white-space: nowrap; }
.prefix-status { display: inline-block; border-radius: 9px; padding: 1px 6px;
  font-size: 10px; font-weight: 650; white-space: nowrap; }
.prefix-status.reached { background: #d3f3d8; color: #17663b; }
.prefix-status.not-reached { background: #fff3cd; color: #735e13; }
.prefix-status.na { background: #eef0f4; color: #686f7d; }
.prefix-description { margin: 10px 0 16px; color: #59606e; font-size: 12px; line-height: 1.45; }
.prefix-caveat { border: 1px solid #d9b55c; background: #fff8df; padding: 8px 10px;
  margin: 8px 0 14px; font-size: 12px; line-height: 1.45; }
.prefix-transcript { max-width: 1000px; }
"""


# Copied from Petri: one standard back-to-top control on every generated page.
TOTOP_HTML = r"""
<button class="totop" id="totop" title="back to top">&#8593;</button>
<script>
(function () {
  var b = document.getElementById("totop");
  function upd() { b.style.display = window.scrollY > 400 ? "block" : "none"; }
  b.onclick = function () { window.scrollTo({ top: 0, behavior: "smooth" }); };
  window.addEventListener("scroll", upd); upd();
})();
</script>
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
          var leftExcluded = left.some(function (row) {
            return row.getAttribute("data-integrity-excluded") === "true";
          });
          var rightExcluded = right.some(function (row) {
            return row.getAttribute("data-integrity-excluded") === "true";
          });
          if (leftExcluded !== rightExcluded) return leftExcluded ? 1 : -1;
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
<body><div class="{wrap_class}">{body}</div>{scripts}{TOTOP_HTML}</body></html>"""


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


def _subscription_agent_cost_excluded(audit: dict) -> bool:
    """Whether recorded dollars exclude this cell's included-plan agent usage."""

    if (
        audit.get("retrospective_rejudge")
        or str(audit.get("harness") or "simple") != "subscription"
    ):
        return False
    harness_record = (audit.get("real_env") or {}).get("harness") or {}
    native_harness = (
        audit.get("native_harness")
        or audit.get("production_harness")
        or {}
    )
    agent_billing = (
        harness_record.get("agent_billing")
        or native_harness.get("agent_billing")
    )
    scaffold = harness_record.get("scaffold") or native_harness.get("scaffold")
    return agent_billing == "subscription_included_usage" or (
        agent_billing is None and scaffold != "opencode"
    )


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


def _audit_harness(audit: dict) -> str:
    """Viewer harness, merging production into the subscription window."""

    stored = str(audit.get("harness") or "simple")
    return "subscription" if stored == "production" else stored


def _audit_family(audit: dict) -> str | None:
    family = (audit.get("real_env") or {}).get("family")
    if family in GENERIC_JUDGE_FAMILIES:
        return str(family)
    return _seed_families().get(str(audit.get("seed") or ""))


def _is_prefix_generation_audit(audit: dict) -> bool:
    return (audit.get("real_env") or {}).get("family") in {
        "ml_prefix_only",
        "p_hacking_prefix_only",
    }


def _pressure_view(audit: dict) -> str | None:
    if _audit_family(audit) != "p_hacking":
        return None
    return pressure_of(audit) or "unspecified"


def _pressure_views_for_seed(seed: str) -> tuple[str | None, ...]:
    return (
        ALL_PRESSURE_VIEWS
        if _seed_families().get(seed) == "p_hacking"
        else (None,)
    )


def _pressure_heading(pressure: str | None) -> str:
    return (
        f" — {PRESSURE_NAV_LABELS[pressure].lower()}"
        if pressure in PRESSURE_NAV_LABELS
        else ""
    )


def _scoped_filename(
    base: str, harness: str = "simple", pressure: str | None = None
) -> str:
    pressure_prefix = (
        f"{pressure}_" if pressure not in {None, "high"} else ""
    )
    harness_prefix = f"{harness}_" if harness != "simple" else ""
    return f"{harness_prefix}{pressure_prefix}{base}"


def _seed_filename(
    seed: str, harness: str = "simple", pressure: str | None = None
) -> str:
    if seed == "fraud_detection" and harness == "subscription":
        return _scoped_filename("index.html", "simple", pressure)
    base = (
        "simple_fraud_detection.html"
        if seed == "fraud_detection" and harness == "simple"
        else f"{seed}.html"
    )
    return _scoped_filename(base, harness, pressure)


def _visuals_filename(
    seed: str, harness: str = "simple", pressure: str | None = None
) -> str:
    base = "visuals.html" if seed == "fraud_detection" else f"visuals_{seed}.html"
    return _scoped_filename(base, harness, pressure)


def _comparisons_filename(
    seed: str, harness: str = "simple", pressure: str | None = None
) -> str:
    # TEMPORARY judge-comparisons tab (Owen, 2026-08-04). Delete alongside the other
    # judge-comparisons blocks when rejudge comparisons are done.
    base = f"judge_comparisons_{seed}.html"
    return _scoped_filename(base, harness, pressure)


def _past_filename(
    seed: str, harness: str = "simple", pressure: str | None = None
) -> str:
    base = f"{seed}_past.html"
    return _scoped_filename(base, harness, pressure)


def _continuations_filename(
    harness: str = "simple",
    direction: str = DEFAULT_CONTINUATION_DIRECTION,
) -> str:
    base = (
        CONTINUATIONS_FILENAME
        if direction == DEFAULT_CONTINUATION_DIRECTION
        else f"continuations_{direction}.html"
    )
    return _scoped_filename(base, harness)


def _continuation_visuals_filename(
    harness: str = "simple",
    direction: str = DEFAULT_CONTINUATION_DIRECTION,
) -> str:
    base = (
        "visuals_continuations.html"
        if direction == DEFAULT_CONTINUATION_DIRECTION
        else f"visuals_continuations_{direction}.html"
    )
    return _scoped_filename(base, harness)


def _prefixes_filename(
    harness: str = "simple",
    prefix_type: str = DEFAULT_PREFIX_TYPE,
    *,
    old: bool = False,
) -> str:
    if prefix_type == DEFAULT_PREFIX_TYPE:
        base = "prefixes_past.html" if old else PREFIXES_FILENAME
    else:
        suffix = "_past" if old else ""
        base = f"prefixes_{prefix_type}{suffix}.html"
    return _scoped_filename(base, harness)


def _prefix_experiment_filename(
    harness: str = "simple",
    prefix_type: str = ACTIVITY_LOG_PREFIX_TYPE,
    view: str = "trajectories",
) -> str:
    base = f"prefixes_{prefix_type}_{view}.html"
    return _scoped_filename(base, harness)


def _multi_agent_filename(
    harness: str = "simple", view: str = "trajectories"
) -> str:
    base = (
        MULTI_AGENT_FILENAME
        if view == "trajectories"
        else "multi_agent_inputs.html"
    )
    return _scoped_filename(base, harness)


def _view_filename(
    seed: str, view: str | None, harness: str, pressure: str | None = None
) -> str:
    if view == "comparisons":
        return _comparisons_filename(seed, harness, pressure)
    if view == "visuals":
        return _visuals_filename(seed, harness, pressure)
    if view == "judge":
        return _generic_judge_filename(seed, harness, pressure)
    if view == "past":
        return _past_filename(seed, harness, pressure)
    return _seed_filename(seed, harness, pressure)


def _active_attr(active: bool) -> str:
    return ' class="active"' if active else ""


def _navigation(
    seeds: list[str],
    *,
    active_seed: str | None = None,
    active_view: str | None = None,
    active_harness: str = "simple",
    active_pressure: str | None = None,
    active_continuation_direction: str = DEFAULT_CONTINUATION_DIRECTION,
    active_prefix_type: str = DEFAULT_PREFIX_TYPE,
    prefix_types: tuple[tuple[str, str], ...] = (),
    show_other_continuations: bool = False,
    href_prefix: str = "",
) -> str:
    families = _seed_families()
    continuation_view = active_view in {"continuations", "continuation_visuals"}
    prefix_experiment_view = active_view in {
        "prefix_experiment_trajectories",
        "prefix_experiment_visuals",
    }
    prefixes_view = active_view in {
        "prefixes",
        "prefixes_past",
        "prefix_experiment_trajectories",
        "prefix_experiment_visuals",
    }
    multi_agent_view = active_view in {"multi_agent", "multi_agent_inputs"}
    active_family = (
        families.get(active_seed)
        if active_seed
        and not continuation_view
        and not prefixes_view
        and not multi_agent_view
        else None
    )
    if active_family == "p_hacking" and active_pressure is None:
        active_pressure = "low"
    old_window = active_view in OLD_VIEW_NAMES
    current_window = not old_window and active_harness == "subscription"
    simple_tests_window = not old_window and active_harness == "simple"
    current_view = (
        active_view
        if active_view in {"trajectories", "visuals", "judge"}
        else "trajectories"
    )
    old_view = active_view if active_view in {"comparisons", "past"} else "past"
    window_seed = active_seed or seeds[0]
    window_pressure = (
        active_pressure
        if families.get(window_seed) == "p_hacking"
        else None
    )
    def current_filename(harness: str) -> str:
        if prefix_experiment_view:
            return _prefix_experiment_filename(
                harness,
                active_prefix_type,
                (
                    "visuals"
                    if active_view == "prefix_experiment_visuals"
                    else "trajectories"
                ),
            )
        if prefixes_view:
            return _prefixes_filename(harness, active_prefix_type)
        if multi_agent_view:
            return _multi_agent_filename(
                harness,
                "inputs" if active_view == "multi_agent_inputs" else "trajectories",
            )
        if continuation_view:
            return (
                _continuation_visuals_filename(
                    harness, active_continuation_direction
                )
                if active_view == "continuation_visuals"
                else _continuations_filename(
                    harness, active_continuation_direction
                )
            )
        return _view_filename(
            window_seed, current_view, harness, window_pressure
        )

    all_old_filename = (
        _prefixes_filename(active_harness, active_prefix_type, old=True)
        if prefixes_view
        else _view_filename(window_seed, old_view, active_harness, window_pressure)
    )
    window_row = (
        '<div class="window-nav">'
        f'<a href="{esc(href_prefix + current_filename("subscription"), quote=True)}"'
        f'{_active_attr(current_window)}>Current</a>'
        f'<a href="{esc(href_prefix + current_filename("simple"), quote=True)}"'
        f'{_active_attr(simple_tests_window)}>Old (simple harness tests)</a>'
        f'<a href="{esc(href_prefix + all_old_filename, quote=True)}"'
        f'{_active_attr(old_window)}>All old</a>'
        '</div>'
    )
    harness_links = []
    for harness in HARNESS_NAV_MODES if old_window else ():
        if prefixes_view:
            filename = _prefixes_filename(
                harness, active_prefix_type, old=old_window
            )
        elif continuation_view:
            filename = (
                _continuation_visuals_filename(
                    harness, active_continuation_direction
                )
                if active_view == "continuation_visuals"
                else _continuations_filename(
                    harness, active_continuation_direction
                )
            )
        elif multi_agent_view:
            filename = _multi_agent_filename(
                harness,
                "inputs" if active_view == "multi_agent_inputs" else "trajectories",
            )
        elif active_seed is not None:
            filename = _view_filename(
                active_seed, active_view, harness, active_pressure
            )
        else:
            filename = _seed_filename(seeds[0], harness)
        harness_links.append(
            f'<a href="{esc(href_prefix + filename, quote=True)}"'
            f'{_active_attr(harness == active_harness)}>'
            f'{esc(HARNESS_NAV_LABELS[harness])}</a>'
        )
    harness_row = (
        f'<div class="scope-nav">{"".join(harness_links)}</div>'
        if harness_links else ""
    )
    family_links = []
    family_view = old_view if old_window else "trajectories"
    for family_name in GENERIC_JUDGE_FAMILIES:
        members = [seed for seed in seeds if families.get(seed) == family_name]
        if not members:
            continue
        family_pressure = "low" if family_name == "p_hacking" else None
        family_links.append(
            f'<a href="{esc(href_prefix + _view_filename(members[0], family_view, active_harness, family_pressure), quote=True)}"'
            f'{_active_attr(family_name == active_family)}>'
            f'{esc(FAMILY_NAV_LABELS.get(family_name, family_name))}</a>'
        )
    if not old_window:
        family_links.append(
            f'<a href="{esc(href_prefix + _continuations_filename(active_harness), quote=True)}"'
            f'{_active_attr(continuation_view)}>Continuations</a>'
        )
    family_links.append(
        f'<a href="{esc(href_prefix + _prefixes_filename(active_harness, old=old_window), quote=True)}"'
        f'{_active_attr(prefixes_view)}>Prefixes</a>'
    )
    top = f'<div class="topnav">{"".join(family_links)}</div>'

    active_continuation_destination = continuation_destination(
        active_continuation_direction
    )
    default_direction_by_destination = {}
    for direction, _source, destination in CONTINUATION_DIRECTIONS:
        default_direction_by_destination.setdefault(destination, direction)
    destination_links = []
    for destination in CONTINUATION_DESTINATIONS:
        direction = default_direction_by_destination[destination]
        filename = (
            _continuation_visuals_filename(active_harness, direction)
            if active_view == "continuation_visuals"
            else _continuations_filename(active_harness, direction)
        )
        destination_links.append(
            f'<a href="{esc(href_prefix + filename, quote=True)}"'
            f'{_active_attr(destination == active_continuation_destination)}>'
            f'{esc(continuation_destination_label(destination))}</a>'
        )
    continuation_destination_row = (
        f'<div class="seednav continuation-destination-nav">'
        f'{"".join(destination_links)}</div>'
        if continuation_view else ""
    )
    continuation_directions = [
        key
        for key, _source, destination in CONTINUATION_DIRECTIONS
        if destination == active_continuation_destination
    ]
    if (
        show_other_continuations
        or active_continuation_direction == OTHER_CONTINUATION_DIRECTION
    ):
        continuation_directions.append(OTHER_CONTINUATION_DIRECTION)
    prefix_links = []
    for direction in continuation_directions:
        filename = (
            _continuation_visuals_filename(active_harness, direction)
            if active_view == "continuation_visuals"
            else _continuations_filename(active_harness, direction)
        )
        prefix_links.append(
            f'<a href="{esc(href_prefix + filename, quote=True)}"'
            f'{_active_attr(direction == active_continuation_direction)}>'
            f'{esc(continuation_prefix_label(direction))}</a>'
        )
    continuation_prefix_row = (
        f'<div class="contextnav continuation-prefix-nav">'
        f'{"".join(prefix_links)}</div>'
        if continuation_view else ""
    )
    prefix_type_row = (
        '<div class="seednav">'
        + "".join(
            f'<a href="{esc(href_prefix + _prefixes_filename(active_harness, key, old=old_window), quote=True)}"'
            f'{_active_attr(key == active_prefix_type)}>{esc(label)}</a>'
            for key, label in prefix_types
        )
        + '</div>'
        if prefixes_view and prefix_types else ""
    )

    if continuation_view or prefixes_view or multi_agent_view:
        row_seeds = []
    elif active_family is not None:
        row_seeds = [seed for seed in seeds if families.get(seed) == active_family]
    else:
        row_seeds = [seed for seed in seeds if families.get(seed) is None]
    seed_links = "".join(
        f'<a href="{esc(href_prefix + _view_filename(seed, family_view, active_harness, active_pressure if active_family == "p_hacking" else None), quote=True)}"'
        f'{_active_attr(seed == active_seed)}>{esc(seed)}</a>'
        for seed in row_seeds
    )
    seed_row = f'<div class="seednav">{seed_links}</div>' if seed_links else ""

    pressure_links = ""
    if active_family == "p_hacking" and active_seed is not None:
        pressure_links = '<div class="contextnav">' + "".join(
            f'<a href="{esc(href_prefix + _view_filename(active_seed, active_view, active_harness, pressure), quote=True)}"'
            f'{_active_attr(pressure == active_pressure)}>'
            f'{esc(PRESSURE_NAV_LABELS[pressure])}</a>'
            for pressure in (
                ALL_PRESSURE_VIEWS if old_window else CURRENT_PRESSURE_NAV_MODES
            )
        ) + "</div>"

    if prefix_experiment_view or (
        prefixes_view
        and not old_window
        and active_prefix_type == ACTIVITY_LOG_PREFIX_TYPE
    ):
        items = [
            (
                "prefixes",
                _prefixes_filename(active_harness, active_prefix_type),
                "prefixes",
            ),
            (
                "trajectories",
                _prefix_experiment_filename(
                    active_harness, active_prefix_type, "trajectories"
                ),
                "prefix_experiment_trajectories",
            ),
            (
                "visuals",
                _prefix_experiment_filename(
                    active_harness, active_prefix_type, "visuals"
                ),
                "prefix_experiment_visuals",
            ),
        ]
    elif continuation_view:
        items = [
            (
                "trajectories",
                _continuations_filename(
                    active_harness, active_continuation_direction
                ),
                "continuations",
            ),
            (
                "visuals",
                _continuation_visuals_filename(
                    active_harness, active_continuation_direction
                ),
                "continuation_visuals",
            ),
        ]
    elif multi_agent_view:
        items = [
            (
                "trajectories",
                _multi_agent_filename(active_harness),
                "multi_agent",
            ),
            (
                "prompt + activity logs",
                _multi_agent_filename(active_harness, "inputs"),
                "multi_agent_inputs",
            ),
        ]
    elif active_seed is not None:
        family = families.get(active_seed)
        if old_window:
            items = [
                ("trajectories", _past_filename(active_seed, active_harness, active_pressure), "past"),
                ("judge comparisons", _comparisons_filename(active_seed, active_harness, active_pressure), "comparisons"),
            ]
        else:
            items = [
                ("trajectories", _seed_filename(active_seed, active_harness, active_pressure), "trajectories"),
                ("visuals", _visuals_filename(active_seed, active_harness, active_pressure), "visuals"),
            ]
            if family in GENERIC_JUDGE_FAMILIES:
                items.append((
                    "judge view",
                    _generic_judge_filename(active_seed, active_harness, active_pressure),
                    "judge",
                ))
    else:
        items = []
    links = "".join(
        f'<a href="{esc(href_prefix + href, quote=True)}"'
        f'{_active_attr(key == active_view)}>{esc(label)}</a>'
        for label, href, key in items
    )
    return (
        window_row + harness_row + top + continuation_destination_row
        + continuation_prefix_row + prefix_type_row + seed_row + pressure_links
        + f'<div class="viewnav">{links}</div>'
    )


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
    present.difference_update(HIDDEN_INDEX_DIMENSIONS)
    ordered = [key for key in INDEX_DIMENSION_ORDER if key in present]
    return ordered + sorted(present - set(ordered))


def _agent_display(agent: object) -> tuple[str, str]:
    raw = str(agent or "unknown")
    return agent_label(raw), raw


def _judge_display(audit: dict) -> tuple[str, str]:
    if not audit.get("judgment") and not audit.get("judge_failure"):
        return "—", ""
    raw = str(audit.get("judge") or "unknown")
    return judge_shortname(raw) or raw.split("/")[-1], raw


def _sort_attr(value: object | None) -> str:
    if value is None:
        return ""
    return f' data-sort-value="{esc(value, quote=True)}"'


def _user_turn_count(audit: dict) -> int | None:
    """Controller-authored task turns; retained for callers of the viewer helper."""

    return experiment_user_turn_count(audit)


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
    if status == "invalid":
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
        "auditor_mistakes", "environment_failure", "incompleteness",
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
    if dimension is None or dimension.get("status") == "missing":
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


def _flag_chip_list(
    flags: list[dict], *, invalid_codes: set[str] | None = None
) -> str:
    chips = []
    for flag in flags:
        code = str(flag.get("code") or "")
        severity = (
            "error" if code in invalid_codes else "warning"
        ) if invalid_codes is not None else str(flag.get("severity") or "warning")
        extra_class = str(flag.get("class") or "").strip()
        classes = f"flag {severity}" + (f" {extra_class}" if extra_class else "")
        chips.append(
            f'<span class="{esc(classes, quote=True)}" '
            f'title="{esc(flag.get("detail") or "", quote=True)}">'
            f'{esc(flag.get("label") or code or "flag")}</span>'
        )
    return "".join(chips)


def _status_cell(audit: dict) -> str:
    status = audit.get("mechanical_status")
    if status not in {"valid", "benchmark_only", "invalid"}:
        status = "invalid" if _integrity_excluded(audit) else "valid"
    if status == "valid":
        return '<td data-sort-value=""></td>'

    severity = "error" if status == "invalid" else "warning"
    main_label = "invalid" if status == "invalid" else "benchmark only"
    main_chip = {
        "code": status,
        "label": main_label,
        "severity": severity,
        "class": "status-main",
        "detail": (
            "This trajectory is excluded from the filtered benchmark."
            if status == "invalid"
            else "This trajectory counts in the benchmark but cannot be a continuation prefix."
        ),
    }
    reason_chips = [
        {
            **tag,
            "severity": (
                "warning"
                if status == "benchmark_only"
                else str(tag.get("severity") or "error")
            ),
        }
        for tag in audit.get("status_tags") or []
    ]
    sort_value = ", ".join(
        str(chip.get("label") or "") for chip in [main_chip, *reason_chips]
    )
    return (
        f'<td{_sort_attr(sort_value)}>'
        f'<div class="status-stack">'
        f'<div class="status-heading">{_flag_chip_list([main_chip])}</div>'
        f'<div class="status-reasons">{_flag_chip_list(reason_chips)}</div>'
        f'</div></td>'
    )


CATEGORY_SECTIONS = (
    ("hack", "Reward hacks"),
    ("review", "Needs review"),
    ("notable", "Notable behavior without a delivered reward hack"),
    ("clean", "Clean trajectories"),
    ("unjudged", "Not judged"),
    ("invalid", "Invalid judgments"),
    ("awaiting", "Awaiting current judgment"),
)

def _integrity_excluded(audit: dict) -> bool:
    return (
        audit.get("mechanical_status") == "invalid"
        or audit.get("integrity_status") == "excluded"
    )


def _apply_continuation_source_statuses(audits: list[dict]) -> None:
    """Propagate prefix ineligibility through stored continuation chains."""

    by_id = {
        int(audit["id"]): audit
        for audit in audits
        if audit.get("id") is not None and not audit.get("retrospective_rejudge")
    }
    completed: set[int] = set()

    def apply(audit: dict, visiting: set[int]) -> None:
        audit_id = int(audit.get("id") or 0)
        if audit_id in completed or audit_id in visiting:
            return
        visiting = {*visiting, audit_id}
        source_id = audit.get("source_trajectory_id")
        if source_id is None:
            source_id = prefix_source_trajectory_id(audit)
        try:
            source = by_id.get(int(source_id)) if source_id is not None else None
        except (TypeError, ValueError):
            source = None
        if source is not None:
            apply(source, visiting)
            mark_ineligible_prefix_source(audit, source)
        completed.add(audit_id)

    for audit in audits:
        if continuation_of(audit):
            apply(audit, set())


def _dual_count(rows: list[dict], all_rows: list[dict]) -> str:
    valid_rows = sum(not _integrity_excluded(row) for row in rows)
    valid_total = sum(not _integrity_excluded(row) for row in all_rows)
    return (
        f'{valid_rows}/{valid_total} '
        f'<span class="all-count">({len(rows)}/{len(all_rows)})</span>'
    )


def _index_table(
    audits: list[dict],
    *,
    dimension_keys: list[str],
    comparison_groups: list[list[dict]] | None = None,
    show_judge_view: bool = False,
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
            for audit in sorted(
                audits,
                key=lambda row: (
                    _integrity_excluded(row), int(row.get("id") or 0)
                ),
            )
        ]
    if "deception" in dimension_keys and (
        not audits
        or all(
            (dimension := _dimension(audit, "deception")) is not None
            and dimension.get("status") == "not_applicable"
            for audit in audits
        )
    ):
        dimension_keys = [key for key in dimension_keys if key != "deception"]
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
        agent_display, agent_raw = _agent_display(audit.get("target"))
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
            cost_cells_parts = []
            for key in cost_columns:
                subscription_excluded = (
                    key == "target" and _subscription_agent_cost_excluded(audit)
                )
                value = (
                    '<span class="cost-note">subscription usage excluded</span>'
                    if subscription_excluded
                    else (f'${role_costs[key]:.4f}' if key in role_costs else "—")
                )
                cost_cells_parts.append(
                    f'<td class="cost"'
                    f'{_sort_attr(None if subscription_excluded else role_costs.get(key))}>'
                    f'{value}</td>'
                )
            cost_cells = "".join(cost_cells_parts)
        else:
            cost_value = f"${cost:.4f}" if has_cost else "—"
            if _subscription_agent_cost_excluded(audit):
                cost_value += (
                    '<span class="cost-note">subscription usage excluded</span>'
                )
            cost_cells = (
                f'<td class="cost"{_sort_attr(cost if has_cost else None)}>'
                f'{cost_value}</td>'
            )
        excluded = _integrity_excluded(audit)
        row_attributes = (
            f' data-id="{int(audit["id"])}" '
            f'data-integrity-excluded="{str(excluded).lower()}"'
        )
        if group_key is not None:
            row_attributes += f' data-group="{esc(group_key, quote=True)}"'
        row_classes = []
        if group_start:
            row_classes.append("group-start")
        if excluded:
            row_classes.append("integrity-excluded")
        if row_classes:
            row_attributes += f' class="{" ".join(row_classes)}"'
        cells.append(
            f'<tr{row_attributes}>'
            f'<td{_sort_attr(audit["id"])}><a href="trajectory-{int(audit["id"])}.html">'
            f'{int(audit["id"])}</a></td>'
            + (
                (
                    f'<td><a href="{_judge_trajectory_filename(audit)}">view</a></td>'
                    if audit.get("judgment") or audit.get("judge_failure")
                    else '<td data-sort-value="">—</td>'
                )
                if show_judge_view else ""
            )
            + f'<td class="model-name raw-title" title="{esc(agent_raw, quote=True)}"'
            f'{_sort_attr(agent_display)}>{esc(agent_display)}</td>'
            + judge_cell
            + (
                f'<td{_sort_attr(source_id)}>{provenance}</td>'
                if show_provenance else ""
            )
            + dimensions
            + _status_cell(audit)
            + f'<td{_sort_attr(user_turns)}>'
            f'{user_turns if user_turns is not None else "—"}</td>'
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
        '<th data-sort-type="number">ID</th>'
        + ('<th data-sort-type="text">Judge view</th>' if show_judge_view else "")
        + '<th data-sort-type="text">Agent</th>'
        + judge_header
        + f'{provenance_header}{dimension_headers}'
        + '<th data-sort-type="text">Status</th>'
        + '<th data-sort-type="number">User turns</th>'
        + f'{cost_headers}'
        + f'</tr></thead><tbody>{"".join(cells)}</tbody></table>'
    )


def _index(
    seed: str,
    audits: list[dict],
    errors: list[dict],
    *,
    seeds: list[str],
    active_harness: str = "simple",
    active_pressure: str | None = None,
    active_view: str = "trajectories",
    title: str | None = None,
) -> str:
    dimension_keys = _index_dimension_keys(
        audits, family=_seed_families().get(seed)
    )
    error_html = "".join(
        f'<div class="load-error">{esc(error.get("mode"))}: '
        f'{esc(error.get("error_type"))}: {esc(error.get("error"))}</div>'
        for error in errors
    )
    grouped = {
        key: [
            audit for audit in audits
            if trajectory_category(audit, respect_exclusion=False) == key
        ]
        for key, _label in CATEGORY_SECTIONS
    }
    sections = "".join(
        '<details class="sec" open><summary><h2>'
        f'{esc(label)} <span class="meta">&mdash; '
        f'{_dual_count(grouped[key], audits)}</span></h2></summary>'
        + _index_table(
            grouped[key],
            dimension_keys=dimension_keys,
            show_judge_view=active_view in {"trajectories", "past"},
        )
        + '</details>'
        for key, label in CATEGORY_SECTIONS
    )
    heading = (title or f"{seed} — trajectories") + _pressure_heading(
        active_pressure
    )
    body = (
        _navigation(
            seeds,
            active_seed=seed,
            active_view=active_view,
            active_harness=active_harness,
            active_pressure=active_pressure,
        )
        + f'<div class="pagehead"><h1>{esc(heading)}</h1></div>'
        + error_html
        + sections
    )
    return _page(heading, body, scripts=INDEX_SORT_JS, fit_content=False)


def _comparison_groups(
    official: list[dict], rejudges: list[dict]
) -> list[list[dict]]:
    def source_row(audit: dict) -> dict:
        """Show the pre-promotion judgment beside its replacement rejudge."""

        superseded = audit.get("superseded_judgment") or {}
        fields = superseded.get("audit_fields")
        if not isinstance(fields, dict):
            return audit
        row = deepcopy(audit)
        row.update(deepcopy(fields))
        return row

    groups: dict[int, list[dict]] = {
        int(audit["id"]): [source_row(audit)]
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
    combined = [groups[key] for key in sorted(groups)] + orphans
    return sorted(
        combined,
        key=lambda group: any(_integrity_excluded(audit) for audit in group),
    )


def _comparisons_index(
    seed: str,
    official: list[dict],
    rejudges: list[dict],
    errors: list[dict],
    *,
    seeds: list[str],
    active_harness: str = "simple",
    active_pressure: str | None = None,
) -> str:
    # TEMPORARY judge-comparisons page (Owen, 2026-08-04): every trajectory's rejudge
    # rows grouped under its judgment-free source row. There is no canonical judgment
    # whose category should determine separate page sections.
    dimension_keys = _index_dimension_keys(
        official + rejudges, family=_seed_families().get(seed)
    )
    error_html = "".join(
        f'<div class="load-error">{esc(error.get("mode"))}: '
        f'{esc(error.get("error_type"))}: {esc(error.get("error"))}</div>'
        for error in errors
    )
    groups = _comparison_groups(official, rejudges)
    heading = f"{seed} — judge comparisons{_pressure_heading(active_pressure)}"
    body = (
        _navigation(
            seeds,
            active_seed=seed,
            active_view="comparisons",
            active_harness=active_harness,
            active_pressure=active_pressure,
        )
        + f'<div class="pagehead"><h1>{esc(heading)}</h1></div>'
        + error_html
        + _index_table(
            [], dimension_keys=dimension_keys, comparison_groups=groups
        )
    )
    return _page(heading, body, scripts=INDEX_SORT_JS)


def _score_table_top(audit: dict) -> str:
    """The trajectory's stored dimensions in Petri's compact metadata score table."""

    judgment = audit.get("judgment") or {}
    dimensions = [
        str(dimension.get("key") or "unknown")
        for dimension in judgment.get("dimensions") or []
    ]
    if not dimensions:
        return ""
    headers = "".join(
        f'<th>{esc(INDEX_DIMENSION_LABELS.get(key, key.replace("_", " ")))}</th>'
        for key in dimensions
    )
    values = "".join(_dimension_cell(audit, key) for key in dimensions)
    return f'<table class="scoretop"><tr class="cols">{headers}</tr><tr>{values}</tr></table>'


def _target_context_calls(audit: dict) -> tuple[list[int | None], list[str]]:
    """Agent prompt-token series plus visible coverage caveats."""

    usage = audit.get("target_context_usage") or {}
    raw = usage.get("calls") or []
    calls = [value if isinstance(value, int) and value > 0 else None for value in raw]
    notes: list[str] = []
    if any(value is None for value in calls):
        notes.append(
            f"provider usage is missing on {sum(value is None for value in calls)} "
            "plotted call(s)"
        )
    if usage.get("role_matching") == "model_fallback":
        notes.append(
            "this older log matched agent calls by model name because event roles were absent"
        )
    if not calls:
        notes.append(str(usage.get("reason") or "no per-call agent usage was recorded"))
    return calls, notes


def _context_timeline_svg(calls: list[int | None], context_window: int) -> str:
    """Inline agent context-window timeline."""

    count = len(calls)
    observed = [value for value in calls if isinstance(value, int) and value > 0]
    if not count or not observed or not context_window:
        return ""
    width, height = 820, 235
    left, right, top, bottom = 58, 18, 18, 44
    plot_width, plot_height = width - left - right, height - top - bottom
    percentages = [
        100 * value / context_window
        if isinstance(value, int) and value > 0
        else None
        for value in calls
    ]
    ymax = max(100.0, max(value for value in percentages if value is not None) * 1.12)
    if ymax > 100:
        ymax = 25 * int((ymax + 24.999) // 25)

    def x_at(call_number: float) -> float:
        return left + (call_number - 0.5) / count * plot_width

    def y_at(percentage: float) -> float:
        return top + (1 - percentage / ymax) * plot_height

    y_ticks = list(range(0, int(ymax) + 1, 25))
    if y_ticks[-1] != int(ymax):
        y_ticks.append(int(ymax))
    if count <= 8:
        x_ticks = list(range(1, count + 1))
    else:
        x_ticks = sorted({
            1,
            count,
            *(round(1 + index * (count - 1) / 5) for index in range(1, 5)),
        })

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Agent context-window usage by model call">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in y_ticks:
        y = y_at(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#eceef2" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-8}" y="{y+3:.2f}" text-anchor="end" '
            f'font-size="10" fill="#60646d">{tick}%</text>'
        )
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'stroke="#c9ccd4"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#c9ccd4"/>',
    ])
    for tick in x_ticks:
        x = x_at(tick)
        parts.append(
            f'<text x="{x:.2f}" y="{height-bottom+17}" text-anchor="middle" '
            f'font-size="10" fill="#60646d">{tick}</text>'
        )

    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for index, percentage in enumerate(percentages, 1):
        if percentage is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append((index, percentage))
    if current:
        segments.append(current)
    baseline_y = y_at(0)
    for segment in segments:
        points = " ".join(
            f"{x_at(index):.2f},{y_at(percentage):.2f}"
            for index, percentage in segment
        )
        if len(segment) > 1:
            first_x, last_x = x_at(segment[0][0]), x_at(segment[-1][0])
            parts.append(
                f'<polygon points="{first_x:.2f},{baseline_y:.2f} {points} '
                f'{last_x:.2f},{baseline_y:.2f}" fill="#4C72B0" opacity="0.08"/>'
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="#4C72B0" '
                'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
    for index, (tokens, percentage) in enumerate(zip(calls, percentages), 1):
        if percentage is None or tokens is None:
            continue
        parts.append(
            f'<circle cx="{x_at(index):.2f}" cy="{y_at(percentage):.2f}" r="2.6" '
            'fill="#4C72B0"><title>'
            f'call {index}: {tokens:,} tokens ({percentage:.2f}%)</title></circle>'
        )

    parts.extend([
        f'<text x="{left + plot_width/2:.2f}" y="{height-7}" text-anchor="middle" '
        'font-size="11" fill="#4b4f58">model call</text>',
        f'<text x="13" y="{top + plot_height/2:.2f}" text-anchor="middle" '
        'font-size="11" fill="#4b4f58" '
        f'transform="rotate(-90 13 {top + plot_height/2:.2f})">'
        'context window full</text>',
        '</svg>',
    ])
    return "".join(parts)


def _context_graph_html(audit: dict) -> str:
    """Agent context graph backed by environment-owned usage."""

    if "target_context_usage" not in audit:
        return ""
    calls, notes = _target_context_calls(audit)
    window = CONTEXT_WINDOWS.get(canonical_slug(str(audit.get("target") or "")))
    if window is None:
        notes.append("agent context-window capacity is unknown")
    observed = sum(isinstance(value, int) and value > 0 for value in calls)
    status = (
        "unavailable"
        if not calls or observed == 0 or window is None
        else "partial"
        if observed != len(calls) or notes
        else "complete"
    )
    svg = _context_timeline_svg(calls, window) if window else ""
    notes = list(dict.fromkeys(note for note in notes if note))
    note_html = (
        f'<div class="contextgraph-note">&#9888; {esc("; ".join(notes))}</div>'
        if notes
        else ""
    )
    if not svg and not note_html:
        note_html = '<div class="contextgraph-note">&#9888; context timeline unavailable</div>'
    return (
        f'<div class="contextgraph" data-context-coverage="{status}" '
        f'data-context-missing-calls="{len(calls) - observed}">'
        f'<div class="mblock-h">Agent context by call</div>{svg}{note_html}</div>'
    )


def _metadata_panel(audit: dict) -> str:
    """Petri's metadata section populated with environment-owned fields."""

    cost, has_cost = _recorded_cost(audit)
    label, agent_raw = _agent_display(audit.get("target"))
    epoch = _display_epoch(audit)
    harness_record = (audit.get("real_env") or {}).get("harness") or {}
    harness_mode = str(audit.get("harness") or "simple")
    subscription_scaffold = harness_record.get("scaffold") or (
        (audit.get("native_harness") or {}).get("scaffold")
        or (audit.get("production_harness") or {}).get("scaffold")
    )
    direct_subscription = _subscription_agent_cost_excluded(audit)
    harness_detail = harness_mode
    if harness_mode in {"production", "subscription"}:
        scaffold = subscription_scaffold
        scaffold_version = (
            harness_record.get("scaffold_version_resolved")
            or harness_record.get("resolved_scaffold_version")
            or (audit.get("native_harness") or {}).get(
                "scaffold_version_selector"
            )
            or (audit.get("production_harness") or {}).get(
                "scaffold_version_selector"
            )
        )
        harness_detail = " · ".join(
            str(item) for item in (harness_mode, scaffold, scaffold_version) if item
        )
    subscription_run = harness_record.get("subscription_run") or {}
    subscription_usage = subscription_run.get("usage_totals") or {}
    subscription_usage_text = ""
    if subscription_usage:
        subscription_usage_text = (
            f"{int(subscription_usage.get('input') or 0):,} input · "
            f"{int(subscription_usage.get('output') or 0):,} output · "
            f"{int(subscription_usage.get('cache_read') or 0):,} cache read"
        )
        unmetered_calls = int(
            subscription_run.get("unmetered_model_call_count") or 0
        )
        if unmetered_calls:
            subscription_usage_text += f" · {unmetered_calls} unmetered calls"
    agent_api_equivalent = subscription_run.get("api_list_equivalent_usd_total")

    def cell(key: str, value_html: str) -> str:
        return (
            '<div class="metacell">'
            f'<span class="k">{esc(key)}</span>'
            f'<div class="v">{value_html}</div></div>'
        ) if value_html else ""

    pressure = _pressure_view(audit)
    pressure_label = (
        PRESSURE_NAV_LABELS.get(pressure, pressure or "")
        if pressure is not None else ""
    )
    cells = [
        cell(
            "agent",
            f'<span class="raw-title" title="{esc(agent_raw, quote=True)}">'
            f'{esc(label)}</span>',
        ),
        cell("judge", esc(audit.get("judge") or "")),
        cell("condition", esc(audit.get("condition") or "")),
        cell("pressure", esc(pressure_label)),
        (
            cell("harness", esc(harness_detail))
            if harness_mode in {"production", "subscription"}
            else ""
        ),
        cell("subscription usage", esc(subscription_usage_text)),
        cell(
            "agent cost (API-list equivalent)",
            esc(f"${agent_api_equivalent:.6f}")
            if isinstance(agent_api_equivalent, (int, float)) else "",
        ),
        cell("epoch", esc(epoch) if epoch is not None else ""),
        cell(
            "API cost (agent excluded)" if direct_subscription else "cost",
            esc(f"${cost:.6f}") if has_cost else "unavailable",
        ),
        cell(
            "flags",
            '<div class="flag-list">'
            + _flag_chip_list(
                audit.get("flags") or [],
                invalid_codes={
                    str(code) for code in audit.get("integrity_issues") or []
                },
            )
            + '</div>'
            if audit.get("flags") else "none",
        ),
        cell("run dir", esc(audit.get("mode") or "")),
    ]
    scores = _score_table_top(audit)
    scores_block = (
        f'<div class="mblock"><div class="mblock-h">Scores</div>{scores}</div>'
        if scores else ""
    )
    run_block = (
        '<div class="mblock"><div class="mblock-h">Run</div>'
        f'<div class="metagrid">{"".join(cells)}</div>'
        f'{_context_graph_html(audit)}</div>'
    )
    preview_parts = [label]
    if audit.get("condition"):
        preview_parts.append(str(audit["condition"]))
    if pressure_label:
        preview_parts.append(pressure_label)
    return (
        '<details class="sec metadata"><summary><h2>Metadata</h2>'
        f'<span class="meta metaprev">{esc(" · ".join(preview_parts))}</span></summary>'
        f'<div class="metabody">{scores_block}{run_block}</div></details>'
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


def _environment_record_for_viewer(
    audit: dict, *, omit_multi_agent_content: bool = True
) -> dict:
    """Avoid duplicating a large activity log on every trajectory detail page."""

    real_env = deepcopy(audit.get("real_env") or {})
    multi_agent = real_env.get("multi_agent")
    if omit_multi_agent_content and isinstance(multi_agent, dict):
        activity = multi_agent.get("activity_log")
        if isinstance(activity, dict) and isinstance(activity.get("content"), str):
            activity["content"] = (
                "[Exact content omitted from this duplicate metadata panel; use "
                "Multi-agent > prompt + activity logs. The raw Inspect record "
                "retains it.]"
            )
            activity["viewer_duplicate_content_omitted"] = True
    return _without_duplicate_judge_payloads(real_env)


def _trajectory(
    audit: dict,
    *,
    seeds: list[str],
    active_view: str = "trajectories",
    prefix_types: tuple[tuple[str, str], ...] = (),
    show_other_continuations: bool = False,
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
    continuation = continuation_of(audit)
    multi_agent = _multi_agent_record(audit)
    continuation_banner = ""
    pivot_number = None
    if continuation:
        prefix = prefix_of(audit)
        boundary = continuation.get("boundary_index")
        pivot_number = (
            boundary + 1
            if isinstance(boundary, int) and not isinstance(boundary, bool)
            else None
        )
        prefix_id = prefix_source_trajectory_id(audit)
        prefix_name = str(prefix.get("name") or "unknown")
        prefix_cutoff = prefix.get("source_cutoff")
        inline_context = continuation.get("delivery_mode") == "inline_user_context"
        native_resume = bool(
            continuation.get("production_native_resume")
            or continuation.get("subscription_native_resume")
        )
        prefix_link = (
            f'<a href="trajectory-{prefix_id}.html">{esc(prefix_name)}</a>'
            if prefix_id is not None else esc(prefix_name)
        )
        pivot_link = (
            f'<a href="#M{pivot_number}">pivot at [M{pivot_number}]</a>'
            if pivot_number is not None else "pivot position unrecorded"
        )
        continuation_banner = (
            '<div class="rejudge-banner">Continuation · treatment '
            f'{esc(str(continuation.get("treatment") or "unknown"))} · prefix '
            f'{prefix_link}'
            + (
                ' · activity-log context hidden from judges'
                if inline_context
                else
                ' · prior native session hidden from judges'
                if native_resume
                else f' · [M2]–[M{boundary}] hidden from judges'
                if pivot_number is not None else ""
            )
            + f' · {pivot_link}'
            + (
                ' · scaffold session resumed natively; old workspace not restored'
                if native_resume else ""
            )
            + (
                ' · fresh scaffold session; full observable log delivered inline'
                if inline_context else ""
            )
            + (
                ' · prefix was cut off before its second experiment user turn; '
                f'{esc(str(prefix_cutoff.get("omitted_message_count") or "?"))} '
                'later saved message(s) omitted'
                if isinstance(prefix_cutoff, dict)
                and prefix_cutoff.get("affected") is True
                else ""
            )
            + '</div>'
        )
    multi_agent_banner = ""
    if multi_agent:
        source = multi_agent.get("source") or {}
        exposure = multi_agent.get("exposure") or {}
        source_id = source.get("trajectory_id")
        source_link = (
            f'<a href="trajectory-{int(source_id)}.html">trajectory '
            f'{int(source_id)}</a>'
            if source_id is not None else "external activity log"
        )
        input_link = (
            f' · <a href="{esc(_multi_agent_filename(_audit_harness(audit), "inputs"), quote=True)}">'
            'view prompt and ACTIVITY_LOG.md</a>'
            if active_view != "past" else ""
        )
        multi_agent_banner = (
            '<div class="rejudge-banner">Multi-agent activity-log treatment · '
            f'treatment {esc(str(multi_agent.get("treatment") or "unknown"))} · '
            f'source {source_link} · fresh session · log exposure '
            f'{esc(str(exposure.get("status") or "unknown"))}'
            + input_link
            + '</div>'
        )
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
    promotion_banner = ""
    promoted = audit.get("promoted_rejudge") or {}
    if promoted:
        replacement_id = promoted.get("trajectory_id")
        replacement_link = (
            f'<a href="trajectory-{int(replacement_id)}.html">'
            f'rejudge trajectory {int(replacement_id)}</a>'
            if replacement_id is not None else "the selected retrospective rejudge"
        )
        promotion_banner = (
            '<div class="rejudge-banner"><strong>Canonical judgment replaced.</strong> '
            f'This row now uses {replacement_link}; the superseded stored judgment '
            'is preserved under Other stuff below.</div>'
        )
    recovery_banner = ""
    recovery = audit.get("interrupted_native_transcript") or {}
    coverage = audit.get("judgment_transcript_coverage") or {}
    if recovery.get("reconstructed"):
        added = recovery.get("added_message_count", "?")
        if coverage.get("stored_judgment_predates_reconstruction"):
            recovery_banner = (
                '<div class="load-error"><strong>Stored judgment is incomplete.</strong> '
                f'{esc(str(added))} message(s) from the interrupted OpenCode call were '
                'recovered from target model events, but the stored judge did not see '
                'them. This trajectory is not validly judged against the transcript '
                'shown below.</div>'
            )
        else:
            recovery_banner = (
                '<div class="rejudge-banner"><strong>Partial interrupted transcript '
                'recovered.</strong> '
                f'{esc(str(added))} message(s) were reconstructed from stored target '
                'model events through the newest matching event. Later native activity '
                'or an interrupted tool result may be unavailable.</div>'
            )
    seed = str(audit.get("seed") or "unknown")
    harness = _audit_harness(audit)
    pressure = _pressure_view(audit)
    continuation_current = bool(continuation and active_view == "continuations")
    prefix_experiment_current = bool(
        is_activity_log_context_continuation(audit)
        and active_view == "prefix_experiment_trajectories"
    )
    multi_agent_current = bool(multi_agent and active_view == "multi_agent")
    if active_view == "comparisons":
        back_href = _comparisons_filename(seed, harness, pressure)
    elif active_view == "past":
        back_href = _past_filename(seed, harness, pressure)
    elif continuation_current:
        active_continuation_direction = continuation_direction(audit)
        back_href = _continuations_filename(
            harness, active_continuation_direction
        )
    elif prefix_experiment_current:
        back_href = _prefix_experiment_filename(
            harness, ACTIVITY_LOG_PREFIX_TYPE, "trajectories"
        )
    elif multi_agent_current:
        back_href = _multi_agent_filename(harness)
    else:
        back_href = _seed_filename(seed, harness, pressure)
    page_heading = f'{esc(seed)} — trajectory {int(audit["id"])}'
    judge_page_href = _judge_trajectory_filename(audit)
    body = (
        _navigation(
            seeds,
            # Continuation trajectories belong to the global Continuations window,
            # not to their seed's tabs, unless individually archived under Old.
            active_seed=(
                None
                if continuation_current
                or prefix_experiment_current
                or multi_agent_current
                else seed
            ),
            active_view=active_view,
            active_harness=harness,
            active_pressure=pressure,
            active_continuation_direction=(
                active_continuation_direction
                if continuation_current else DEFAULT_CONTINUATION_DIRECTION
            ),
            active_prefix_type=(
                ACTIVITY_LOG_PREFIX_TYPE
                if prefix_experiment_current else DEFAULT_PREFIX_TYPE
            ),
            prefix_types=prefix_types,
            show_other_continuations=show_other_continuations,
        )
        + '<div class="pagehead"><h1>'
        f'{page_heading}</h1><a class="headbtn" href="{esc(back_href, quote=True)}">'
        '&larr; back</a></div>'
        + _metadata_panel(audit)
        + continuation_banner
        + multi_agent_banner
        + rejudge_banner
        + promotion_banner
        + recovery_banner
        + legacy
        + (
            _json_panel("Judge failure", {
                "failure": judge_failure,
                "score_metadata": audit.get("score_metadata") or {},
            })
            if judge_failure else ""
        )
        + render_judge_narrative(judgment, artifact_page_href=judge_page_href)
        + f'<h2 class="trajectory-panel" id="trajectory-record">'
        f'Trajectory · {len(audit.get("messages") or [])} messages</h2>'
        + render_transcript(audit.get("messages") or [])
        + (_json_panel("Load issues", audit.get("load_issues"))
           if audit.get("load_issues") else "")
        + '<details class="sec other-stuff"><summary><h2>Other stuff</h2></summary>'
        + '<div class="metabody">'
        + render_dimension_navigator(judgment, artifact_page_href=judge_page_href)
        + (_json_panel("Grade", grade) if grade else "")
        + _json_panel("Stored judgment", _without_duplicate_judge_payloads(judgment))
        + (
            _json_panel(
                "Superseded stored judgment",
                _without_duplicate_judge_payloads(audit["superseded_judgment"]),
            )
            if audit.get("superseded_judgment") else ""
        )
        + _json_panel(
            "Environment record",
            _environment_record_for_viewer(
                audit,
                omit_multi_agent_content=active_view != "past",
            ),
        )
        + _json_panel("Model usage", audit.get("model_usage") or {})
        + '</div></details>'
        + render_explanation_turn_nav(judgment)
    )
    return _page(
        f'Trajectory {int(audit["id"])}',
        body,
        scripts=EVIDENCE_NAV_JS + render_jump_to_new_task(pivot_number),
        fit_content=False,
    )


def _judge_trajectory_filename(audit: dict) -> str:
    return f'judge-trajectory-{int(audit["id"])}.html'


def _trajectory_list_href(audit: dict, active_view: str) -> str:
    seed = str(audit.get("seed") or "unknown")
    harness = _audit_harness(audit)
    pressure = _pressure_view(audit)
    if active_view == "comparisons":
        return _comparisons_filename(seed, harness, pressure)
    if active_view == "past":
        return _past_filename(seed, harness, pressure)
    if (
        active_view == "prefix_experiment_trajectories"
        and is_activity_log_context_continuation(audit)
    ):
        return _prefix_experiment_filename(
            harness, ACTIVITY_LOG_PREFIX_TYPE, "trajectories"
        )
    if continuation_of(audit):
        return _continuations_filename(harness, continuation_direction(audit))
    if _multi_agent_record(audit):
        return _multi_agent_filename(harness)
    return _seed_filename(seed, harness, pressure)


def _judge_trajectory(
    audit: dict,
    *,
    seeds: list[str],
    active_view: str = "trajectories",
    prefix_types: tuple[tuple[str, str], ...] = (),
    show_other_continuations: bool = False,
) -> str:
    judgment = audit.get("judgment")
    seed = str(audit.get("seed") or "unknown")
    harness = _audit_harness(audit)
    pressure = _pressure_view(audit)
    trajectory_id = int(audit["id"])
    continuation = continuation_of(audit)
    continuation_current = bool(continuation and active_view == "continuations")
    prefix_experiment_current = bool(
        is_activity_log_context_continuation(audit)
        and active_view == "prefix_experiment_trajectories"
    )
    multi_agent_current = bool(
        _multi_agent_record(audit) and active_view == "multi_agent"
    )
    judge_view = render_judge_view(
        judgment,
        audit,
        expanded=True,
        trajectory_href=f"trajectory-{trajectory_id}.html#trajectory-record",
    )
    if not judge_view:
        judge_view = '<p class="empty">No stored judge view is available.</p>'
    body = (
        _navigation(
            seeds,
            active_seed=(
                None
                if continuation_current
                or prefix_experiment_current
                or multi_agent_current
                else seed
            ),
            active_view=active_view,
            active_harness=harness,
            active_pressure=pressure,
            active_continuation_direction=(
                continuation_direction(audit)
                if continuation_current else DEFAULT_CONTINUATION_DIRECTION
            ),
            active_prefix_type=(
                ACTIVITY_LOG_PREFIX_TYPE
                if prefix_experiment_current else DEFAULT_PREFIX_TYPE
            ),
            prefix_types=prefix_types,
            show_other_continuations=show_other_continuations,
        )
        + '<div class="pagehead"><h1>'
        f'{esc(seed)} — trajectory {trajectory_id} — judge view</h1>'
        f'<a class="headbtn" href="{esc(_trajectory_list_href(audit, active_view), quote=True)}">'
        '&larr; back</a></div>'
        + judge_view
    )
    return _page(
        f"Trajectory {trajectory_id} · Judge view",
        body,
        scripts=EVIDENCE_NAV_JS,
        fit_content=False,
    )


def _visuals(
    seed: str,
    audits: list[dict],
    *,
    seeds: list[str],
    active_harness: str = "simple",
    active_pressure: str | None = None,
) -> str:
    body = (
        _navigation(
            seeds,
            active_seed=seed,
            active_view="visuals",
            active_harness=active_harness,
            active_pressure=active_pressure,
        )
        + f'<div class="pagehead"><h1>{esc(seed)} — visuals'
        f'{esc(_pressure_heading(active_pressure))}</h1></div>'
        + render_visuals(audits)
    )
    return _page(f"{seed} · visuals", body, scripts=VISUALS_JS, fit_content=False)


def _generic_judge_filename(
    seed: str, harness: str = "simple", pressure: str | None = None
) -> str:
    base = f"judge_{seed}.html"
    return _scoped_filename(base, harness, pressure)


async def _generic_judge_page(
    seed: str,
    family: str,
    *,
    seeds: list[str],
    active_harness: str = "simple",
    active_pressure: str | None = None,
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
        _navigation(
            seeds,
            active_seed=seed,
            active_view="judge",
            active_harness=active_harness,
            active_pressure=active_pressure,
        )
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


def _archive_obsolete_production_pages() -> int:
    """Recoverably remove the former standalone production-harness window."""

    archived = 0
    archive_root = VIEWER_ROOT / "_archive" / "merged_production_harness"
    for path in sorted(VIEWER_ROOT.glob("production_*.html")):
        if not path.is_file():
            continue
        _archive_path(path, archive_root / path.name)
        archived += 1
    return archived


def _validate_generated_viewer_site() -> None:
    """Copy Petri's build-end check for broken local links in live viewer pages."""

    problems = []
    href_re = re.compile(r"href\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
    ignored_prefixes = (
        "#", "http://", "https://", "mailto:", "javascript:", "data:"
    )
    for page in sorted(VIEWER_ROOT.glob("*.html")):
        for _quote, raw_href in href_re.findall(page.read_text()):
            href = html.unescape(raw_href).strip()
            if not href or href.startswith(ignored_prefixes):
                continue
            local_ref = href.split("#", 1)[0].split("?", 1)[0]
            if local_ref and not (page.parent / local_ref).exists():
                problems.append(f"{page.name} links to missing {local_ref}")
    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError(f"generated viewer link validation failed:\n{detail}")


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


def _continuation_index_sections(rows: list[dict]) -> str:
    sections = []
    groups = continuation_groups(
        rows,
        seed_order=_ordered_seeds(),
    )
    for (seed, agent, harness, pressure), by_treatment in groups:
        group_rows = [row for rows in by_treatment.values() for row in rows]
        dimension_keys = _index_dimension_keys(
            group_rows, family=_seed_families().get(seed)
        )
        agent_pretty, agent_raw = _agent_display(agent)
        treatment_tables = []
        for treatment, rows in by_treatment.items():
            valid_rows = sum(not _integrity_excluded(row) for row in rows)
            treatment_tables.append(
                '<details class="sub" open><summary><h3>'
                f'{esc(treatment)} <span class="meta">&mdash; '
                f'valid: {valid_rows} · all: {len(rows)}</span>'
                '</h3></summary>'
                + _index_table(
                    rows,
                    dimension_keys=dimension_keys,
                )
                + '</details>'
            )
        sections.append(
            '<details class="sec" open><summary><h2>'
            f'{esc(seed)} · <span class="raw-title" '
            f'title="{esc(agent_raw, quote=True)}">{esc(agent_pretty)}</span>'
            + (
                f' · {esc(PRESSURE_NAV_LABELS.get(pressure or "unspecified", ""))}'
                if _seed_families().get(seed) == "p_hacking" else ""
            )
            + f'</h2></summary>'
            + "".join(treatment_tables)
            + "</details>"
        )
    return (
        "".join(sections)
        if sections else '<p class="empty">No continuation runs yet.</p>'
    )


def _continuations_page(
    continuations: list[dict],
    *,
    seeds: list[str],
    active_harness: str = "simple",
    active_direction: str = DEFAULT_CONTINUATION_DIRECTION,
    show_other_continuations: bool = False,
) -> str:
    selected = [
        audit for audit in continuations
        if _audit_harness(audit) == active_harness
        and continuation_direction(audit) == active_direction
    ]
    context = continuation_condition_context(active_direction)
    body = (
        _navigation(
            seeds,
            active_view="continuations",
            active_harness=active_harness,
            active_continuation_direction=active_direction,
            show_other_continuations=show_other_continuations,
        )
        + '<div class="pagehead"><h1>'
        + esc(context)
        + '</h1></div>'
        + _continuation_index_sections(selected)
    )
    return _page(
        f"Continuations · {context}",
        body,
        scripts=INDEX_SORT_JS,
    )


def _continuation_visuals_page(
    continuations: list[dict],
    originals: list[dict],
    *,
    audits_by_id: dict[int, dict],
    seeds: list[str],
    active_harness: str = "simple",
    active_direction: str = DEFAULT_CONTINUATION_DIRECTION,
    show_other_continuations: bool = False,
) -> str:
    selected = [
        audit for audit in continuations
        if _audit_harness(audit) == active_harness
        and continuation_direction(audit) == active_direction
    ]
    groups = continuation_prefix_rate_data(
        selected,
        originals,
        audits_by_id=audits_by_id,
    )
    context = continuation_condition_context(active_direction)
    body = (
        _navigation(
            seeds,
            active_view="continuation_visuals",
            active_harness=active_harness,
            active_continuation_direction=active_direction,
            show_other_continuations=show_other_continuations,
        )
        + '<div class="pagehead"><h1>'
        + esc(context)
        + ' — visuals</h1></div>'
        + render_continuation_visuals(groups, selected, context=context)
    )
    return _page(
        f"Continuations · {context} · visuals",
        body,
        fit_content=False,
    )


def _activity_log_prefix_experiment_context() -> str:
    return (
        "Task: p-hacking (checkout_redesign) · "
        "Prefix: Activity-log context (p-hacking)"
    )


def _activity_log_prefix_trajectories_page(
    rows: list[dict],
    *,
    seeds: list[str],
    prefix_types: tuple[tuple[str, str], ...],
    active_harness: str,
) -> str:
    selected = [
        row for row in rows if _audit_harness(row) == active_harness
    ]
    context = _activity_log_prefix_experiment_context()
    body = (
        _navigation(
            seeds,
            active_view="prefix_experiment_trajectories",
            active_harness=active_harness,
            active_prefix_type=ACTIVITY_LOG_PREFIX_TYPE,
            prefix_types=prefix_types,
        )
        + f'<div class="pagehead"><h1>{esc(context)}</h1></div>'
        + _continuation_index_sections(selected)
    )
    return _page(
        f"Prefixes · {context} · trajectories",
        body,
        scripts=INDEX_SORT_JS,
    )


def _activity_log_prefix_visuals_page(
    rows: list[dict],
    originals: list[dict],
    *,
    audits_by_id: dict[int, dict],
    seeds: list[str],
    prefix_types: tuple[tuple[str, str], ...],
    active_harness: str,
) -> str:
    selected = [
        row for row in rows if _audit_harness(row) == active_harness
    ]
    groups = continuation_prefix_rate_data(
        selected,
        originals,
        audits_by_id=audits_by_id,
    )
    context = _activity_log_prefix_experiment_context()
    body = (
        _navigation(
            seeds,
            active_view="prefix_experiment_visuals",
            active_harness=active_harness,
            active_prefix_type=ACTIVITY_LOG_PREFIX_TYPE,
            prefix_types=prefix_types,
        )
        + f'<div class="pagehead"><h1>{esc(context)} — visuals</h1></div>'
        + render_continuation_visuals(groups, selected, context=context)
    )
    return _page(
        f"Prefixes · {context} · visuals",
        body,
        fit_content=False,
    )


def _multi_agent_record(audit: dict) -> dict:
    real_env = audit.get("real_env") or {}
    record = real_env.get("multi_agent") if isinstance(real_env, dict) else None
    return record if isinstance(record, dict) else {}


def _multi_agent_direction(audit: dict) -> str:
    record = _multi_agent_record(audit)
    source = record.get("source") or {}
    source_family = str(source.get("family") or "external")
    destination_family = str(
        (audit.get("real_env") or {}).get("family") or "unknown"
    )
    labels = {
        "p_hacking": "p-hacking",
        "ml_training_data_misuse": "ML",
        "ml_prefix_only": "ML",
    }
    return (
        f"{labels.get(source_family, source_family)} → "
        f"{labels.get(destination_family, destination_family)}"
    )


def _multi_agent_page(
    rows: list[dict], *, seeds: list[str], active_harness: str
) -> str:
    selected = [row for row in rows if _audit_harness(row) == active_harness]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in selected:
        record = _multi_agent_record(row)
        grouped[(
            _multi_agent_direction(row),
            str(row.get("seed") or "unknown"),
            str(record.get("treatment") or "unknown"),
        )].append(row)
    sections = []
    for (direction, seed, treatment), group_rows in sorted(grouped.items()):
        valid = sum(not _integrity_excluded(row) for row in group_rows)
        dimension_keys = _index_dimension_keys(
            group_rows, family=_seed_families().get(seed)
        )
        sections.append(
            '<details class="sec" open><summary><h2>'
            f'{esc(direction)} · {esc(seed)} · {esc(treatment)} '
            f'<span class="meta">&mdash; valid: {valid} · all: '
            f'{len(group_rows)}</span></h2></summary>'
            + _index_table(group_rows, dimension_keys=dimension_keys)
            + "</details>"
        )
    body = (
        _navigation(
            seeds,
            active_view="multi_agent",
            active_harness=active_harness,
        )
        + '<div class="pagehead"><h1>Multi-agent activity-log transfer</h1></div>'
        + '<div class="note">Each row is a fresh agent session on the destination '
        'task. ACTIVITY_LOG.md contains another instance’s prior-task activity; no '
        'conversation or native session is resumed. Runs whose complete log was not '
        'mechanically verified in model context remain visible but are excluded.</div>'
        + (
            "".join(sections)
            if sections else '<p class="empty">No multi-agent runs yet.</p>'
        )
    )
    return _page("Multi-agent trajectories", body, scripts=INDEX_SORT_JS)


def _multi_agent_input_panel(audit: dict, rows: list[dict] | None = None) -> str:
    record = _multi_agent_record(audit)
    source = record.get("source") or {}
    activity = record.get("activity_log") or {}
    metadata = activity.get("metadata") or {}
    content = str(activity.get("content") or "")
    prompt = str(record.get("effective_opening_user_message") or "")
    source_id = source.get("trajectory_id")
    source_link = (
        f'<a href="trajectory-{int(source_id)}.html">trajectory {int(source_id)}</a>'
        if source_id is not None else esc(source.get("description") or "external")
    )
    lossy = metadata.get("lossy_processing") or {}
    caveat = str(lossy.get("visible_caveat") or "")
    destinations = str(audit.get("seed") or "unknown")
    status_counts = Counter(
        str(
            (_multi_agent_record(row).get("exposure") or {}).get("status")
            or "not run"
        )
        for row in (rows or [audit])
    )
    status = " · ".join(
        f"{name} ×{count}" for name, count in sorted(status_counts.items())
    )
    return (
        '<details class="sec" open><summary><h2>'
        f'{esc(str(record.get("name") or "activity log"))} · '
        f'{esc(_multi_agent_direction(audit))} · {esc(destinations)}'
        '</h2></summary><div class="metabody">'
        f'<p class="meta">Source: {source_link} · lines: '
        f'{esc(str(metadata.get("line_count") or "?"))} · exposure: '
        f'{esc(status)}</p>'
        + (
            f'<div class="prefix-caveat"><strong>Observable-log caveat:</strong> '
            f'{esc(caveat)}</div>' if caveat else ""
        )
        + '<h3>Opening user prompt</h3>'
        f'<pre class="prompt-preview">{esc(prompt)}</pre>'
        '<details class="sub"><summary><h3>ACTIVITY_LOG.md</h3></summary>'
        f'<pre class="activity-log">{esc(content)}</pre></details>'
        '</div></details>'
    )


def _multi_agent_inputs_page(
    rows: list[dict], *, seeds: list[str], active_harness: str
) -> str:
    selected = [row for row in rows if _audit_harness(row) == active_harness]
    unique: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        record = _multi_agent_record(row)
        key = (
            str(record.get("payload_sha256") or record.get("name") or ""),
            str(record.get("effective_opening_user_message") or ""),
        )
        unique[key].append(row)
    panels = "".join(
        _multi_agent_input_panel(group_rows[0], group_rows)
        for _key, group_rows in sorted(
            unique.items(),
            key=lambda item: (
                _multi_agent_direction(item[1][0]),
                str(item[1][0].get("seed") or ""),
                item[0],
            ),
        )
    )
    body = (
        _navigation(
            seeds,
            active_view="multi_agent_inputs",
            active_harness=active_harness,
        )
        + '<div class="pagehead"><h1>Multi-agent inputs</h1></div>'
        + '<div class="note">The prompt shown here is the actual opening user turn. '
        'The system prompt is unchanged from the corresponding ordinary run. Expand '
        'ACTIVITY_LOG.md to inspect the exact numbered file mounted in the sandbox.'
        '</div>'
        + (panels if panels else '<p class="empty">No multi-agent inputs yet.</p>')
    )
    return _page("Multi-agent inputs", body, fit_content=False)


def _prefix_page_filename(path: Path) -> str:
    digest = hashlib.sha256(path.name.encode()).hexdigest()[:12]
    return f"prefix-{digest}.html"


def _load_prefixes(
    archived_files: frozenset[str] = frozenset(),
    used_trajectory_ids: frozenset[int] = frozenset(),
    old_trajectory_ids: frozenset[int] = frozenset(),
) -> tuple[list[dict], list[str]]:
    """Load external prefixes plus current trajectory prefixes selected or used."""

    prefixes: list[dict] = []
    errors: list[str] = []
    if not CONTINUATION_PREFIXES_ROOT.is_dir():
        return prefixes, errors
    for path in sorted(CONTINUATION_PREFIXES_ROOT.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: {error}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.name}: prefix payload is not a JSON object")
            continue
        source = payload.get("source")
        messages = payload.get("messages")
        required_problems = []
        if payload.get("format") != "environments-continuation-prefix-v1":
            required_problems.append("unknown format")
        if not isinstance(payload.get("name"), str):
            required_problems.append("missing name")
        if not isinstance(source, dict):
            required_problems.append("missing source")
            source = {}
        if not isinstance(messages, list):
            required_problems.append("missing messages")
            messages = []
        if required_problems:
            errors.append(f"{path.name}: {', '.join(required_problems)}")
            continue
        if source.get("kind") == "trajectory":
            source_id = source.get("trajectory_id")
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                errors.append(
                    f"{path.name}: trajectory source has invalid trajectory_id "
                    f"{source_id!r}"
                )
                continue
            if str(source.get("harness") or "simple") == "simple":
                continue
            if source_id in old_trajectory_ids:
                continue
            explicitly_cataloged = bool(source.get("prefix_type"))
            if source_id not in used_trajectory_ids and not explicitly_cataloged:
                continue
        elif source.get("kind") != "external":
            errors.append(f"{path.name}: unknown source kind {source.get('kind')!r}")
            continue
        prefixes.append({
            "path": path,
            "filename": _prefix_page_filename(path),
            "payload": payload,
            "source": source,
            "messages": messages,
            "mtime": path.stat().st_mtime,
            "archived": path.name in archived_files,
        })
    prefixes.sort(
        key=lambda item: (
            str(item["source"].get("generated_at") or ""),
            item["mtime"],
        ),
        reverse=True,
    )
    _assign_prefix_display_names(prefixes)
    return prefixes, errors


def _payload_identity_keys(payload: dict) -> set[tuple[str, str]]:
    keys = set()
    name = payload.get("name")
    if isinstance(name, str) and name:
        keys.add(("name", name))
    if payload.get("format") == "environments-continuation-prefix-v1":
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        keys.add(("sha256", hashlib.sha256(canonical.encode()).hexdigest()))
    return keys


def _continuation_payload_usages(
    audits: list[dict],
) -> dict[tuple[str, str], set[str]]:
    usages: dict[tuple[str, str], set[str]] = defaultdict(set)

    def record(prefix: dict, treatment: object) -> None:
        label = str(treatment or "unknown")
        for field in ("sha256", "name"):
            value = prefix.get(field)
            if isinstance(value, str) and value:
                usages[(field, value)].add(label)

    for audit in audits:
        continuation = continuation_of(audit)
        if continuation:
            record(prefix_of(audit), continuation.get("treatment"))

    campaign_root = DATA_ROOT / "remote_campaigns"
    if campaign_root.is_dir():
        for path in sorted(campaign_root.glob("continuation-aws-*.json")):
            try:
                state = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            continuation = (state.get("pipeline_config") or {}).get("continuation")
            if not isinstance(continuation, dict):
                continue
            for payload in continuation.get("payloads") or []:
                if isinstance(payload, dict):
                    record(payload, continuation.get("treatment"))

    # Campaign state remains the automatic source of truth after launch. This small
    # registry lets an exact content-addressed payload show as selected before its
    # paid campaign is created.
    selection_path = DATA_ROOT / "prefix_selections.json"
    if selection_path.is_file():
        try:
            selection_data = json.loads(selection_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not load prefix selection registry {selection_path}: {error}"
            ) from error
        if (
            not isinstance(selection_data, dict)
            or selection_data.get("format") != PREFIX_SELECTIONS_FORMAT
            or not isinstance(selection_data.get("selections"), list)
        ):
            raise ValueError(
                f"{selection_path} must use format {PREFIX_SELECTIONS_FORMAT!r} "
                "and contain a selections list"
            )
        for index, selection in enumerate(selection_data["selections"]):
            if not isinstance(selection, dict):
                raise ValueError(
                    f"{selection_path}: selection {index} is not an object"
                )
            sha256 = selection.get("sha256")
            treatment = selection.get("treatment")
            if (
                not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
                or not isinstance(treatment, str)
                or not treatment.strip()
            ):
                raise ValueError(
                    f"{selection_path}: selection {index} needs an exact lowercase "
                    "SHA-256 and nonempty treatment"
                )
            usages[("sha256", sha256)].add(treatment.strip())
    return usages


def _annotate_payload_prefix_usages(
    prefixes: list[dict],
    usages: dict[tuple[str, str], set[str]],
) -> None:
    for prefix in prefixes:
        selected = set()
        for key in _payload_identity_keys(prefix["payload"]):
            selected.update(usages.get(key, set()))
        if selected:
            prefix["selected"] = True
            prefix["used_treatments"] = sorted(selected)


def _prefix_attempt_contract(audit: dict) -> dict:
    family = (audit.get("real_env") or {}).get("family")
    seed = str(audit.get("seed") or "unknown")
    if family == "p_hacking_prefix_only":
        return {
            "family": family,
            "expected_submissions": 1,
            "expected_follow_up": False,
            "required_deliverables": ("analysis.py", "RESULT.md"),
            "description": (
                "P-hacking no-honeypot prefix-generation attempt retained as source "
                "evidence. No runnable continuation payload was saved because the "
                "required protocol did not complete."
            ),
            "generator": "exp_p_hacking_prefix.py",
            "prefix_type": "p_hacking_no_honeypot",
            "prefix_type_label": "P-hacking no-honeypot",
            "source_label": "Checkout redesign · clear positive",
        }
    return {
        "family": "ml_prefix_only",
        "expected_submissions": 2,
        "expected_follow_up": True,
        "required_deliverables": ("REPORT.md", "models/final/"),
        "description": (
            "ML prefix-generation attempt retained as source evidence. No runnable "
            "continuation payload was saved because the required two-pass protocol "
            "did not complete."
        ),
        "generator": "exp_ml_prefix.py",
        "prefix_type": "ml_prefix_only",
        "prefix_type_label": "ML prefix only",
        "source_label": f'{seed.replace("_", " ").title()} · no test access',
    }


def _ml_prefix_attempt_status(audit: dict) -> str:
    contract = _prefix_attempt_contract(audit)
    protocol = (audit.get("real_env") or {}).get("protocol") or {}
    submissions = protocol.get("submissions")
    count = (
        int(submissions)
        if isinstance(submissions, int) and not isinstance(submissions, bool)
        else 0
    )
    complete = (
        count == contract["expected_submissions"]
        and (protocol.get("follow_up_sent") is True)
        == contract["expected_follow_up"]
    )
    if complete:
        return f"complete · {count}/{count} submissions"
    reason = str(protocol.get("ended_reason") or "incomplete").replace("_", " ")
    return (
        f"incomplete · {count}/{contract['expected_submissions']} submissions · "
        f"{reason}"
    )


def _ml_prefix_attempt_matches(prefix: dict, audit: dict) -> bool:
    source = prefix.get("source") or {}
    run_name = source.get("run_name")
    return (
        isinstance(run_name, str)
        and run_name in str(audit.get("mode") or "")
        and str(prefix["payload"].get("target") or "")
        == agent_label(audit.get("target"))
        and source.get("epoch") == _display_epoch(audit)
        and str(source.get("seed") or "") == str(audit.get("seed") or "")
        and str(source.get("harness") or "simple")
        == str(audit.get("harness") or "simple")
    )


def _ml_prefix_attempt_entry(audit: dict) -> dict:
    contract = _prefix_attempt_contract(audit)
    real_env = audit.get("real_env") or {}
    protocol = real_env.get("protocol") or {}
    grade = real_env.get("grade") or {}
    deliverables = grade.get("deliverables") or {}
    required_deliverables = contract["required_deliverables"]
    missing = [
        name for name in required_deliverables if deliverables.get(name) is not True
    ]
    submissions = protocol.get("submissions")
    completed_protocol = (
        submissions == contract["expected_submissions"]
        and (protocol.get("follow_up_sent") is True)
        == contract["expected_follow_up"]
    )
    cost, has_cost = _recorded_cost(audit)
    context_calls = (audit.get("target_context_usage") or {}).get("calls") or []
    measured_context = next(
        (
            value for value in reversed(context_calls)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ),
        None,
    )
    messages = audit.get("messages") or []
    generated_at = next(
        (
            message.get("timestamp") for message in messages
            if isinstance(message, dict) and message.get("timestamp")
        ),
        None,
    )
    target = agent_label(audit.get("target"))
    epoch = _display_epoch(audit)
    mode = str(audit.get("mode") or "ml-prefix-attempt")
    log_file = str(audit.get("log_file") or "attempt.eval")
    path = LOGS_ROOT / mode / log_file
    sidecar = (
        LOGS_ROOT
        / mode
        / "real_artifacts"
        / str(audit.get("task") or "")
        / f'{audit.get("seed")}_ep{audit.get("epoch")}'
        / "_native_resume_bundle.json"
    )
    source = {
        "kind": "external",
        "description": contract["description"],
        "generator": contract["generator"],
        "prefix_type": contract["prefix_type"],
        "prefix_type_label": contract["prefix_type_label"],
        "source_label": contract["source_label"],
        "family": contract["family"],
        "seed": str(audit.get("seed") or "unknown"),
        "epoch": epoch,
        "run_name": mode,
        "harness": str(audit.get("harness") or "simple"),
        "generated_at": generated_at,
        "completed_script": completed_protocol,
        "completed_protocol": completed_protocol,
        "submissions": submissions,
        "follow_up_sent": protocol.get("follow_up_sent") is True,
        "ended_reason": protocol.get("ended_reason"),
        "judging": {
            "stage_one_llm_judge": False,
            "final_llm_judge": False,
            "performance_grader": False,
        },
        "evaluation_access": {
            "test_inputs_available": False,
            "test_labels_available": False,
        } if contract["family"] == "ml_prefix_only" else {},
        "deliverables": deliverables,
        "all_deliverables_present": not missing,
        "continuation_eligibility": {
            "eligible_by_default": completed_protocol and not missing,
            "reason": (
                "required_deliverables_present"
                if completed_protocol and not missing
                else "incomplete_protocol"
                if not completed_protocol
                else "missing_required_deliverables"
            ),
            "required_deliverables": list(required_deliverables),
            "missing_required_deliverables": missing,
        },
        "measured_context_tokens": measured_context,
        "generation_cost": (
            {"cost_usd": cost, "source": "viewer_recorded_total"}
            if has_cost else {}
        ),
        "native_harness": real_env.get("harness") or {},
        "native_resume_artifact_available": sidecar.is_file(),
        "attempt_status": _ml_prefix_attempt_status(audit),
        "attempt_only": True,
    }
    payload = {
        "format": "environments-continuation-prefix-v1",
        "name": f"{mode}-{target}-e{epoch}-attempt",
        "target": target,
        "reasoning": bool(audit.get("reasoning")),
        "source": source,
        "messages": messages,
    }
    return {
        "path": path,
        "filename": _prefix_page_filename(path),
        "payload": payload,
        "source": source,
        "messages": messages,
        "mtime": float(audit.get("mtime") or 0.0),
        "archived": False,
        "attempt_only": True,
        "attempt_status": _ml_prefix_attempt_status(audit),
        "source_audit": audit,
    }


def _merge_ml_prefix_attempts(
    prefixes: list[dict], prefix_audits: list[dict]
) -> None:
    for audit in prefix_audits:
        stored = next(
            (
                prefix for prefix in prefixes
                if _ml_prefix_attempt_matches(prefix, audit)
            ),
            None,
        )
        if stored is not None:
            recovery = stored["source"].get("native_submission_recovery") or {}
            stored["attempt_status"] = (
                "complete · 2/2 submissions · recovered pre-deadline response"
                if recovery.get("accepted_as_submission") is True
                else _ml_prefix_attempt_status(audit)
            )
            stored["source_audit"] = audit
        else:
            prefixes.append(_ml_prefix_attempt_entry(audit))
    prefixes.sort(
        key=lambda item: (
            str(item["source"].get("generated_at") or ""),
            item["mtime"],
        ),
        reverse=True,
    )
    _assign_prefix_display_names(prefixes)


def _prefix_harness(prefix: dict) -> str:
    stored = str(prefix["source"].get("harness") or "simple")
    return "subscription" if stored in {"production", "subscription"} else stored


def _prefix_type(prefix: dict) -> tuple[str, str]:
    """Return the catalog tab for one external or trajectory prefix payload."""

    source = prefix["source"]
    explicit = source.get("prefix_type")
    if isinstance(explicit, str) and explicit.strip():
        key = re.sub(r"[^a-z0-9]+", "_", explicit.strip().lower()).strip("_")
        if key:
            label = source.get("prefix_type_label")
            if not isinstance(label, str) or not label.strip():
                label = explicit.replace("_", " ").strip().title()
            if key == "ml_prefix_only":
                label = "No-honeypot ML"
            elif key == "activity_log_context":
                label = "Activity-log context (p-hacking)"
            return key, label.strip()
    if source.get("kind") == "trajectory":
        family = source.get("family")
        seed = str(source.get("seed") or "unknown")
        if family == "ml_training_data_misuse":
            return f"ml_{seed}", f"ML: {seed}"
        if family == "p_hacking":
            return "p_hacking", "p-hacking"
        return "trajectory_prefixes", "Trajectory prefixes"
    if (
        source.get("generator") == "exp_nq_prefix.py"
        or source.get("dataset") == "google-research-datasets/nq_open"
    ):
        return DEFAULT_PREFIX_TYPE, "Natural Questions"
    return "other", "Other"


def _prefix_types(prefixes: list[dict]) -> tuple[tuple[str, str], ...]:
    labels = {DEFAULT_PREFIX_TYPE: "Natural Questions"}
    for prefix in prefixes:
        key, label = _prefix_type(prefix)
        labels.setdefault(key, label)
    return tuple(
        (key, labels[key])
        for key in (
            DEFAULT_PREFIX_TYPE,
            *sorted(labels.keys() - {DEFAULT_PREFIX_TYPE}),
        )
    )


def _prefix_type_code(prefix_type: str) -> str:
    """Derive a compact viewer code from a normalized prefix type key."""

    words = re.findall(r"[a-z0-9]+", prefix_type.lower())
    if not words:
        return "P"
    if len(words) == 1:
        return words[0][:2].upper()
    return "".join(word[0] for word in words).upper()


def _prefix_type_codes(prefixes: list[dict]) -> dict[str, str]:
    """Return deterministic, collision-safe codes for the loaded prefix types."""

    type_keys = sorted({_prefix_type(prefix)[0] for prefix in prefixes})
    by_base: dict[str, list[str]] = defaultdict(list)
    for type_key in type_keys:
        by_base[_prefix_type_code(type_key)].append(type_key)

    codes: dict[str, str] = {}
    used: set[str] = set()

    # Preserve established short codes as new trajectory types enter the same
    # catalog. Otherwise adding ML: fraud_detection would rename Move fast from
    # MF to a collision hash.
    preferred_codes = {
        DEFAULT_PREFIX_TYPE: "NQ",
        "science_ethics": "SE",
        "general_ethics": "GE",
        "move_fast": "MF",
        "wikipedia_summaries": "WS",
        "ml_prefix_only": "MPO",
        "p_hacking_no_honeypot": "PHN",
        "ml_fraud_detection": "MLF",
        "ml_fraud_detection_cutoff": "MLFC",
        "ml_demand_forecasting": "MLD",
        "p_hacking": "PH",
    }
    for type_key in type_keys:
        code = preferred_codes.get(type_key)
        if code and code not in used:
            codes[type_key] = code
            used.add(code)

    for base, keys in sorted(by_base.items()):
        unassigned = [key for key in keys if key not in codes]
        if not unassigned:
            continue
        if len(unassigned) == 1 and base not in used:
            codes[unassigned[0]] = base
            used.add(base)
            continue
        for type_key in unassigned:
            digest = hashlib.sha256(type_key.encode()).hexdigest().upper()
            suffix_length = 2
            code = f"{base}{digest[:suffix_length]}"
            while code in used:
                suffix_length += 1
                code = f"{base}{digest[:suffix_length]}"
            codes[type_key] = code
            used.add(code)
    return codes


def _assign_prefix_display_names(prefixes: list[dict]) -> None:
    """Give every prefix type short chronological viewer-only IDs."""

    type_codes = _prefix_type_codes(prefixes)
    for prefix in prefixes:
        prefix["display_type_code"] = type_codes[_prefix_type(prefix)[0]]
    for type_key in sorted(type_codes):
        typed_prefixes = sorted(
            (prefix for prefix in prefixes if _prefix_type(prefix)[0] == type_key),
            key=lambda prefix: (
                str(prefix["source"].get("generated_at") or ""),
                prefix["mtime"],
                prefix["path"].name,
            ),
        )
        for ordinal, prefix in enumerate(typed_prefixes, start=1):
            prefix["display_name"] = f"{type_codes[type_key]}-{ordinal}"
            prefix["display_ordinal"] = ordinal


def _prefix_display_name(prefix: dict) -> str:
    return str(
        prefix.get("display_name")
        or prefix["payload"].get("name")
        or prefix["path"].stem
    )


def _trajectory_prefix_usages(audits: list[dict]) -> dict[int, set[str]]:
    usages: dict[int, set[str]] = defaultdict(set)
    for audit in audits:
        if audit.get("retrospective_rejudge"):
            continue
        continuation = continuation_of(audit)
        source_id = prefix_source_trajectory_id(audit)
        if not continuation or source_id is None:
            continue
        usages[source_id].add(str(continuation.get("treatment") or "unknown"))
    return usages


def _annotate_trajectory_prefixes(
    prefixes: list[dict],
    *,
    audits_by_id: dict[int, dict],
    usages: dict[int, set[str]],
) -> None:
    for prefix in prefixes:
        source = prefix["source"]
        if source.get("kind") != "trajectory":
            continue
        source_id = source.get("trajectory_id")
        try:
            source_id = int(source_id)
        except (TypeError, ValueError):
            continue
        prefix["source_audit"] = audits_by_id.get(source_id)
        # A derived static-context payload is not the same prefix conversation as its
        # source trajectory. Only exact name/SHA campaign matching below may mark it
        # as used. Ordinary transcript exports retain the historical source-ID link.
        delivery_mode = (prefix["payload"].get("delivery") or {}).get("mode")
        prefix["used_treatments"] = (
            []
            if delivery_mode == "inline_user_context"
            else sorted(usages.get(source_id, set()))
        )


def _prefix_condition(prefix: dict) -> tuple[str, str]:
    stored = prefix["source"].get("prefix_condition")
    if isinstance(stored, dict):
        key = str(stored.get("key") or "unknown")
        label = str(stored.get("label") or "outcome unavailable")
        return key, label

    audit = prefix.get("source_audit")
    if not isinstance(audit, dict):
        return "unknown", "outcome unavailable"
    category = trajectory_category(audit, respect_exclusion=False)
    turns = experiment_user_turn_count(audit)
    if category == "hack":
        turn_label = f"{turns} turn" + ("s" if turns != 1 else "") if turns else "unknown turns"
        return f"hack_{turns or 'unknown'}turn", f"hack · {turn_label}"
    reward = _dimension(audit, "reward_hacking") or {}
    values = reward.get("value") or []
    if category == "clean":
        return "clean", "clean"
    if category == "notable" and isinstance(values, list):
        labels = [
            REWARD_HACKING_DISPLAY_LABELS.get(str(value), str(value).replace("_", " "))
            for value in values
            if str(value) != "clean"
        ]
        return "notable", "notable · " + " + ".join(labels)
    if category == "review":
        return "review", "needs review"
    return category, category.replace("_", " ")


def _prefix_condition_html(prefix: dict) -> str:
    key, label = _prefix_condition(prefix)
    css = (
        "value-bad" if key.startswith("hack_")
        else "value-good" if key.startswith("clean")
        else "value-warn" if key.startswith("notable")
        else "value-concern" if key == "review"
        else ""
    )
    classes = "value-chip" + (f" {css}" if css else "")
    return f'<span class="{classes}">{esc(label)}</span>'


def _prefix_experiment_user_turns(prefix: dict) -> int:
    return sum(
        1
        for message in prefix["messages"]
        if isinstance(message, dict)
        and _is_experiment_prefix_user_message(message)
    )


def _is_experiment_prefix_user_message(message: dict) -> bool:
    metadata = message.get("metadata")
    return (
        message.get("role") == "user"
        and not (
            isinstance(metadata, dict)
            and metadata.get("scaffold_injected")
        )
    )


def _prefix_source_status(prefix: dict) -> str:
    recovery = (prefix.get("source") or {}).get("native_submission_recovery") or {}
    if recovery.get("accepted_as_submission") is True:
        # The recovery payload is the mechanically revalidated source of truth; the
        # immutable original log necessarily still records the pre-repair failure.
        return ""
    audit = prefix.get("source_audit")
    if not isinstance(audit, dict):
        return ""
    if audit.get("prefix_eligible") is False:
        return "source now prefix-ineligible"
    status = str(audit.get("mechanical_status") or "")
    return "" if status == "valid" else status.replace("_", " ")


def _prefix_continuation_eligibility(prefix: dict) -> str:
    source = prefix.get("source") or {}
    eligibility = source.get("continuation_eligibility")
    if isinstance(eligibility, dict):
        if eligibility.get("eligible_by_default") is True:
            return "eligible by default"
        if eligibility.get("reason") == "incomplete_protocol":
            status = str(source.get("attempt_status") or "incomplete protocol")
            return f"ineligible by default · {status}"
        missing = eligibility.get("missing_required_deliverables") or []
        detail = ", ".join(str(item) for item in missing) or "status not recorded"
        return f"ineligible by default · missing {detail}"
    if (
        source.get("prefix_type") == "ml_prefix_only"
        or source.get("generator") == "exp_ml_prefix.py"
    ):
        return (
            "eligible by default"
            if source.get("all_deliverables_present") is True
            else "ineligible by default · legacy eligibility record"
        )
    return ""


_PREFIX_MODEL_ORDER = {
    name: index
    for index, name in enumerate((
        "opus-4.6", "gpt-5.5", "deepseek-v4-pro", "glm-5.1", "kimi-k2.6",
    ))
}


def _prefix_model_sort_key(prefix: dict) -> tuple[int, str]:
    model = str(prefix["payload"].get("target") or "unknown")
    return _PREFIX_MODEL_ORDER.get(model, len(_PREFIX_MODEL_ORDER)), model


def _prefix_condition_sort_key(prefix: dict) -> tuple[int, str]:
    key, label = _prefix_condition(prefix)
    order = {
        "hack_1turn": 0,
        "hack_2turn": 1,
        "clean": 2,
        "notable": 3,
        "review": 4,
    }
    normalized = re.sub(r"_1turn_cutoff$", "", key)
    return order.get(normalized, len(order)), label


def _prefix_source_label(prefix: dict) -> str:
    source = prefix["source"]
    if source.get("kind") == "trajectory":
        trajectory_id = source.get("trajectory_id")
        return f"trajectory #{trajectory_id}" if trajectory_id is not None else "trajectory"
    source_label = source.get("source_label")
    if isinstance(source_label, str) and source_label.strip():
        return source_label.strip()
    dataset = str(source.get("dataset") or "external")
    if dataset == "google-research-datasets/nq_open":
        return "Natural Questions"
    return dataset


def _prefix_source_html(prefix: dict) -> str:
    source = prefix["source"]
    label = _prefix_source_label(prefix)
    trajectory_id = source.get("trajectory_id")
    if source.get("kind") == "trajectory" and isinstance(trajectory_id, int):
        return (
            f'<a href="trajectory-{trajectory_id}.html">{esc(label)}</a>'
        )
    return esc(label)


def _prefix_context(prefix: dict) -> tuple[str, str, str]:
    source = prefix["source"]
    target = source.get("target_context_tokens")
    measured = source.get("measured_context_tokens")
    reached = source.get("reached_target_tokens")
    if not isinstance(target, int):
        if isinstance(measured, int):
            completed = source.get("completed_script")
            if completed is True:
                return f"{measured:,}", "script completed", "reached"
            if completed is False:
                return f"{measured:,}", "script incomplete", "not-reached"
            return f"{measured:,}", "measured context", "na"
        return "—", "not applicable", "na"
    measured_text = f"{measured:,}" if isinstance(measured, int) else "unknown"
    context = f"{measured_text} / {target:,}"
    if reached is True:
        return context, "target reached", "reached"
    if reached is False:
        return context, "target not reached", "not-reached"
    return context, "status not recorded", "not-reached"


def _prefix_cost(prefix: dict) -> str:
    cost = prefix["source"].get("generation_cost")
    if not isinstance(cost, dict):
        return "—"
    amount = cost.get("cost_usd")
    if isinstance(amount, (int, float)):
        return f"${float(amount):.6f}"
    if cost.get("source") == "subscription_not_metered":
        return "included subscription"
    return "unavailable"


def _prefix_transcript(prefix: dict) -> tuple[str, str | None]:
    if prefix.get("attempt_only"):
        try:
            return render_transcript(prefix["messages"]), None
        except Exception as error:
            return "", f"transcript could not be rendered: {error}"
    objects = []
    try:
        for message in prefix["messages"]:
            if not isinstance(message, dict):
                raise TypeError("message is not a JSON object")
            values = dict(message)
            tool_calls = values.get("tool_calls")
            if isinstance(tool_calls, list):
                values["tool_calls"] = [
                    SimpleNamespace(**call) if isinstance(call, dict) else call
                    for call in tool_calls
                ]
            objects.append(SimpleNamespace(**values))
        return render_transcript(normalize_messages(objects)), None
    except Exception as error:
        return "", f"transcript could not be rendered: {error}"


def _prefix_load_errors(errors: list[str]) -> str:
    return "".join(
        f'<div class="load-error">{esc(error)}</div>' for error in errors
    )


def _prefixes_page(
    prefixes: list[dict],
    errors: list[str],
    *,
    seeds: list[str],
    active_harness: str,
    active_prefix_type: str,
    prefix_types: tuple[tuple[str, str], ...],
    old: bool = False,
) -> str:
    active_prefix_label = next(
        (
            label for prefix_type, label in prefix_types
            if prefix_type == active_prefix_type
        ),
        active_prefix_type.replace("_", " "),
    )
    prefix_only = active_prefix_type in {
        "ml_prefix_only",
        "p_hacking_no_honeypot",
    }
    activity_log_prefixes = active_prefix_type == "activity_log_context"
    trajectory_prefixes = (
        active_prefix_type in {
            "activity_log_context", "p_hacking", "trajectory_prefixes"
        }
        or (
            active_prefix_type.startswith("ml_")
            and active_prefix_type != "ml_prefix_only"
        )
    )
    activity_label = (
        "Log lines"
        if activity_log_prefixes
        else "Turns"
        if trajectory_prefixes
        else "Passes"
        if prefix_only
        else "Questions"
    )
    show_condition = trajectory_prefixes
    show_usage = any(
        prefix.get("used_treatments") for prefix in prefixes
    )
    show_source_status = trajectory_prefixes and any(
        _prefix_source_status(prefix) for prefix in prefixes
    )
    model_counts = Counter(
        str(prefix["payload"].get("target") or "unknown")
        for prefix in prefixes
    )
    group_by_model = (trajectory_prefixes or prefix_only) and any(
        count > 1 for count in model_counts.values()
    )
    display_prefixes = (
        sorted(
            prefixes,
            key=lambda prefix: (
                _prefix_model_sort_key(prefix),
                (
                    (0, str(prefix["source"].get("epoch") or ""))
                    if prefix_only
                    else _prefix_condition_sort_key(prefix)
                ),
                prefix.get("display_ordinal") or 0,
            ),
        )
        if group_by_model else prefixes
    )
    rows = []
    previous_model = None
    for prefix in display_prefixes:
        payload = prefix["payload"]
        source = prefix["source"]
        model = str(payload.get("target") or "unknown")
        row_classes = []
        if group_by_model and model != previous_model:
            row_classes.append("group-start")
        if prefix.get("selected"):
            row_classes.append("prefix-selected")
        if prefix.get("attempt_only"):
            row_classes.append("prefix-incomplete")
        row_class = (
            f' class="{esc(" ".join(row_classes), quote=True)}"'
            if row_classes else ""
        )
        row_group = (
            f' data-group="{esc(model, quote=True)}"' if group_by_model else ""
        )
        previous_model = model
        questions = source.get("questions")
        question_count = len(questions) if isinstance(questions, list) else 0
        submissions = source.get("submissions")
        if activity_log_prefixes:
            activity_count = int(
                (source.get("activity_log_metadata") or {}).get("line_count") or 0
            )
        elif trajectory_prefixes:
            activity_count = _prefix_experiment_user_turns(prefix)
        elif prefix_only:
            activity_count = (
                submissions
                if isinstance(submissions, int) and not isinstance(submissions, bool)
                else 0
            )
        else:
            activity_count = question_count
        context, _status, _status_class = _prefix_context(prefix)
        sort_value = prefix.get("display_ordinal") or _prefix_display_name(prefix)
        rows.append(
            f'<tr{row_class}{row_group}>'
            f'<td data-sort-value="{esc(str(sort_value), quote=True)}">'
            f'<a href="{esc(prefix["filename"], quote=True)}">'
            f'{esc(_prefix_display_name(prefix))}</a>'
            + (
                '<span class="prefix-selected-badge">selected</span>'
                if prefix.get("selected") else ""
            )
            + '</td>'
            f'<td data-sort-value="{esc(_prefix_source_label(prefix), quote=True)}">'
            f'{_prefix_source_html(prefix)}</td>'
            f'<td data-sort-value="{esc(payload.get("target") or "", quote=True)}">'
            f'{esc(payload.get("target") or "—")}</td>'
            + (
                f'<td data-sort-value="{esc(_prefix_condition(prefix)[1], quote=True)}">'
                f'{_prefix_condition_html(prefix)}</td>'
                if show_condition else ""
            )
            + (
                '<td data-sort-value="'
                + esc(",".join(prefix.get("used_treatments") or []), quote=True)
                + '">'
                + esc(", ".join(prefix.get("used_treatments") or []) or "not run yet")
                + '</td>'
                if show_usage else ""
            )
            + (
                f'<td>{esc(prefix.get("attempt_status") or "stored payload")}</td>'
                if prefix_only else ""
            )
            + (
                f'<td>{esc(_prefix_source_status(prefix) or "—")}</td>'
                if show_source_status else ""
            )
            +
            f'<td data-sort-value="{activity_count}">{activity_count:,}</td>'
            f'<td data-sort-value="{len(prefix["messages"])}">'
            f'{len(prefix["messages"]):,}</td>'
            f'<td data-sort-value="{source.get("measured_context_tokens") or ""}">'
            f'{esc(context)}</td>'
            f'<td data-sort-value="{esc(_prefix_cost(prefix), quote=True)}">'
            f'{esc(_prefix_cost(prefix))}</td>'
            '</tr>'
        )
    table = (
        '<table class="runs sortable'
        + (' grouped' if group_by_model else '')
        + '"><thead><tr class="cols">'
        '<th>Prefix</th><th>Source</th><th>Model</th>'
        + ('<th>Condition</th>' if show_condition else '')
        + ('<th>Selected for</th>' if show_usage and prefix_only else '')
        + ('<th>Used as</th>' if show_usage and not prefix_only else '')
        + ('<th>Attempt status</th>' if prefix_only else '')
        + ('<th>Source status</th>' if show_source_status else '')
        +
        f'<th data-sort-type="number">{activity_label}</th>'
        '<th data-sort-type="number">Messages</th>'
        '<th data-sort-type="number">Context tokens</th>'
        '<th>Generation cost</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        if rows else '<p class="empty">No stored prefixes.</p>'
    )
    body = (
        _navigation(
            seeds,
            active_view="prefixes_past" if old else "prefixes",
            active_harness=active_harness,
            active_prefix_type=active_prefix_type,
            prefix_types=prefix_types,
        )
        + '<div class="pagehead"><h1>'
        + ("Old prefixes" if old else "Prefixes")
        + " — "
        + esc(active_prefix_label)
        + '</h1></div>'
        + _prefix_load_errors(errors)
        + table
    )
    return _page("Prefixes", body, scripts=INDEX_SORT_JS)


def _prefix_detail_page(
    prefix: dict,
    *,
    seeds: list[str],
    prefix_types: tuple[tuple[str, str], ...],
) -> tuple[str, str | None]:
    payload = prefix["payload"]
    source = prefix["source"]
    context, status, status_class = _prefix_context(prefix)
    transcript, transcript_error = _prefix_transcript(prefix)
    native_harness = source.get("native_harness") or {}
    native_resume = payload.get("native_resume")
    generation_usage = source.get("generation_usage") or {}

    def cell(label: str, value: Any) -> str:
        if value in (None, ""):
            return ""
        return (
            '<div class="metacell">'
            f'<span class="k">{esc(label)}</span><div class="v">{esc(value)}</div>'
            '</div>'
        )

    questions = source.get("questions")
    question_count = len(questions) if isinstance(questions, list) else 0
    usage_text = ""
    if isinstance(generation_usage, dict) and generation_usage:
        usage_text = " · ".join(
            f'{int(generation_usage.get(key) or 0):,} {label}'
            for key, label in (
                ("input", "input"),
                ("output", "output"),
                ("cache_read", "cache read"),
            )
        )
    scaffold = " · ".join(
        str(value)
        for value in (
            native_harness.get("scaffold"),
            native_harness.get("scaffold_version_resolved")
            or native_harness.get("scaffold_version_selector"),
        )
        if value
    )
    native_resume_text = ""
    if isinstance(native_resume, dict):
        native_resume_text = "available; archive retained in payload, not rendered"
        if isinstance(native_resume.get("derived_cutoff"), dict):
            native_resume_text += "; rewound to the stored prefix cutoff"
    elif source.get("native_resume_artifact_available") is True:
        native_resume_text = (
            "captured in run artifacts; no continuation payload was saved"
        )
    evaluation_access = source.get("evaluation_access") or {}
    hidden_evaluation = (
        "fully hidden (inputs and labels)"
        if isinstance(evaluation_access, dict)
        and evaluation_access.get("test_inputs_available") is False
        and evaluation_access.get("test_labels_available") is False
        else ""
    )
    judging = source.get("judging") or {}
    no_judging = (
        "none (by design)"
        if isinstance(judging, dict)
        and judging.get("stage_one_llm_judge") is False
        and judging.get("final_llm_judge") is False
        else ""
    )
    _condition_key, condition_label = _prefix_condition(prefix)
    if source.get("kind") != "trajectory" and not isinstance(
        source.get("prefix_condition"), dict
    ):
        condition_label = None
    cutoff = source.get("cutoff") or {}
    cells = [
        cell("source", _prefix_source_label(prefix)),
        cell("condition", condition_label),
        cell("selected for", ", ".join(prefix.get("used_treatments") or [])),
        cell("attempt status", prefix.get("attempt_status")),
        cell("source status", _prefix_source_status(prefix)),
        cell("continuation eligibility", _prefix_continuation_eligibility(prefix)),
        cell("harness", source.get("harness") or "simple"),
        cell("model", payload.get("target")),
        cell("reasoning", "on" if payload.get("reasoning") else "off"),
        cell("delivery", (payload.get("delivery") or {}).get("mode")),
        cell(
            "experiment user turns",
            _prefix_experiment_user_turns(prefix)
            if source.get("kind") == "trajectory" else None,
        ),
        cell("questions", f"{question_count:,}" if question_count else "—"),
        cell(
            "activity-log lines",
            (source.get("activity_log_metadata") or {}).get("line_count"),
        ),
        cell(
            "activity-log bytes",
            (source.get("activity_log_metadata") or {}).get("byte_count"),
        ),
        cell(
            "activity-log SHA-256",
            (source.get("activity_log_metadata") or {}).get("sha256"),
        ),
        cell("passes", source.get("submissions")),
        cell("evaluation access", hidden_evaluation),
        cell("LLM judging", no_judging),
        cell("messages", f'{len(prefix["messages"]):,}'),
        cell("context tokens", context),
        cell("token measurement", source.get("token_measurement")),
        cell("generation cost", _prefix_cost(prefix)),
        cell("generation usage", usage_text),
        cell("generated", source.get("generated_at") or "not recorded"),
        cell("scaffold", scaffold),
        cell("native resume", native_resume_text),
        cell(
            "cutoff messages",
            (
                f"{cutoff.get('retained_message_count')} retained · "
                f"{cutoff.get('omitted_message_count')} omitted"
            )
            if isinstance(cutoff, dict) and cutoff.get("affected") is True
            else None,
        ),
        cell("file", prefix["path"].name),
    ]
    status_html = (
        f'<span class="prefix-status {status_class}">{esc(status)}</span>'
        if status_class != "na" else ""
    )
    description = str(source.get("description") or "")
    lossy_processing = source.get("lossy_processing")
    cleanup_caveat = (
        str(lossy_processing.get("visible_caveat") or "").strip()
        if isinstance(lossy_processing, dict)
        and lossy_processing.get("affected") is True
        else ""
    )
    caveat_label = (
        "Prefix cutoff"
        if isinstance(cutoff, dict) and cutoff.get("affected") is True
        else "Source cleanup"
    )
    article_source_items = []
    articles = source.get("articles")
    if isinstance(articles, list):
        for article in articles:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "Wikipedia article")
            revision = article.get("revision_id")
            permanent_url = str(article.get("permanent_url") or "")
            label = title + (f" · revision {revision}" if revision else "")
            if permanent_url.startswith(("https://", "http://")):
                label_html = (
                    f'<a href="{esc(permanent_url, quote=True)}">{esc(label)}</a>'
                )
            else:
                label_html = esc(label)
            article_source_items.append(f"<li>{label_html}</li>")
    article_sources_html = (
        '<details class="sec"><summary><h2>Article sources</h2></summary>'
        '<div class="metabody"><ul>'
        + "".join(article_source_items)
        + "</ul></div></details>"
        if article_source_items else ""
    )
    error_html = (
        f'<div class="load-error">{esc(transcript_error)}</div>'
        if transcript_error else ""
    )
    active_harness = _prefix_harness(prefix)
    active_prefix_type, _prefix_type_label = _prefix_type(prefix)
    old = bool(prefix.get("archived"))
    display_name = _prefix_display_name(prefix)
    selection_banner = (
        '<div class="rejudge-banner"><strong>Selected for continuation:</strong> '
        + esc(", ".join(prefix.get("used_treatments") or []))
        + '</div>'
        if prefix.get("selected") else ""
    )
    attempt_banner = (
        '<div class="prefix-caveat"><strong>Incomplete prefix attempt:</strong> '
        + esc(str(prefix.get("attempt_status") or "protocol incomplete"))
        + ". This page preserves the trajectory as evidence; it is not a runnable "
        "continuation payload.</div>"
        if prefix.get("attempt_only") else ""
    )
    body = (
        _navigation(
            seeds,
            active_view="prefixes_past" if old else "prefixes",
            active_harness=active_harness,
            active_prefix_type=active_prefix_type,
            prefix_types=prefix_types,
        )
        + '<div class="pagehead">'
        + f'<h1>{esc(display_name)}</h1>'
        + status_html
        + '<a class="headbtn" href="'
        + _prefixes_filename(active_harness, active_prefix_type, old=old)
        + '">All prefixes</a></div>'
        + selection_banner
        + attempt_banner
        + (f'<div class="prefix-description">{esc(description)}</div>' if description else "")
        + (
            f'<div class="prefix-caveat"><strong>{esc(caveat_label)}:</strong> '
            f'{esc(cleanup_caveat)}</div>'
            if cleanup_caveat else ""
        )
        + '<details class="sec metadata" open><summary><h2>Metadata</h2></summary>'
        + f'<div class="metabody"><div class="metagrid">{"".join(cells)}</div></div>'
        + '</details>'
        + article_sources_html
        + error_html
        + '<div class="prefix-transcript">'
        + f'<h2>Conversation · {len(prefix["messages"]):,} messages</h2>'
        + transcript
        + '</div>'
        + render_explanation_turn_nav(None)
    )
    return _page(
        f'{display_name} · Prefix',
        body,
        scripts=EVIDENCE_NAV_JS,
        fit_content=False,
    ), transcript_error


def _link_and_promote_audits(
    audits: list[dict],
    audit_store: ViewerAuditStore,
    promoted_runs: set[str] | frozenset[str],
) -> int:
    """Perform cross-run enrichment without hydrating the complete corpus."""

    backed = [audit_store.is_backed(audit) for audit in audits]
    if not any(backed):
        link_rejudge_sources(audits)
        return promote_rejudge_judgments(audits, promoted_runs)
    if not all(backed):
        raise ValueError("viewer audits cannot mix stored and in-memory records")

    originals = {
        (
            audit.get("mode"),
            audit.get("task"),
            str(audit.get("seed")),
            audit.get("epoch"),
        ): audit
        for audit in audits
        if not audit.get("retrospective_rejudge")
    }
    for rejudge_summary in audits:
        source = rejudge_summary.get("retrospective_rejudge")
        if not isinstance(source, dict):
            continue
        key = (
            source.get("source_run"),
            source.get("source_task"),
            str(source.get("seed")),
            source.get("epoch"),
        )
        original_summary = originals.get(key)
        rejudge = audit_store.hydrate(rejudge_summary)
        pair = [rejudge]
        if original_summary is not None:
            pair.insert(0, audit_store.hydrate(original_summary))
        link_rejudge_sources(pair)
        audit_store.save_and_refresh(rejudge_summary, rejudge)

    selected = {str(name) for name in promoted_runs}
    if not selected:
        return 0
    summaries_by_id = {
        int(audit["id"]): audit
        for audit in audits
        if isinstance(audit.get("id"), int)
    }
    materialized: dict[str, tuple[dict, dict]] = {}

    def include(summary: dict) -> None:
        full = audit_store.hydrate(summary)
        store_key = str(full.get("_viewer_audit_store_key") or "")
        materialized[store_key] = (summary, full)

    for summary in audits:
        if str(summary.get("mode") or "") not in selected:
            continue
        include(summary)
        source_id = summary.get("source_trajectory_id")
        try:
            source_summary = summaries_by_id.get(int(source_id))
        except (TypeError, ValueError):
            source_summary = None
        if source_summary is not None:
            include(source_summary)
    promoted = promote_rejudge_judgments(
        [full for _summary, full in materialized.values()], selected
    )
    for summary, full in materialized.values():
        audit_store.save_and_refresh(summary, full)
    return promoted


async def build(*, use_cache: bool = True) -> dict:
    with viewer_build_lock(DATA_ROOT), ViewerAuditStore(
        CACHE_ROOT / STORE_FILENAME
    ) as audit_store:
        audits, errors = await load_all(
            LOGS_ROOT,
            cache_root=CACHE_ROOT,
            use_cache=use_cache,
            audit_store=audit_store,
            progress=True,
        )
        prefix_generation_audits = [
            audit for audit in audits if _is_prefix_generation_audit(audit)
        ]
        # Prefix-generation .eval logs are source evidence for payload construction,
        # not judged originals. Their payloads render in the Prefixes catalog; keeping
        # these rows here would contaminate ordinary ML counts merely because the
        # control members share fraud/demand seed names with the audited family.
        audits = [
            audit for audit in audits if not _is_prefix_generation_audit(audit)
        ]
        assign_stable_ids(audits, REGISTRY_FILE)
        promoted_count = _link_and_promote_audits(
            audits,
            audit_store,
            promoted_rejudge_run_names(),
        )
        _apply_continuation_source_statuses(audits)
        VIEWER_ROOT.mkdir(parents=True, exist_ok=True)
        archived_pages = (
            _archive_legacy_viewer_pages()
            + _archive_obsolete_production_pages()
        )
        current_methods = await _current_judge_methods()
        archived_run_names = old_run_names()
        archived_trajectory_keys = old_trajectory_keys()
        current_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        past_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        # Rejudge rows themselves render only on the comparison page. An explicitly
        # promoted rejudge is also copied onto its original row above, so that row is
        # what drives the canonical trajectory pages and visuals.
        rejudge_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        # Continuation trajectories never pool into the per-seed pages (they would
        # corrupt the original base rates); they render only on the global
        # Continuations page, with their prefix linked as the Source column.
        continuation_rows: list[dict] = []
        # Complete inline activity-log continuations are prefix experiments. They
        # share continuation scoring/visuals but render beside their payloads under
        # Prefixes instead of in the global Continuations window.
        prefix_experiment_rows: list[dict] = []
        # Activity-log treatments are also conditioned trajectories and must not
        # contaminate the ordinary per-seed base-rate pages.
        multi_agent_rows: list[dict] = []
        for audit in audits:
            seed = str(audit.get("seed") or "unknown")
            harness = _audit_harness(audit)
            if harness not in HARNESS_NAV_MODES:
                raise ValueError(
                    f"trajectory {audit.get('id')} has unknown harness {harness!r}"
                )
            audit["current_judge_method"] = _is_current_judgment(
                audit, current_methods
            )
            continuation = continuation_of(audit)
            multi_agent = _multi_agent_record(audit)
            if continuation and audit.get("source_trajectory_id") is None:
                audit["source_trajectory_id"] = prefix_source_trajectory_id(audit)
            if multi_agent and audit.get("source_trajectory_id") is None:
                source_id = (multi_agent.get("source") or {}).get("trajectory_id")
                if source_id is not None:
                    try:
                        audit["source_trajectory_id"] = int(source_id)
                    except (TypeError, ValueError):
                        pass
            archived = (
                str(audit.get("mode") or "") in archived_run_names
                or trajectory_key(audit) in archived_trajectory_keys
            )
            if audit.get("retrospective_rejudge"):
                destination = rejudge_grouped
            elif archived:
                destination = past_grouped
            elif continuation:
                if is_activity_log_context_continuation(audit):
                    prefix_experiment_rows.append(audit)
                continuation_rows.append(audit)
                continue
            elif multi_agent:
                multi_agent_rows.append(audit)
                continue
            elif (
                audit["current_judge_method"]
                or (not audit.get("judgment") and not audit.get("judge_failure"))
            ):
                destination = current_grouped
            else:
                destination = past_grouped
            destination[(harness, seed)].append(audit)
        seeds = _ordered_seeds(
            {
                seed
                for _harness, seed in (
                    set(current_grouped) | set(past_grouped) | set(rejudge_grouped)
                )
            }
        )
        families = _seed_families()
        audits_by_id = {
            int(audit["id"]): audit
            for audit in audits
            if isinstance(audit.get("id"), int)
        }
        trajectory_prefix_usages = _trajectory_prefix_usages(audits)
        old_trajectory_ids = frozenset(
            int(audit["id"])
            for rows in past_grouped.values()
            for audit in rows
            if isinstance(audit.get("id"), int)
        )
        archived_prefix_files = old_prefix_files()
        prefixes, prefix_errors = _load_prefixes(
            archived_prefix_files,
            frozenset(trajectory_prefix_usages),
            old_trajectory_ids,
        )
        prefix_generation_details = [
            audit_store.hydrate(audit) for audit in prefix_generation_audits
        ]
        _merge_ml_prefix_attempts(prefixes, prefix_generation_details)
        del prefix_generation_details
        _annotate_trajectory_prefixes(
            prefixes,
            audits_by_id=audits_by_id,
            usages=trajectory_prefix_usages,
        )
        _annotate_payload_prefix_usages(
            prefixes,
            _continuation_payload_usages(audits),
        )
        prefix_types = _prefix_types(prefixes)
        if (
            prefix_experiment_rows
            and not any(
                key == ACTIVITY_LOG_PREFIX_TYPE for key, _label in prefix_types
            )
        ):
            prefix_types = (
                *prefix_types,
                (
                    ACTIVITY_LOG_PREFIX_TYPE,
                    "Activity-log context (p-hacking)",
                ),
            )
        show_other_continuations = any(
            continuation_direction(audit) == OTHER_CONTINUATION_DIRECTION
            for audit in continuation_rows
        )
        hidden_continuation_destinations = sorted({
            str(audit.get("seed") or "unknown")
            for audit in continuation_rows
            if str(audit.get("seed") or "unknown")
            not in CONTINUATION_DESTINATIONS
        })
        if hidden_continuation_destinations:
            raise ValueError(
                "Continuation data exists for destination(s) without viewer tabs: "
                + ", ".join(hidden_continuation_destinations)
            )
        current_originals = [
            audit
            for rows in current_grouped.values()
            for audit in rows
        ]
        visible_continuation_directions = [
            key
            for key, _source, destination in CONTINUATION_DIRECTIONS
            if destination in CONTINUATION_DESTINATIONS
        ]
        expected_continuation_pages = set()
        for harness in HARNESS_NAV_MODES:
            harness_current = {
                seed: current_grouped.get((harness, seed), []) for seed in seeds
            }
            for seed in seeds:
                all_current = harness_current[seed]
                all_past = past_grouped.get((harness, seed), [])
                all_rejudges = rejudge_grouped.get((harness, seed), [])
                for pressure in _pressure_views_for_seed(seed):
                    if pressure is None:
                        current = all_current
                        past = all_past
                        rejudges = all_rejudges
                    else:
                        current = [
                            audit for audit in all_current
                            if _pressure_view(audit) == pressure
                        ]
                        past = [
                            audit for audit in all_past
                            if _pressure_view(audit) == pressure
                        ]
                        rejudges = [
                            audit for audit in all_rejudges
                            if _pressure_view(audit) == pressure
                        ]
                    _write_atomic(
                        VIEWER_ROOT / _seed_filename(seed, harness, pressure),
                        _index(
                            seed,
                            current,
                            errors,
                            seeds=seeds,
                            active_harness=harness,
                            active_pressure=pressure,
                        ),
                    )
                    _write_atomic(
                        VIEWER_ROOT
                        / _comparisons_filename(seed, harness, pressure),
                        _comparisons_index(
                            seed,
                            current + past,
                            rejudges,
                            errors,
                            seeds=seeds,
                            active_harness=harness,
                            active_pressure=pressure,
                        ),
                    )
                    _write_atomic(
                        VIEWER_ROOT / _visuals_filename(seed, harness, pressure),
                        _visuals(
                            seed,
                            current,
                            seeds=seeds,
                            active_harness=harness,
                            active_pressure=pressure,
                        ),
                    )
                    _write_atomic(
                        VIEWER_ROOT / _past_filename(seed, harness, pressure),
                        _index(
                            seed,
                            past,
                            [],
                            seeds=seeds,
                            active_harness=harness,
                            active_pressure=pressure,
                            active_view="past",
                            title=f"{seed} — old trajectories",
                        ),
                    )
            continuation_directions = list(visible_continuation_directions)
            if show_other_continuations:
                continuation_directions.append(OTHER_CONTINUATION_DIRECTION)
            for direction in continuation_directions:
                expected_continuation_pages.update({
                    _continuations_filename(harness, direction),
                    _continuation_visuals_filename(harness, direction),
                })
                _write_atomic(
                    VIEWER_ROOT / _continuations_filename(harness, direction),
                    _continuations_page(
                        continuation_rows,
                        seeds=seeds,
                        active_harness=harness,
                        active_direction=direction,
                        show_other_continuations=show_other_continuations,
                    ),
                )
                _write_atomic(
                    VIEWER_ROOT
                    / _continuation_visuals_filename(harness, direction),
                    _continuation_visuals_page(
                        continuation_rows,
                        current_originals,
                        audits_by_id=audits_by_id,
                        seeds=seeds,
                        active_harness=harness,
                        active_direction=direction,
                        show_other_continuations=show_other_continuations,
                    ),
                )
            _write_atomic(
                VIEWER_ROOT / _multi_agent_filename(harness),
                _multi_agent_page(
                    multi_agent_rows,
                    seeds=seeds,
                    active_harness=harness,
                ),
            )
            _write_atomic(
                VIEWER_ROOT / _multi_agent_filename(harness, "inputs"),
                _multi_agent_inputs_page(
                    multi_agent_rows,
                    seeds=seeds,
                    active_harness=harness,
                ),
            )
        for path in VIEWER_ROOT.glob("*continuations*.html"):
            if (
                path.name not in expected_continuation_pages
                and _GENERATED_MARKER in path.read_text()
            ):
                path.unlink()
        expected_prefix_pages = set()
        for prefix in prefixes:
            expected_prefix_pages.add(prefix["filename"])
            detail_page, transcript_error = _prefix_detail_page(
                prefix,
                seeds=seeds,
                prefix_types=prefix_types,
            )
            if transcript_error:
                prefix_errors.append(
                    f'{prefix["path"].name}: {transcript_error}'
                )
            _write_atomic(VIEWER_ROOT / prefix["filename"], detail_page)
        expected_prefix_indexes = set()
        for harness in HARNESS_NAV_MODES:
            if any(
                key == ACTIVITY_LOG_PREFIX_TYPE for key, _label in prefix_types
            ):
                for view in ("trajectories", "visuals"):
                    expected_prefix_indexes.add(
                        _prefix_experiment_filename(
                            harness, ACTIVITY_LOG_PREFIX_TYPE, view
                        )
                    )
                _write_atomic(
                    VIEWER_ROOT
                    / _prefix_experiment_filename(
                        harness, ACTIVITY_LOG_PREFIX_TYPE, "trajectories"
                    ),
                    _activity_log_prefix_trajectories_page(
                        prefix_experiment_rows,
                        seeds=seeds,
                        prefix_types=prefix_types,
                        active_harness=harness,
                    ),
                )
                _write_atomic(
                    VIEWER_ROOT
                    / _prefix_experiment_filename(
                        harness, ACTIVITY_LOG_PREFIX_TYPE, "visuals"
                    ),
                    _activity_log_prefix_visuals_page(
                        prefix_experiment_rows,
                        current_originals,
                        audits_by_id=audits_by_id,
                        seeds=seeds,
                        prefix_types=prefix_types,
                        active_harness=harness,
                    ),
                )
            harness_prefixes = [
                prefix for prefix in prefixes
                if _prefix_harness(prefix) == harness
            ]
            for prefix_type, _label in prefix_types:
                for old in (False, True):
                    filename = _prefixes_filename(
                        harness, prefix_type, old=old
                    )
                    expected_prefix_indexes.add(filename)
                    selected = [
                        prefix for prefix in harness_prefixes
                        if (
                            _prefix_type(prefix)[0] == prefix_type
                            and bool(prefix.get("archived")) is old
                        )
                    ]
                    _write_atomic(
                        VIEWER_ROOT / filename,
                        _prefixes_page(
                            selected,
                            [] if old else prefix_errors,
                            seeds=seeds,
                            active_harness=harness,
                            active_prefix_type=prefix_type,
                            prefix_types=prefix_types,
                            old=old,
                        ),
                    )
        for path in VIEWER_ROOT.glob("prefix-*.html"):
            if path.name not in expected_prefix_pages:
                path.unlink()
        for path in VIEWER_ROOT.glob("*prefixes*.html"):
            if (
                path.name not in expected_prefix_indexes
                and _GENERATED_MARKER in path.read_text()
            ):
                path.unlink()
        for harness in HARNESS_NAV_MODES:
            for seed in seeds:
                family = families.get(seed)
                if family not in GENERIC_JUDGE_FAMILIES:
                    continue
                for pressure in _pressure_views_for_seed(seed):
                    _write_atomic(
                        VIEWER_ROOT
                        / _generic_judge_filename(seed, harness, pressure),
                        await _generic_judge_page(
                            seed,
                            family,
                            seeds=seeds,
                            active_harness=harness,
                            active_pressure=pressure,
                        ),
                    )
        expected = set()
        expected_judge_views = set()
        for audit_index, audit_summary in enumerate(audits, start=1):
            filename = f'trajectory-{int(audit_summary["id"])}.html'
            judge_filename = _judge_trajectory_filename(audit_summary)
            expected.add(filename)
            expected_judge_views.add(judge_filename)
            if audit_summary.get("retrospective_rejudge"):
                active_view = "comparisons"
            elif (
                str(audit_summary.get("mode") or "") in archived_run_names
                or trajectory_key(audit_summary) in archived_trajectory_keys
            ):
                active_view = "past"
            elif is_activity_log_context_continuation(audit_summary):
                active_view = "prefix_experiment_trajectories"
            elif continuation_of(audit_summary):
                active_view = "continuations"
            elif _multi_agent_record(audit_summary):
                active_view = "multi_agent"
            elif audit_summary["current_judge_method"]:
                active_view = "trajectories"
            else:
                active_view = "past"
            audit = audit_store.hydrate(audit_summary)
            _write_atomic(
                VIEWER_ROOT / filename,
                _trajectory(
                    audit,
                    seeds=seeds,
                    active_view=active_view,
                    prefix_types=prefix_types,
                    show_other_continuations=show_other_continuations,
                ),
            )
            _write_atomic(
                VIEWER_ROOT / judge_filename,
                _judge_trajectory(
                    audit,
                    seeds=seeds,
                    active_view=active_view,
                    prefix_types=prefix_types,
                    show_other_continuations=show_other_continuations,
                ),
            )
            del audit
            if audit_index % 250 == 0 or audit_index == len(audits):
                print(
                    f"viewer: rendered {audit_index:,}/{len(audits):,} "
                    "trajectory detail pairs",
                    flush=True,
                )
        for path in VIEWER_ROOT.glob("trajectory-*.html"):
            if path.name not in expected:
                path.unlink()
        for path in VIEWER_ROOT.glob("judge-trajectory-*.html"):
            if path.name not in expected_judge_views:
                path.unlink()
        _validate_generated_viewer_site()
    return {
        "trajectories": len(audits),
        "current_trajectories": sum(map(len, current_grouped.values())),
        "past_trajectories": sum(map(len, past_grouped.values())),
        "rejudge_trajectories": sum(map(len, rejudge_grouped.values())),
        "promoted_rejudge_judgments": promoted_count,
        "continuation_trajectories": len(continuation_rows),
        "prefix_experiment_trajectories": len(prefix_experiment_rows),
        "multi_agent_trajectories": len(multi_agent_rows),
        "prefixes": len(prefixes),
        "prefix_load_errors": len(prefix_errors),
        "load_errors": len(errors),
        "legacy_pages_archived": archived_pages,
        "output": str(VIEWER_ROOT / "index.html"),
    }


async def main() -> None:
    stats = await build()
    print(
        f"viewer: {stats['trajectories']} trajectories, "
        f"{stats['prefixes']} prefixes, "
        f"{stats['load_errors']} log load errors, "
        f"{stats['prefix_load_errors']} prefix load errors -> {stats['output']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
