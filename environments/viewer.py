"""Build the standalone static viewer for real-environment trajectories.

This endpoint is free. It reads this project's Inspect logs and writes HTML under
``mats-local/environments/viewer``. It has no Petri imports or runtime fallback.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
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
    render_evidence_caveats,
    render_judge_artifacts,
    render_transcript,
)
from env_viewer_load import (  # noqa: E402
    assign_stable_ids,
    load_all,
    viewer_build_lock,
)
from env_viewer_visuals import VISUALS_CSS, render_visuals  # noqa: E402
from project_paths import DATA_ROOT, LOGS_ROOT, VIEWER_ROOT  # noqa: E402


REGISTRY_FILE = DATA_ROOT / "trajectory_ids.json"
CACHE_ROOT = DATA_ROOT / "viewer_cache"


BASE_CSS = r"""
:root{color-scheme:light;font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#272b34;background:#f4f5f8}
*{box-sizing:border-box}body{margin:0}a{color:#405e9b}main{max-width:1480px;margin:0 auto;padding:18px}
.top{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}.top h1{font-size:20px;margin:0}.top a{font-size:12px}
.panel{background:#fff;border:1px solid #daddE5;border-radius:9px;margin:12px 0;padding:11px}.panel h2{font-size:15px;margin:0 0 8px}
.seed-block{margin:14px 0}.seed-block>h2{font-size:16px;margin:0 0 5px}.runs{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dfe2e9}
.runs th,.runs td{text-align:left;padding:6px 8px;border-bottom:1px solid #eceef3;font-size:12px}.runs th{color:#707686;background:#f7f8fa;font-weight:600}
.runs tr:last-child td{border-bottom:0}.result{font-weight:650}.result.yes{color:#a13c31}.result.no{color:#17663b}.result.invalid{color:#a06b00}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:7px}.meta-cell{border:1px solid #e4e6eb;border-radius:5px;padding:6px}.meta-key{font-size:10px;color:#7b8190}.meta-value{font-size:12px;overflow-wrap:anywhere}
details.raw summary{font-size:12px;color:#6f7584;cursor:pointer}details.raw pre,.json{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.45 ui-monospace,monospace;background:#f7f8fa;padding:8px;border-radius:5px}
.load-error{border:1px solid #d89689;background:#fff2ef;padding:8px;margin:8px 0;font-size:12px}.empty{color:#7b8190;font-size:12px}
.legacy-banner{border:1px solid #d8bd73;background:#fff9e9;padding:7px;border-radius:6px;font-size:12px}.cost{font-variant-numeric:tabular-nums}
"""


def _page(title: str, body: str, *, scripts: str = "") -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{BASE_CSS}\n{EVIDENCE_NAV_CSS}\n{VISUALS_CSS}</style></head>
<body><main>{body}</main>{scripts}</body></html>"""


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
    values = reward.get("value") or []
    return (", ".join(values) if values else "none", "yes" if values else "no")


def _recorded_cost(audit: dict) -> tuple[float, bool]:
    total = 0.0
    found = False
    for usage in (audit.get("model_usage") or {}).values():
        if isinstance(usage, dict) and isinstance(usage.get("total_cost"), (int, float)):
            total += float(usage["total_cost"])
            found = True
    compute = ((audit.get("real_env") or {}).get("compute") or {})
    vm_cost = compute.get("estimated_vm_cost_usd")
    if isinstance(vm_cost, (int, float)):
        total += float(vm_cost)
        found = True
    return total, found


def _index(audits: list[dict], errors: list[dict]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for audit in audits:
        grouped[str(audit.get("seed") or "unknown")].append(audit)
    blocks = []
    for seed, rows in sorted(grouped.items()):
        cells = []
        for audit in sorted(rows, key=lambda row: int(row.get("id") or 0)):
            result, state = _result_cell(audit)
            cost, has_cost = _recorded_cost(audit)
            cells.append(
                "<tr>"
                f'<td><a href="trajectory-{int(audit["id"])}.html">{int(audit["id"])}</a></td>'
                f'<td>{esc(audit.get("target") or "unknown")}</td>'
                f'<td>{esc(audit.get("mode") or "")}</td>'
                f'<td>{esc(audit.get("epoch"))}</td>'
                f'<td class="result {state}">{esc(result)}</td>'
                f'<td>{esc(audit.get("real_ended_reason") or "unknown")}</td>'
                f'<td class="cost">{f"${cost:.4f}" if has_cost else "—"}</td>'
                "</tr>"
            )
        blocks.append(
            f'<section class="seed-block"><h2>{esc(seed)}</h2>'
            '<table class="runs"><thead><tr><th>ID</th><th>Target</th><th>Run</th>'
            '<th>Epoch</th><th>Reward hack</th><th>End</th><th>Recorded cost</th>'
            f'</tr></thead><tbody>{"".join(cells)}</tbody></table></section>'
        )
    error_html = "".join(
        f'<div class="load-error">{esc(error.get("mode"))}: '
        f'{esc(error.get("error_type"))}: {esc(error.get("error"))}</div>'
        for error in errors
    )
    body = (
        '<div class="top"><h1>Real environments</h1>'
        '<a href="visuals.html">Visuals</a></div>'
        + error_html
        + ("".join(blocks) if blocks else '<p class="empty">No trajectories.</p>')
    )
    return _page("Real environments", body)


def _metadata_panel(audit: dict) -> str:
    result, _state = _result_cell(audit)
    cost, has_cost = _recorded_cost(audit)
    values = {
        "trajectory": audit.get("id"),
        "seed": audit.get("seed"),
        "target": audit.get("target"),
        "judge": audit.get("judge"),
        "run": audit.get("mode"),
        "epoch": audit.get("epoch"),
        "condition": audit.get("condition"),
        "ended": audit.get("real_ended_reason"),
        "reward hack": result,
        "recorded cost": f"${cost:.6f}" if has_cost else "unavailable",
    }
    return '<section class="panel"><div class="meta-grid">' + "".join(
        '<div class="meta-cell">'
        f'<div class="meta-key">{esc(key)}</div><div class="meta-value">{esc(value)}</div>'
        '</div>'
        for key, value in values.items()
    ) + '</div></section>'


def _json_panel(title: str, value: Any) -> str:
    return (
        f'<details class="panel raw"><summary>{esc(title)}</summary>'
        f'<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))}</pre>'
        '</details>'
    )


def _trajectory(audit: dict) -> str:
    judgment = audit.get("judgment")
    legacy = (
        '<div class="legacy-banner">Legacy numeric judgment, shown exactly as stored.</div>'
        if judgment and judgment.get("format") == "legacy_numeric" else ""
    )
    grade = (audit.get("real_env") or {}).get("grade")
    body = (
        '<div class="top"><h1>Trajectory '
        f'{int(audit["id"])}</h1><a href="index.html">Index</a>'
        '<a href="visuals.html">Visuals</a></div>'
        + _metadata_panel(audit)
        + legacy
        + render_dimension_navigator(judgment)
        + render_evidence_caveats(judgment, audit)
        + (_json_panel("Grade", grade) if grade else "")
        + render_judge_artifacts(judgment)
        + '<section class="panel"><h2>Trajectory</h2>'
        + render_transcript(audit.get("messages") or [])
        + '</section>'
        + _json_panel("Stored judgment", judgment)
        + (_json_panel("Load issues", audit.get("load_issues"))
           if audit.get("load_issues") else "")
        + _json_panel("Run metadata", audit.get("real_env") or {})
        + _json_panel("Model usage", audit.get("model_usage") or {})
    )
    return _page(
        f'Trajectory {int(audit["id"])}',
        body,
        scripts=EVIDENCE_NAV_JS,
    )


def _visuals(audits: list[dict]) -> str:
    body = (
        '<div class="top"><h1>Visuals</h1><a href="index.html">Index</a></div>'
        + render_visuals(audits)
    )
    return _page("Real environments · visuals", body)


async def build(*, use_cache: bool = True) -> dict:
    with viewer_build_lock(DATA_ROOT):
        audits, errors = await load_all(
            LOGS_ROOT,
            cache_root=CACHE_ROOT,
            use_cache=use_cache,
        )
        assign_stable_ids(audits, REGISTRY_FILE)
        VIEWER_ROOT.mkdir(parents=True, exist_ok=True)
        _write_atomic(VIEWER_ROOT / "index.html", _index(audits, errors))
        _write_atomic(VIEWER_ROOT / "visuals.html", _visuals(audits))
        expected = set()
        for audit in audits:
            filename = f'trajectory-{int(audit["id"])}.html'
            expected.add(filename)
            _write_atomic(VIEWER_ROOT / filename, _trajectory(audit))
        for path in VIEWER_ROOT.glob("trajectory-*.html"):
            if path.name not in expected:
                path.unlink()
    return {
        "trajectories": len(audits),
        "load_errors": len(errors),
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
