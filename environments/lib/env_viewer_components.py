"""Small environment-owned HTML components for judgments and transcripts.

The evidence navigator is intentionally one flat row per dimension.  Its arrows cycle
only through that dimension's cited turns, then scroll and briefly highlight the exact
message.  Artifact citations are separate direct links because they are not turns.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any


DIMENSION_LABELS = {
    "noticed_honeypot": "Honeypot",
    "environment_failure": "Env failure",
    "reward_hacking": "Reward hack",
    "deception": "Deception",
}


def esc(value: Any, *, quote: bool = False) -> str:
    return html.escape(str(value), quote=quote)


def artifact_dom_id(path: str, snapshot: str) -> str:
    digest = hashlib.sha256(f"{path}\0{snapshot}".encode()).hexdigest()[:12]
    return f"artifact-{digest}"


def _state(dimension: dict) -> str:
    if dimension.get("status") != "ok":
        return str(dimension.get("status") or "invalid")
    value = dimension.get("value")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if dimension.get("key") == "reward_hacking":
        return "yes" if value else "no"
    return "neutral"


def render_dimension_navigator(judgment: dict | None) -> str:
    """Render the flat result/evidence rows for one structured judgment.

    Old numeric judgments get a clearly labeled, non-interactive table.  They do not
    acquire invented evidence references or new-schema meanings.
    """
    if not judgment:
        return (
            '<section class="judgment-strip judgment-missing">'
            '<div class="judgment-format">No valid judgment was stored.</div></section>'
        )
    if judgment.get("format") == "legacy_numeric":
        rows = "".join(
            "<tr>"
            f"<th>{esc(DIMENSION_LABELS.get(dim['key'], dim['key'].replace('_', ' ')))}</th>"
            f"<td>{esc(dim.get('display_value', ''))}</td>"
            "</tr>"
            for dim in judgment.get("dimensions") or []
        )
        return (
            '<details class="legacy-judgment"><summary>Legacy numeric judgment '
            '<span class="muted">(shown as stored; not translated)</span></summary>'
            f'<table class="legacy-score-table">{rows}</table></details>'
        )

    rows = []
    for dimension in judgment.get("dimensions") or []:
        key = str(dimension.get("key") or "unknown")
        label = DIMENSION_LABELS.get(key, key.replace("_", " "))
        evidence = dimension.get("evidence") or {}
        message_refs = evidence.get("messages") or []
        targets = [
            f"M{reference['number']}"
            for reference in message_refs
            if isinstance(reference.get("number"), int)
        ]
        missing_targets = {
            f"M{reference['number']}"
            for reference in message_refs
            if reference.get("available") is False
        }
        target_json = esc(json.dumps(targets, separators=(",", ":")), quote=True)
        missing_json = esc(
            json.dumps(sorted(missing_targets), separators=(",", ":")), quote=True
        )

        if targets:
            controls = (
                '<span class="evidence-cycle">'
                f'<button type="button" class="evidence-prev" '
                f'aria-label="previous evidence turn for {esc(label, quote=True)}">&#8249;</button>'
                '<button type="button" class="evidence-current" '
                f'aria-label="open evidence turn for {esc(label, quote=True)}">'
                f'{esc(targets[0])}</button>'
                f'<span class="evidence-position">1/{len(targets)}</span>'
                f'<button type="button" class="evidence-next" '
                f'aria-label="next evidence turn for {esc(label, quote=True)}">&#8250;</button>'
                '</span>'
            )
        else:
            controls = '<span class="evidence-none">no cited turns</span>'

        artifact_links = []
        for artifact in evidence.get("artifacts") or []:
            path = str(artifact.get("path") or "")
            snapshot = str(artifact.get("snapshot") or "")
            artifact_links.append(
                f'<a class="artifact-ref" href="#{artifact_dom_id(path, snapshot)}">'
                f'{esc(path)} · {esc(snapshot)}</a>'
            )
        artifacts = (
            f'<span class="artifact-refs">{"".join(artifact_links)}</span>'
            if artifact_links else ""
        )
        explanation = str(dimension.get("explanation") or "")
        why = (
            '<details class="dimension-why"><summary>why</summary>'
            f'<span>{link_message_refs(explanation)}</span></details>'
            if explanation else ""
        )
        issue_count = len(evidence.get("issues") or [])
        issue = (
            f'<span class="evidence-warning" title="{issue_count} invalid or missing '
            f'evidence reference(s)">&#9888;</span>'
            if issue_count else ""
        )
        rows.append(
            f'<div class="dimension-row" data-evidence-targets="{target_json}" '
            f'data-missing-targets="{missing_json}">'
            f'<span class="dimension-label">{esc(label)}</span>'
            f'<span class="dimension-value state-{_state(dimension)}">'
            f'{esc(dimension.get("display_value", ""))}</span>'
            f'{controls}{artifacts}{why}{issue}</div>'
        )

    schema = str(judgment.get("schema_version") or "unknown schema")
    issue_count = len(judgment.get("issues") or [])
    warning = (
        f'<span class="judgment-warning">&#9888; {issue_count} stored judgment issue(s)</span>'
        if issue_count else ""
    )
    overall = str(judgment.get("overall_explanation") or "")
    overall_html = (
        f'<div class="judgment-overall">{link_message_refs(overall)}</div>'
        if overall else ""
    )
    return (
        '<section class="judgment-strip" aria-label="judgment and evidence navigation">'
        f'<div class="judgment-format">{esc(schema)}{warning}</div>'
        f'{overall_html}{"".join(rows)}</section>'
    )


def render_evidence_caveats(judgment: dict | None, audit: dict | None = None) -> str:
    """Render every stored judge-input omission or degradation in one visible panel.

    The judge builder owns policy caveats.  The loader also supplies upstream target-tool
    and artifact caps because they happened before the judge assembled its prompt.  We
    report only stored facts; absence of a caveat is never converted into "complete".
    """
    audit = audit or {}
    envelope = (judgment or {}).get("envelope") or {}
    evidence = (judgment or {}).get("evidence_scope") or {}
    if not evidence and isinstance(envelope, dict):
        evidence = envelope.get("evidence") or {}
    evidence = evidence if isinstance(evidence, dict) else {}
    caveats = []
    for raw in evidence.get("caveats") or []:
        if not isinstance(raw, dict):
            caveats.append({
                "code": "unparsed_judge_caveat",
                "description": str(raw),
                "source": "judge_builder",
            })
            continue
        caveats.append(dict(raw))

    # Older/future envelopes may put the same queryable list one level higher.
    for raw in envelope.get("caveats") or []:
        if isinstance(raw, dict):
            caveats.append(dict(raw))

    tool_truncations = audit.get("tool_truncations") or []
    if tool_truncations and not any(
        item.get("code") == "tool_output_truncated_upstream" for item in caveats
    ):
        caveats.append({
            "code": "tool_output_truncated_upstream",
            "description": (
                f"{len(tool_truncations)} tool result(s) were capped before the target "
                "and judge could read the full output."
            ),
            "source": "upstream",
        })

    artifacts = ((audit.get("real_env") or {}).get("artifacts") or {}).get("files") or []
    truncated_artifacts = [item for item in artifacts if item.get("truncated")]
    if truncated_artifacts and not any(
        item.get("code") == "artifact_content_truncated_upstream" for item in caveats
    ):
        caveats.append({
            "code": "artifact_content_truncated_upstream",
            "description": (
                f"{len(truncated_artifacts)} stored artifact(s) were capped before judging."
            ),
            "source": "upstream",
            "artifacts": [
                {"path": str(item.get("path") or "unknown"), "snapshot": "final"}
                for item in truncated_artifacts
            ],
        })

    # Exact de-duplication matters when upstream caveats were copied into judge metadata.
    unique = []
    seen = set()
    for item in caveats:
        key = (
            str(item.get("code") or "unknown"),
            str(item.get("description") or ""),
            json.dumps(item.get("messages") or [], sort_keys=True),
            json.dumps(item.get("artifacts") or [], sort_keys=True),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)

    policy = evidence.get("native_reasoning_policy")
    reasoning_messages = evidence.get("native_reasoning_message_count")
    reasoning_blocks = evidence.get("native_reasoning_block_count")
    selection = evidence.get("message_selection")
    builder_truncated = evidence.get("builder_truncated_evidence")
    facts = []
    if policy is not None:
        count_detail = ""
        if isinstance(reasoning_messages, int) and isinstance(reasoning_blocks, int):
            count_detail = (
                f" ({reasoning_blocks} block(s) across {reasoning_messages} message(s))"
            )
        facts.append(f"native reasoning: {policy}{count_detail}")
    if selection is not None:
        facts.append(f"messages: {str(selection).replace('_', ' ')}")
    if builder_truncated is not None:
        facts.append(
            "judge builder truncation: " + ("yes" if builder_truncated else "no")
        )
    facts_html = (
        '<div class="evidence-scope-facts">'
        + "".join(f"<span>{esc(fact)}</span>" for fact in facts)
        + "</div>"
        if facts else ""
    )
    if not unique and not facts:
        return ""

    rows = []
    for item in unique:
        code = str(item.get("code") or "unknown_caveat")
        description = str(item.get("description") or code.replace("_", " "))
        refs = []
        for reference in item.get("messages") or []:
            number = (
                reference.get("number") if isinstance(reference, dict) else reference
            )
            if isinstance(number, int):
                refs.append(f'<a href="#M{number}">[M{number}]</a>')
        for artifact in item.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            path = str(artifact.get("path") or "unknown")
            snapshot = str(artifact.get("snapshot") or "unknown")
            refs.append(
                f'<a href="#{artifact_dom_id(path, snapshot)}">'
                f'{esc(path)} · {esc(snapshot)}</a>'
            )
        ref_html = f'<span class="caveat-refs">{" ".join(refs)}</span>' if refs else ""
        rows.append(
            '<li>'
            f'<code>{esc(code)}</code> {esc(description)} {ref_html}'
            f'<span class="caveat-source">{esc(item.get("source") or "stored")}</span>'
            '</li>'
        )
    return (
        '<section class="evidence-caveats">'
        '<h2>Judge evidence coverage</h2>'
        f'{facts_html}'
        f'{"<ul>" + "".join(rows) + "</ul>" if rows else ""}'
        '</section>'
    )


def link_message_refs(text: str) -> str:
    escaped = esc(text)
    return re.sub(r"\[M(\d+)\]", r'<a href="#M\1">[M\1]</a>', escaped)


def _json_block(value: Any) -> str:
    return esc(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def render_transcript(messages: list[dict]) -> str:
    """Render normalized messages with the exact anchors used by evidence navigation."""
    if not messages:
        return '<p class="empty-transcript">No transcript was stored.</p>'
    rows = []
    for message in messages:
        number = int(message["number"])
        role = str(message.get("role") or "other")
        role_class = re.sub(r"[^a-z]", "", role.lower()) or "other"
        indices = []
        if message.get("assistant_turn"):
            indices.append(f"A{message['assistant_turn']}")
        if message.get("user_turn"):
            indices.append(f"U{message['user_turn']}")
        index_html = "".join(
            f'<span class="turn-index">[{esc(index)}]</span>' for index in indices
        )
        body = f'<pre class="message-text">{esc(message.get("text") or "")}</pre>'
        if message.get("reasoning"):
            body += (
                '<details class="message-reasoning"><summary>reasoning</summary>'
                f'<pre>{esc(message["reasoning"])}</pre></details>'
            )
        for call in message.get("tool_calls") or []:
            body += (
                '<div class="tool-call"><div class="tool-call-head">tool call: '
                f'{esc(call.get("function") or "unknown")}</div>'
                f'<pre>{_json_block(call.get("arguments"))}</pre></div>'
            )
        if message.get("other_content_blocks"):
            body += (
                '<details class="unknown-content"><summary>&#9888; unrecognized content '
                'blocks (preserved)</summary>'
                f'<pre>{_json_block(message["other_content_blocks"])}</pre></details>'
            )
        if message.get("error"):
            body += f'<pre class="message-error">{_json_block(message["error"])}</pre>'
        rows.append(
            f'<article class="message role-{role_class}" id="M{number}" '
            f'data-message-number="{number}">'
            f'<header>[M{number}]{index_html} {esc(role)}</header>{body}</article>'
        )
    return "\n".join(rows)


def render_artifacts(artifacts: list[dict]) -> str:
    """Render stored artifact snapshots using the same IDs as judgment citations."""
    if not artifacts:
        return ""
    rows = []
    for artifact in artifacts:
        path = str(artifact.get("path") or "unknown")
        snapshot = str(
            artifact.get("snapshot") or artifact.get("submission") or "unknown"
        )
        content = artifact.get("content")
        caveats = []
        if artifact.get("truncated"):
            caveats.append("artifact extraction was capped")
        if artifact.get("omitted"):
            caveats.append("artifact content was omitted")
        warning = (
            f'<div class="artifact-warning">&#9888; {esc("; ".join(caveats))}</div>'
            if caveats else ""
        )
        body = (
            f'<pre>{esc(content)}</pre>' if isinstance(content, str)
            else '<p class="muted">No stored text content.</p>'
        )
        rows.append(
            f'<article class="artifact" id="{artifact_dom_id(path, snapshot)}">'
            f'<header>{esc(path)} <span class="muted">{esc(snapshot)}</span></header>'
            f'{warning}{body}</article>'
        )
    return "\n".join(rows)


def render_judge_artifacts(judgment: dict | None) -> str:
    """Render the exact artifact snapshots recorded in official judge metadata."""
    scope = (judgment or {}).get("evidence_scope") or {}
    return render_artifacts(scope.get("artifacts") or [])


EVIDENCE_NAV_CSS = r"""
.judgment-strip{border:1px solid #d8dbe5;border-radius:8px;margin:14px 0;background:#fff}
.judgment-format{padding:5px 9px;color:#747887;font-size:11px;border-bottom:1px solid #eceef3}
.judgment-overall{padding:6px 9px;border-bottom:1px solid #eceef3;font-size:12px;color:#4f5360}
.judgment-warning,.evidence-warning{margin-left:8px;color:#9d6300}
.dimension-row{min-height:32px;display:flex;align-items:center;gap:7px;padding:3px 8px;border-bottom:1px solid #f0f1f5;font-size:13px}
.dimension-row:last-child{border-bottom:0}.dimension-label{width:92px;color:#555b68;flex:0 0 auto}
.dimension-value{min-width:92px;font-weight:650;white-space:nowrap}.state-yes{color:#a33}.state-no{color:#17663b}
.state-missing,.state-invalid{color:#9d6300}.evidence-cycle{display:inline-flex;align-items:center;gap:2px}
.evidence-cycle button{border:1px solid #d8dbe5;background:#f8f9fb;height:23px;min-width:25px;padding:0 5px;cursor:pointer;color:#343945}
.evidence-cycle button:hover{background:#eceff5}.evidence-position{min-width:31px;text-align:center;color:#858997;font-size:11px}
.evidence-none{color:#999;font-size:11px}.artifact-refs{display:flex;gap:4px;flex-wrap:wrap}
.artifact-ref{font-size:11px;border:1px solid #dfe2e9;border-radius:10px;padding:1px 6px;text-decoration:none;color:#4c5570}
.dimension-why{font-size:11px;margin-left:auto;max-width:52%}.dimension-why summary{cursor:pointer;color:#747887}
.dimension-why span{display:block;padding:5px 0;color:#4a4e58}.evidence-flash{animation:evidence-flash 1.4s ease-out}
@keyframes evidence-flash{0%{box-shadow:0 0 0 4px #efc84a;background:#fff8d8}100%{box-shadow:none}}
.legacy-judgment{border:1px solid #d8dbe5;border-radius:8px;padding:8px;margin:14px 0}.muted{color:#858997;font-weight:normal}
.legacy-score-table th{text-align:left;padding-right:20px}.legacy-score-table td{font-variant-numeric:tabular-nums}
.evidence-caveats{border:1px solid #e1bf6a;background:#fff9e9;border-radius:8px;padding:8px 10px;margin:14px 0}
.evidence-caveats h2{font-size:14px;margin:0 0 5px}.evidence-caveats ul{margin:5px 0 0;padding-left:22px;font-size:12px}
.evidence-scope-facts{display:flex;gap:5px;flex-wrap:wrap}.evidence-scope-facts span{font-size:11px;border:1px solid #e1d4ad;border-radius:10px;padding:1px 6px;background:#fff}
.caveat-refs{margin-left:5px}.caveat-source{color:#8d8068;font-size:10px;margin-left:6px}
.message{border:1px solid #dfe2e9;border-radius:7px;margin:9px 0;background:#fff;scroll-margin-top:25px}
.message header,.artifact header{padding:5px 9px;font:600 12px ui-monospace,monospace;background:#f3f4f7;border-bottom:1px solid #e1e3e9}
.message pre,.artifact pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:9px;font:12px/1.45 ui-monospace,monospace}
.turn-index{color:#8b90a0;margin-left:4px}.message-reasoning,.unknown-content{padding:5px 9px;color:#696f7d}
.tool-call{margin:7px 9px;border:1px solid #e0e2e8;border-radius:5px}.tool-call-head{padding:4px 7px;background:#f7f8fa;font-size:11px}
.message-error,.artifact-warning{color:#9b311e}.artifact{border:1px solid #dfe2e9;border-radius:7px;margin:9px 0}.artifact-warning{padding:6px 9px}
@media(max-width:700px){.dimension-row{flex-wrap:wrap}.dimension-label{width:82px}.dimension-why{max-width:100%;margin-left:0;width:100%}}
"""


EVIDENCE_NAV_JS = r"""
<script>
(function () {
  function parse(row, name) {
    try { return JSON.parse(row.getAttribute(name) || "[]"); } catch (_) { return []; }
  }
  function setup(row) {
    var targets = parse(row, "data-evidence-targets");
    var missing = new Set(parse(row, "data-missing-targets"));
    if (!targets.length) return;
    var index = 0;
    var current = row.querySelector(".evidence-current");
    var position = row.querySelector(".evidence-position");
    function update() {
      current.textContent = targets[index];
      position.textContent = (index + 1) + "/" + targets.length;
      current.classList.toggle("missing", missing.has(targets[index]));
      current.title = missing.has(targets[index]) ? "cited turn is missing" : "jump to cited turn";
    }
    function go() {
      var element = document.getElementById(targets[index]);
      update();
      if (!element) return;
      element.scrollIntoView({behavior: "smooth", block: "center"});
      element.classList.remove("evidence-flash");
      void element.offsetWidth;
      element.classList.add("evidence-flash");
    }
    row.querySelector(".evidence-prev").addEventListener("click", function () {
      index = (index - 1 + targets.length) % targets.length; go();
    });
    row.querySelector(".evidence-next").addEventListener("click", function () {
      index = (index + 1) % targets.length; go();
    });
    current.addEventListener("click", go);
    update();
  }
  document.querySelectorAll(".dimension-row").forEach(setup);
})();
</script>
"""
