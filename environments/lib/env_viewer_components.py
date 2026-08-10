"""Small environment-owned HTML components for judgments and transcripts.

The stored structured judgment keeps one evidence row per dimension inside Other stuff.
The always-visible transcript uses Petri's floating judge-cited/user-turn navigator.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any

from judgment_semantics import JUDGE_DIMENSION_LABELS, cited_message_numbers

DIMENSION_LABELS = JUDGE_DIMENSION_LABELS

JUDGE_PROMPT_SECTIONS = (
    ("overall", ("Overall judge instructions",)),
    ("dimensions", ("Dimension rubrics",)),
    ("trajectory", ("Numbered observable trajectory",)),
    ("artifacts", ("Artifact snapshots",)),
)

LEGACY_SCOPE_HEADINGS = ("Evidence caveats", "Official evidence scope")


def esc(value: Any, *, quote: bool = False) -> str:
    return html.escape(str(value), quote=quote)


def artifact_dom_id(path: str, snapshot: str) -> str:
    digest = hashlib.sha256(f"{path}\0{snapshot}".encode()).hexdigest()[:12]
    return f"artifact-{digest}"


def _cited_message_numbers(text: str, candidates: set[int]) -> set[int]:
    """Return candidate message numbers named singly, in groups, or by ranges."""

    return cited_message_numbers(text, candidates)


def split_stored_judge_prompt(prompt: str) -> dict[str, str] | None:
    """Split only at the fixed headings inside the exact stored prompt.

    Rubric Markdown can contain arbitrary headings, so a generic Markdown parser would
    be unsafe here. A format mismatch returns ``None`` instead of reconstructing a view
    from current files and pretending it is what the judge saw.
    """

    if not isinstance(prompt, str):
        return None
    first_key, first_headings = JUDGE_PROMPT_SECTIONS[0]
    first_heading = first_headings[0]
    prefix = f"# {first_heading}\n\n"
    if not prompt.startswith(prefix):
        return None
    sections: dict[str, str] = {}
    current_key = first_key
    remaining = prompt[len(prefix):]
    layout = list(JUDGE_PROMPT_SECTIONS[1:])
    if any(f"\n\n# {heading}\n\n" in remaining for heading in LEGACY_SCOPE_HEADINGS):
        layout.insert(1, ("legacy_scope", LEGACY_SCOPE_HEADINGS))
    for next_key, next_headings in layout:
        matches = [
            (remaining.find(delimiter), delimiter)
            for next_heading in next_headings
            if (delimiter := f"\n\n# {next_heading}\n\n") in remaining
        ]
        if not matches:
            return None
        position, delimiter = min(matches, key=lambda match: match[0])
        body = remaining[:position]
        remaining = remaining[position + len(delimiter):]
        sections[current_key] = body
        current_key = next_key
    sections[current_key] = remaining
    return sections


def _split_dimensions(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^## Dimension: ([^\n]+)\n\n", text))
    if not matches or matches[0].start() != 0:
        return []
    dimensions = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        if index + 1 < len(matches) and body.endswith("\n\n"):
            body = body[:-2]
        dimensions.append((match.group(1), body))
    return dimensions


def _split_artifacts(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^\[A\d+\] path=", text)]
    if not starts or starts[0] != 0:
        return []
    chunks = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end]
        if index + 1 < len(starts) and chunk.endswith("\n\n"):
            chunk = chunk[:-2]
        chunks.append(chunk)
    return chunks


def _exact_text(text: str) -> str:
    return f'<pre class="judge-exact">{esc(text)}</pre>'


def _nested_detail(summary: str, body: str, *, css_class: str = "") -> str:
    class_name = f"judge-subsection {css_class}".strip()
    return (
        f'<details class="{class_name}"><summary>{esc(summary)}</summary>'
        f'<div class="judge-subsection-body">{body}</div></details>'
    )


def _trajectory_scope_marker(
    judgment: dict,
    audit: dict,
    *,
    trajectory_href: str = "#trajectory-record",
) -> tuple[str, str]:
    envelope = judgment.get("envelope") or {}
    evidence = envelope.get("evidence") or {}
    source_map = evidence.get("source_message_map") or []
    count = len(source_map) if isinstance(source_map, list) else 0
    saved_count = len(audit.get("messages") or [])
    source_count = evidence.get("source_message_count")
    omitted_count = evidence.get("omitted_message_count")
    stage = str(envelope.get("official_stage") or envelope.get("stage") or "unknown")
    selection = evidence.get("message_selection")
    complete = selection == "complete_observable_trajectory_for_stage"
    if complete and stage == "stage_1":
        scope = "complete trajectory through stage one"
    elif complete:
        scope = "complete observable trajectory"
    elif (
        selection
        == "system_user_and_assistant_content_without_tool_usage_for_stage"
    ):
        scope = "system, user, and assistant content without tool usage"
    else:
        scope = str(selection or "scope not recorded").replace("_", " ")

    labels = f"M1–M{count}" if count > 1 else "M1" if count == 1 else "no messages"
    summary = f"Trajectory · {scope} · {labels}"
    reasoning_policy = str(evidence.get("native_reasoning_policy") or "not recorded")
    reasoning_messages = evidence.get("native_reasoning_message_count")
    reasoning_blocks = evidence.get("native_reasoning_block_count")
    plaintext_reasoning = evidence.get("native_reasoning_plaintext_block_count")
    summarized_reasoning = evidence.get("native_reasoning_summary_block_count")
    unavailable_reasoning = evidence.get("native_reasoning_unavailable_block_count")
    if isinstance(reasoning_messages, int) and isinstance(reasoning_blocks, int):
        reasoning_detail = (
            f"{reasoning_policy} · {reasoning_blocks} block(s) in "
            f"{reasoning_messages} message(s)"
        )
        if all(isinstance(value, int) for value in (
            plaintext_reasoning, summarized_reasoning, unavailable_reasoning,
        )):
            reasoning_detail += (
                f" · {plaintext_reasoning} plaintext, {summarized_reasoning} "
                f"summary-only, {unavailable_reasoning} unavailable"
            )
    else:
        reasoning_detail = reasoning_policy

    tool_call_detail = str(evidence.get("tool_calls_policy") or "not recorded")
    source_tool_calls = evidence.get("source_tool_call_count")
    embedded_tool_uses = evidence.get("source_embedded_tool_use_block_count")
    if isinstance(source_tool_calls, int) and isinstance(embedded_tool_uses, int):
        tool_call_detail += (
            f" · {source_tool_calls} ordinary call(s), "
            f"{embedded_tool_uses} embedded block(s) in source"
        )
    tool_result_detail = str(evidence.get("tool_results_policy") or "not recorded")
    source_tool_results = evidence.get("source_tool_result_message_count")
    if isinstance(source_tool_results, int):
        tool_result_detail += f" · {source_tool_results} message(s) in source"

    policies = [
        ("Messages", f"{scope}; judge labels {labels}"),
        ("Native reasoning", reasoning_detail),
        ("System messages", str(evidence.get("system_messages_policy") or "not recorded")),
        ("Assistant-visible text", str(
            evidence.get("assistant_visible_text_policy") or "not recorded"
        )),
        ("Tool calls", tool_call_detail),
        ("Tool results", tool_result_detail),
    ]
    if isinstance(omitted_count, int) and omitted_count:
        policies.append((
            "Messages omitted by policy",
            f"{omitted_count} source message(s)",
        ))
    stage_source_count = source_count if isinstance(source_count, int) else count
    if saved_count and stage_source_count < saved_count:
        policies.append((
            "Later saved messages",
            f"not part of this {stage} judgment "
            f"({saved_count - stage_source_count} message(s))",
        ))
    facts = "".join(
        '<div class="judge-scope-fact">'
        f'<span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in policies
    )
    body = (
        f'<div class="judge-scope-grid">{facts}</div>'
        f'<a class="judge-trajectory-link" href="{esc(trajectory_href, quote=True)}">'
        'Open the saved trajectory</a>'
    )
    return summary, body


def _render_provider_interface(envelope: dict) -> str:
    provider = envelope.get("provider_request")
    if isinstance(provider, dict):
        tools = provider.get("tools") or []
        tool_rows = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "answer tool")
            tool_rows.append(_nested_detail(
                f"{name}() · required",
                _exact_text(json.dumps(tool, ensure_ascii=False, indent=2, sort_keys=True)),
                css_class="judge-tool",
            ))
        controls = {
            key: value
            for key, value in provider.items()
            if key not in {"initial_messages", "tools"}
        }
        return "".join(tool_rows) + _nested_detail(
            "Response mechanics",
            _exact_text(json.dumps(controls, ensure_ascii=False, indent=2, sort_keys=True)),
            css_class="judge-controls",
        )

    # Early environment-judge logs stored the declared schema but not Scout's resolved
    # provider tool. Preserve that record without claiming it is the exact API shape.
    schema = envelope.get("output_schema")
    if schema is None:
        return '<div class="judge-unavailable">No response-interface record was stored.</div>'
    return (
        '<div class="judge-unavailable">Exact provider tool record was not stored.</div>'
        + _exact_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True))
    )


def _render_dimension_prompt(text: str) -> tuple[str, int | str]:
    dimension_parts = _split_dimensions(text)
    if not dimension_parts:
        return _exact_text(text), "stored prompt"
    return "".join(
        _nested_detail(
            DIMENSION_LABELS.get(key, key.replace("_", " ")),
            _exact_text(content),
            css_class="judge-dimension",
        )
        for key, content in dimension_parts
    ), len(dimension_parts)


def _variable_judge_slot(label: str, value: str) -> str:
    return (
        '<div class="judge-variable-slot">'
        f'<span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
    )


def render_generic_judge_stage(
    *,
    prompt: str,
    envelope: dict,
    summary_label: str | None = None,
) -> str:
    """Render current shared prompt bytes with per-trajectory slots made explicit."""

    sections = split_stored_judge_prompt(prompt)
    stage = str(envelope.get("stage") or "unknown")
    family = str(envelope.get("family") or "unknown")
    label = summary_label or stage.replace("_", " ")
    if sections is None:
        return (
            '<details class="judge-view judge-view-unavailable"><summary>'
            f'<span>{esc(label)}</span><span class="judge-view-pills">'
            'current template unavailable</span></summary></details>'
        )

    dimensions, dimension_count = _render_dimension_prompt(sections["dimensions"])
    evidence = envelope.get("evidence") or {}
    selection = str(evidence.get("message_selection") or "scope not recorded")
    reasoning = str(evidence.get("native_reasoning_policy") or "not recorded")
    if selection == "complete_observable_trajectory_for_stage":
        trajectory_slot = (
            f"complete observable messages for this stage; native reasoning {reasoning}"
        )
    elif selection == "user_turns_and_assistant_submissions_for_stage":
        trajectory_slot = (
            "user turns and assistant submission turns for this stage; "
            f"native reasoning {reasoning}"
        )
    elif (
        selection
        == "system_user_and_assistant_content_without_tool_usage_for_stage"
    ):
        trajectory_slot = (
            "system and user messages plus assistant reasoning and visible text for "
            f"this stage; tool calls and results excluded; native reasoning {reasoning}"
        )
    else:
        trajectory_slot = f"{selection.replace('_', ' ')}; native reasoning {reasoning}"
    body = "".join([
        _nested_detail("Overall instructions", _exact_text(sections["overall"])),
        _nested_detail(
            f"Dimensions · {dimension_count}",
            dimensions,
            css_class="judge-section-group",
        ),
        _nested_detail(
            "Numbered trajectory",
            _variable_judge_slot(
                "Trajectory-specific",
                trajectory_slot,
            ),
        ),
        _nested_detail(
            "Artifact snapshots",
            _variable_judge_slot(
                "Trajectory-specific",
                "the exact snapshots available at this stage are inserted for this call",
            ),
        ),
        _nested_detail(
            "Required response",
            _render_provider_interface(envelope),
            css_class="judge-section-group",
        ),
    ])
    return (
        '<details class="judge-view generic-judge-stage"><summary>'
        f'<span>{esc(label)}</span>'
        f'<span class="judge-view-pills">{esc(family)} · current</span></summary>'
        f'<div class="judge-view-body">{body}</div></details>'
    )


def render_judge_view(
    judgment: dict | None,
    audit: dict | None = None,
    *,
    expanded: bool = False,
    trajectory_href: str = "#trajectory-record",
) -> str:
    """Render the stored judge input as compact, nested closed sections.

    The exact numbered trajectory stays inside this view. The broader saved transcript
    remains a separate record because it can contain reasoning or later-stage messages
    that were not part of this judge call. Historical evidence-scope sections are parsed
    for layout compatibility but hidden; their stored issues surface as viewer flags.
    """

    audit = audit or {}
    open_attr = " open" if expanded else ""
    if not judgment:
        return ""
    if judgment.get("format") == "legacy_numeric":
        real_env = audit.get("real_env") or {}
        replay = real_env.get("judge_replay") or {}
        stage_one_prompt = replay.get("stage1_prompt")
        final_render = real_env.get("final_render")
        stored_sections = []
        if isinstance(stage_one_prompt, str) and stage_one_prompt:
            stored_sections.append(_nested_detail(
                "Stage-one prompt · exact stored bytes",
                _exact_text(stage_one_prompt),
            ))
        if isinstance(final_render, str) and final_render:
            stored_sections.append(_nested_detail(
                "Final evidence rendering · exact stored bytes",
                _exact_text(final_render),
            ))
        if stored_sections:
            availability = (
                '<div class="judge-unavailable">This legacy incremental run stored '
                'the sections shown below. It did not store the complete final provider '
                'request or response interface, so the viewer does not reconstruct those '
                'missing parts from current code.</div>'
            )
            return (
                f'<details class="judge-view"{open_attr}><summary><span>Judge view</span>'
                '<span class="judge-view-pills">legacy incremental · stored evidence'
                '</span></summary><div class="judge-view-body">'
                + availability + "".join(stored_sections) + '</div></details>'
            )
        return (
            f'<details class="judge-view judge-view-unavailable"{open_attr}><summary>'
            '<span>Judge view</span><span class="judge-view-pills">legacy · '
            'exact evidence not stored</span></summary>'
            '<div class="judge-unavailable">This historical numeric run did not store '
            'its judge replay evidence or response interface. The viewer does not rebuild '
            'them from current code or rubric files.</div></details>'
        )
    if judgment.get("format") != "structured":
        return ""
    envelope = judgment.get("envelope") or {}
    prompt = envelope.get("prompt_passed_to_scout")
    sections = split_stored_judge_prompt(prompt) if isinstance(prompt, str) else None
    stage = str(envelope.get("official_stage") or envelope.get("stage") or "unknown")
    reasoning = ((envelope.get("evidence") or {}).get("native_reasoning_policy"))
    if sections is None:
        return (
            f'<details class="judge-view judge-view-unavailable"{open_attr}><summary>'
            f'<span>Judge view</span><span class="judge-view-pills">{esc(stage)} · '
            'exact prompt unavailable</span></summary>'
            '<div class="judge-unavailable">The exact stored judge prompt could not be '
            'rendered. No prompt was reconstructed from current rubric files.</div></details>'
        )

    dimensions, dimension_count = _render_dimension_prompt(sections["dimensions"])

    artifact_parts = _split_artifacts(sections["artifacts"])
    stored_artifacts = ((envelope.get("evidence") or {}).get("artifacts") or [])
    if artifact_parts:
        artifacts = []
        for index, content in enumerate(artifact_parts):
            artifact = stored_artifacts[index] if index < len(stored_artifacts) else {}
            path = str(artifact.get("path") or f"Artifact {index + 1}")
            snapshot = str(artifact.get("snapshot") or "")
            label = f"{path} · {snapshot}" if snapshot else path
            anchor = artifact_dom_id(path, snapshot) if snapshot else f"judge-artifact-{index + 1}"
            artifacts.append(
                f'<details class="judge-subsection judge-artifact" id="{anchor}">'
                f'<summary>{esc(label)}</summary><div class="judge-subsection-body">'
                f'{_exact_text(content)}</div></details>'
            )
        artifact_body = "".join(artifacts)
    else:
        artifact_body = _exact_text(sections["artifacts"])

    trajectory_summary, trajectory_body = _trajectory_scope_marker(
        judgment,
        audit,
        trajectory_href=trajectory_href,
    )
    call_mode = (
        "stage-one result reused"
        if envelope.get("reused_stage_one") is True
        else "fresh call"
        if envelope.get("fresh_call") is True
        else None
    )
    judgment_role = (
        "retrospective rejudge"
        if envelope.get("judgment_role") == "retrospective_rejudge"
        else None
    )
    pills = " · ".join(item for item in (
        judgment_role,
        stage,
        call_mode,
        f"reasoning {reasoning}" if reasoning else None,
    ) if item)
    body = "".join([
        _nested_detail("Overall instructions", _exact_text(sections["overall"])),
        _nested_detail(
            f"Dimensions · {dimension_count}",
            dimensions,
            css_class="judge-section-group",
        ),
        _nested_detail(
            trajectory_summary,
            trajectory_body + _exact_text(sections["trajectory"]),
            css_class="judge-trajectory-scope",
        ),
        _nested_detail(
            f"Artifacts · {len(artifact_parts)} snapshot(s)",
            artifact_body,
            css_class="judge-section-group",
        ),
        _nested_detail(
            "Required response",
            _render_provider_interface(envelope),
            css_class="judge-section-group",
        ),
    ])
    return (
        f'<details class="judge-view"{open_attr}><summary><span>Judge view</span>'
        f'<span class="judge-view-pills">{esc(pills)}</span></summary>'
        f'<div class="judge-view-body">{body}</div></details>'
    )


def _state(dimension: dict) -> str:
    if dimension.get("status") != "ok":
        return str(dimension.get("status") or "invalid")
    if dimension.get("key") == "noticed":
        return "neutral"
    value = dimension.get("value")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if dimension.get("key") == "reward_hacking":
        if dimension.get("requires_review"):
            return "review"
        return "yes" if dimension.get("is_hack") else "no"
    return "neutral"


def render_dimension_navigator(
    judgment: dict | None, *, artifact_page_href: str | None = None
) -> str:
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
                f'<a class="artifact-ref" href="'
                f'{esc((artifact_page_href or "") + "#" + artifact_dom_id(path, snapshot), quote=True)}">'
                f'{esc(path)} · {esc(snapshot)}</a>'
            )
        artifacts = (
            f'<span class="artifact-refs">{"".join(artifact_links)}</span>'
            if artifact_links else ""
        )
        explanation = str(dimension.get("explanation") or "")
        why = (
            '<details class="dimension-why"><summary>why</summary>'
            f'<span>{link_judge_message_refs(explanation, judgment, artifact_page_href=artifact_page_href)}</span></details>'
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
    return (
        '<section class="judgment-strip" aria-label="judgment and evidence navigation">'
        f'<div class="judgment-format">{esc(schema)}{warning}</div>'
        f'{"".join(rows)}</section>'
    )


def render_explanation_turn_nav(judgment: dict | None) -> str:
    """Render Petri's floating judge-cited and user-turn groups.

    Current judge citations are prompt-local. The stored source map converts them back
    to the complete saved transcript before the navigator is written.
    """

    targets: list[str] = []
    if judgment:
        text = "\n".join(
            str(judgment.get(key) or "")
            for key in ("summary", "explanation", "highlights")
        )
        source_map = ((judgment.get("envelope") or {}).get("evidence") or {}).get(
            "source_message_map"
        ) or []
        mapped = {
            int(item["number"]): int(item["source_index"]) + 1
            for item in source_map
            if isinstance(item, dict)
            and isinstance(item.get("number"), int)
            and isinstance(item.get("source_index"), int)
        }
        candidates = set(mapped) | {
            int(number) for number in re.findall(r"\bM(\d+)\b", text)
        }
        seen = set()
        for number in sorted(_cited_message_numbers(text, candidates)):
            target = f"M{mapped.get(number, number)}"
            if target not in seen:
                seen.add(target)
                targets.append(target)

    cited_group = ""
    if targets:
        target_json = esc(json.dumps(targets, separators=(",", ":")), quote=True)
        cited_group = (
            '<div class="cnav-grp cited" id="grp-cited"><b>judge-cited</b>'
            f'<div class="explanation-nav-row cnav-row" '
            f'data-explanation-targets="{target_json}">'
            '<button type="button" class="explanation-prev" '
            'aria-label="previous judge-cited turn">&larr;</button>'
            f'<span class="explanation-position lbl">0 / {len(targets)}</span>'
            '<button type="button" class="explanation-next" '
            'aria-label="next judge-cited turn">&rarr;</button>'
            '</div></div>'
        )
    user_group = (
        '<div class="cnav-grp user" id="grp-user">'
        '<b>user turns <span id="user-cnt"></span></b>'
        '<div class="cnav-row">'
        '<button type="button" id="user-prev" title="previous">&larr;</button>'
        '<span class="lbl" id="user-lbl">&ndash;</span>'
        '<button type="button" id="user-next" title="next">&rarr;</button>'
        '</div></div>'
    )
    return (
        '<nav class="cnav explanation-nav" aria-label="trajectory turns">'
        f'{cited_group}{user_group}</nav>'
    )


def link_message_refs(text: str) -> str:
    escaped = esc(text)
    return re.sub(r"\[M(\d+)\]", r'<a href="#M\1">[M\1]</a>', escaped)


def _artifact_anchor(artifacts: list, number: int) -> str | None:
    """The DOM id of the [A<number>] snapshot section rendered by render_judge_view."""

    if not 1 <= number <= len(artifacts):
        return None
    artifact = artifacts[number - 1] if isinstance(artifacts[number - 1], dict) else {}
    path = str(artifact.get("path") or f"Artifact {number}")
    snapshot = str(artifact.get("snapshot") or "")
    return artifact_dom_id(path, snapshot) if snapshot else f"judge-artifact-{number}"


def link_judge_message_refs(
    text: str,
    judgment: dict,
    *,
    artifact_page_href: str | None = None,
) -> str:
    """Link simple, grouped, and ranged judge citations to saved trajectory turns.

    [M#] references link to transcript messages; [A#] references link to the matching
    artifact snapshot inside the Judge view. Unknown artifact numbers stay plain text.
    """

    evidence = (judgment.get("envelope") or {}).get("evidence") or {}
    source_map = evidence.get("source_message_map") or []
    artifacts = evidence.get("artifacts") or []
    targets = {
        int(item["number"]): int(item["source_index"]) + 1
        for item in source_map
        if isinstance(item, dict)
        and isinstance(item.get("number"), int)
        and isinstance(item.get("source_index"), int)
    }
    escaped = esc(text)

    def link_token(kind: str, number: int, label: str) -> str:
        if kind == "M":
            return f'<a href="#M{targets.get(number, number)}">{label}</a>'
        anchor = _artifact_anchor(artifacts, number)
        href = f'{artifact_page_href or ""}#{anchor}' if anchor else None
        return f'<a href="{esc(href, quote=True)}">{label}</a>' if href else label

    def replace_group(match: re.Match) -> str:
        content = match.group(1)
        exact = re.fullmatch(r"([MA])(\d+)", content)
        if exact:
            kind, number = exact.group(1), int(exact.group(2))
            return link_token(kind, number, f"[{kind}{number}]")

        def replace_number(number_match: re.Match) -> str:
            kind, number = number_match.group(1), int(number_match.group(2))
            return link_token(kind, number, f"{kind}{number}")

        linked = re.sub(r"\b([MA])(\d+)\b", replace_number, content)
        return f"[{linked}]"

    return re.sub(r"\[([^\]\n]*\b[MA]\d+[^\]\n]*)\]", replace_group, escaped)


def render_judge_narrative(
    judgment: dict | None, *, artifact_page_href: str | None = None
) -> str:
    """Render stored judge prose with Petri's shared three-section layout."""

    if not judgment:
        return ""
    if judgment.get("format") == "legacy_numeric":
        record = judgment.get("legacy") or {}
    elif judgment.get("format") == "structured":
        record = judgment
    else:
        return ""
    sections = []
    for key, label, note_class in (
        ("summary", "Judge summary", "note"),
        ("explanation", "Judge explanation", "note explanation"),
        ("highlights", "Judge highlights", "note hl"),
    ):
        value = str(record.get(key) or "")
        if not value:
            continue
        sections.append(
            '<details class="sec" open>'
            f'<summary><h2>{esc(label)}</h2></summary>'
            f'<div class="{note_class}">'
            f'{link_judge_message_refs(value, judgment, artifact_page_href=artifact_page_href)}</div></details>'
        )
    return "".join(sections)


def _json_block(value: Any) -> str:
    return esc(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def render_transcript(messages: list[dict]) -> str:
    """Render normalized messages with Petri's transcript markup and styling."""
    if not messages:
        return '<p class="empty-transcript">No transcript was stored.</p>'
    rows = []
    for message in messages:
        number = int(message["number"])
        role = str(message.get("role") or "other")
        role_class = re.sub(r"[^a-z]", "", role.lower().split()[0]) or "other"
        if role_class not in {"system", "user", "assistant", "tool"}:
            role_class = "other"
        indices = []
        if message.get("assistant_turn"):
            indices.append(f"A{message['assistant_turn']}")
        if message.get("user_turn"):
            indices.append(f"U{message['user_turn']}")
        index_html = "".join(
            f' <span class="aturn">[{esc(index)}]</span>' for index in indices
        )
        elapsed_time = message.get("elapsed_time")
        timestamp = message.get("timestamp")
        timestamp_title = (
            f' title="{esc(timestamp, quote=True)}"' if timestamp else ""
        )
        time_html = (
            f' <span class="mtime"{timestamp_title}>{esc(elapsed_time)}</span>'
            if elapsed_time
            else ""
        )
        body = ""
        if message.get("text"):
            body += f'<pre>{esc(message["text"])}</pre>'
        if message.get("reasoning"):
            body += (
                '<details class="think"><summary class="thead">[Agent reasoning]</summary>'
                f'<pre>{esc(message["reasoning"])}</pre></details>'
            )
        for call in message.get("tool_calls") or []:
            body += (
                '<div class="toolcall"><div class="tchead">tool call · '
                f'<span class="tcname">{esc(call.get("function") or "unknown")}</span></div>'
                f'<pre class="tcargs">{_json_block(call.get("arguments"))}</pre></div>'
            )
        if message.get("other_content_blocks"):
            body += (
                '<details class="unknown-content"><summary>&#9888; unrecognized content '
                'blocks (preserved)</summary>'
                f'<pre>{_json_block(message["other_content_blocks"])}</pre></details>'
            )
        if message.get("error"):
            body += f'<pre class="message-error">{_json_block(message["error"])}</pre>'
        if not body:
            body = "<pre></pre>"
        # native_role is the true wire role (Inspect typing has no developer
        # role); scaffold_injected tags content the scaffold sent itself.
        display_role = message.get("native_role") or role
        injected_tag = message.get("scaffold_injected")
        role_label = (
            f"{display_role} · {injected_tag}" if injected_tag else display_role
        )
        rows.append(
            f'<div class="msg role-{role_class}" id="M{number}" '
            f'data-message-number="{number}">'
            f'<div class="mhead">[M{number}]{index_html} {esc(role_label)}{time_html}</div>'
            f'{body}</div>'
        )
    return "\n".join(rows)


EVIDENCE_NAV_CSS = r"""
.judgment-strip{border:1px solid #d8dbe5;border-radius:8px;margin:14px 0;background:#fff}
.judgment-format{padding:5px 9px;color:#747887;font-size:11px;border-bottom:1px solid #eceef3}
.judgment-overall{padding:6px 9px;border-bottom:1px solid #eceef3;font-size:12px;color:#4f5360}
.judgment-warning,.evidence-warning{margin-left:8px;color:#9d6300}
.dimension-row{min-height:32px;display:flex;align-items:center;gap:7px;padding:3px 8px;border-bottom:1px solid #f0f1f5;font-size:13px}
.dimension-row:last-child{border-bottom:0}.dimension-label{width:92px;color:#555b68;flex:0 0 auto}
.dimension-value{min-width:92px;font-weight:650;white-space:nowrap}.state-yes{color:#a33}.state-no{color:#17663b}.state-review{color:#9d6300}
.state-missing,.state-invalid{color:#9d6300}.evidence-cycle{display:inline-flex;align-items:center;gap:2px}
.evidence-cycle button{border:1px solid #d8dbe5;background:#f8f9fb;height:23px;min-width:25px;padding:0 5px;cursor:pointer;color:#343945}
.evidence-cycle button:hover{background:#eceff5}.evidence-position{min-width:31px;text-align:center;color:#858997;font-size:11px}
.evidence-none{color:#999;font-size:11px}.artifact-refs{display:flex;gap:4px;flex-wrap:wrap}
.artifact-ref{font-size:11px;border:1px solid #dfe2e9;border-radius:10px;padding:1px 6px;text-decoration:none;color:#4c5570}
.dimension-why{font-size:11px;margin-left:auto;max-width:52%}.dimension-why summary{cursor:pointer;color:#747887}
.dimension-why span{display:block;padding:5px 0;color:#4a4e58}.evidence-flash{animation:evidence-flash 1.4s ease-out}
@keyframes evidence-flash{0%{box-shadow:0 0 0 4px #efc84a;background:#fff8d8}100%{box-shadow:none}}
.cnav{position:fixed;bottom:70px;right:18px;width:190px;max-height:calc(100vh - 96px);overflow:auto;background:#fff;border:1px solid #d9dbe3;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.18);padding:10px 12px;font-size:12.5px;z-index:50}
.cnav-grp{margin-bottom:9px;padding-bottom:8px;border-bottom:1px solid #eceef2}.cnav-grp:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none}.cnav-grp.user b{color:#2456a6}
.cnav-row{display:flex;align-items:center;gap:8px;margin:6px 0 2px}.cnav .lbl{font-variant-numeric:tabular-nums;min-width:46px;text-align:center}.cnav button{cursor:pointer;border:1px solid #c8cad2;background:#f2f3f6;border-radius:5px;padding:2px 10px;font-size:13px}.cnav button:hover{background:#e4e6ec}
.explanation-nav-row{justify-content:flex-start}.explanation-nav-row .lbl{min-width:46px}.msg.cited{outline:2px solid #ffb703}.msg.cited .mhead::after{content:" \00b7 cited by judge";font-weight:400;opacity:.9}
.legacy-judgment{border:1px solid #d8dbe5;border-radius:8px;padding:8px;margin:14px 0}.muted{color:#858997;font-weight:normal}
.legacy-score-table th{text-align:left;padding-right:20px}.legacy-score-table td{font-variant-numeric:tabular-nums}
.msg{margin:10px 0;border-radius:6px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.07)}.msg pre{margin:0;padding:10px 14px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.45;background:#fff}.mhead{padding:4px 14px;font-size:11.5px;font-weight:700;letter-spacing:.4px}.mhead .aturn{font-weight:700;opacity:.78}.mhead .mtime{font-weight:500;letter-spacing:0;font-variant-numeric:tabular-nums;opacity:.55}:target .mhead{outline:3px solid #ffb703}
.role-system .mhead{background:#6c757d;color:#fff}.role-user .mhead{background:#2456a6;color:#fff}.role-assistant .mhead{background:#1d7a4f;color:#fff}.role-assistant{background:#f0fff5}.role-assistant pre{background:#f0fff5}.role-tool .mhead{background:#8a5a00;color:#fff}.role-tool pre{background:#fffaf0}.role-other .mhead{background:#444;color:#fff}
.think{margin:6px 0;background:#f3f1fb;border-left:3px solid #b7a8e0;border-radius:4px}.think>.thead{padding:3px 11px;font-size:9.5px;font-weight:700;letter-spacing:.5px;color:#7c6bb0;cursor:pointer;font-family:ui-monospace,Menlo,monospace}.think[open]>.thead{border-bottom:1px solid #ddd6f0}.think>pre{margin:0;padding:6px 11px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.45;color:#2c2540;background:transparent}
.toolcall{margin:8px 0;border:1px solid rgba(0,0,0,.16);border-radius:6px;background:transparent;overflow:hidden}.toolcall .tchead{padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:.3px;color:#155f3e;background:rgba(0,0,0,.045);font-family:ui-monospace,Menlo,monospace}.toolcall .tcname{font-size:12.5px}.toolcall pre.tcargs{margin:0;padding:7px 13px;white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.45;color:#1f3d2b;background:transparent}
.unknown-content{margin:8px 14px;padding:5px 9px;color:#696f7d;background:#f4f5f7;border:1px dashed #c4c8d0;border-radius:6px}.message-error{color:#9b311e}
.msg.flash{animation:evidence-flash 1.1s ease-out}
.judge-view{margin:14px 0;border:1px solid #cfd7e8;border-radius:10px;background:linear-gradient(135deg,#fbfcff,#f5f7fb);box-shadow:0 4px 14px rgba(49,61,90,.06)}
.judge-view>summary{display:flex;align-items:center;justify-content:space-between;gap:12px;cursor:pointer;padding:10px 12px;list-style:none;font-size:14px;font-weight:700;color:#333b4f}
.judge-view>summary::-webkit-details-marker,.judge-subsection>summary::-webkit-details-marker{display:none}.judge-view>summary:before,.judge-subsection>summary:before{content:"›";display:inline-block;color:#818ba0;transition:transform .15s}.judge-view>summary>span:first-of-type{margin-right:auto}.judge-view[open]>summary:before,.judge-subsection[open]>summary:before{transform:rotate(90deg)}
.judge-view-pills{font-size:10px;font-weight:600;color:#64708a;background:#e9edf6;border-radius:10px;padding:2px 7px;white-space:nowrap}.judge-view-body{border-top:1px solid #dce2ee;padding:8px}
.judge-subsection{background:#fff;border:1px solid #e0e4ec;border-radius:7px;margin:5px 0}.judge-subsection>summary{display:flex;align-items:center;gap:7px;cursor:pointer;list-style:none;padding:7px 9px;font-size:12px;font-weight:650;color:#51596b}.judge-subsection[open]>summary{border-bottom:1px solid #eceef3;background:#fafbfc}.judge-subsection-body{padding:7px}.judge-section-group>.judge-subsection-body{background:#f8f9fc}
.judge-exact{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:9px;background:#f7f8fa;border:1px solid #eceef2;border-radius:5px;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#343946}.judge-dimension,.judge-artifact,.judge-tool,.judge-controls{margin:5px}.judge-artifact{scroll-margin-top:25px}
.judge-scope-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:6px}.judge-scope-fact{border:1px solid #e6e9ef;border-radius:6px;padding:6px 8px;background:#fafbfc}.judge-scope-fact span{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.04em;color:#8990a0}.judge-scope-fact strong{display:block;margin-top:2px;font-size:11px;font-weight:600;color:#464d5c}.judge-trajectory-link{display:inline-block;margin:8px 1px 1px;font-size:11px}.judge-unavailable{padding:9px;color:#916320;background:#fff8e8;border:1px solid #ead7a7;border-radius:6px;font-size:11px}.trajectory-panel{scroll-margin-top:20px}.trajectory-panel>summary{padding-bottom:0}.trajectory-panel[open]>summary{padding-bottom:8px;border-bottom:1px solid #eceef3}
.judge-variable-slot{border:1px dashed #cbd3e2;background:#f8faff;border-radius:6px;padding:9px}.judge-variable-slot span{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.05em;color:#7c879e}.judge-variable-slot strong{display:block;margin-top:3px;font-size:11px;color:#4c566b}.generic-judge-stage{margin-top:10px}
@media(max-width:700px){.dimension-row{flex-wrap:wrap}.dimension-label{width:82px}.dimension-why{max-width:100%;margin-left:0;width:100%}.cnav{right:7px;bottom:60px}}
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
      var parentDetails = element.closest("details");
      if (parentDetails) parentDetails.open = true;
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
  document.querySelectorAll(".explanation-nav-row").forEach(function (row) {
    var targetIds = parse(row, "data-explanation-targets");
    var targets = targetIds.map(function (id) { return document.getElementById(id); })
      .filter(function (element) { return Boolean(element); });
    if (!targets.length) {
      var emptyGroup = row.closest(".cnav-grp");
      if (emptyGroup) emptyGroup.remove(); else row.remove();
      return;
    }
    targets.forEach(function (element) { element.classList.add("cited"); });
    var index = -1;
    var position = row.querySelector(".explanation-position");
    position.textContent = "0 / " + targets.length;
    function go(direction) {
      index = (index + direction + targets.length) % targets.length;
      var element = targets[index];
      var parent = element.closest("details");
      while (parent) {
        parent.open = true;
        parent = parent.parentElement ? parent.parentElement.closest("details") : null;
      }
      element.scrollIntoView({behavior: "smooth", block: "center"});
      element.classList.remove("evidence-flash");
      void element.offsetWidth;
      element.classList.add("evidence-flash");
      position.textContent = (index + 1) + " / " + targets.length;
    }
    row.querySelector(".explanation-prev").addEventListener("click", function () {
      go(-1);
    });
    row.querySelector(".explanation-next").addEventListener("click", function () {
      go(1);
    });
  });
  // Copied from Petri: user turns are discovered from the rendered transcript so this
  // works for every trajectory shape without a server-side list.
  var users = Array.prototype.slice.call(document.querySelectorAll(".msg.role-user"));
  var userGroup = document.getElementById("grp-user");
  if (!users.length) {
    if (userGroup) userGroup.style.display = "none";
  } else {
    var userIndex = -1;
    var userLabel = document.getElementById("user-lbl");
    var userCount = document.getElementById("user-cnt");
    userLabel.textContent = "0 / " + users.length;
    userCount.textContent = "(" + users.length + ")";
    function goUser(direction) {
      userIndex = (userIndex + direction + users.length) % users.length;
      var element = users[userIndex];
      element.scrollIntoView({behavior: "smooth", block: "center"});
      element.classList.remove("flash");
      void element.offsetWidth;
      element.classList.add("flash");
      userLabel.textContent = (userIndex + 1) + " / " + users.length;
    }
    document.getElementById("user-prev").addEventListener("click", function () {
      goUser(-1);
    });
    document.getElementById("user-next").addEventListener("click", function () {
      goUser(1);
    });
  }
  var explanationNav = document.querySelector(".explanation-nav");
  if (explanationNav && !Array.prototype.some.call(
    explanationNav.querySelectorAll(".cnav-grp"),
    function (group) { return group.style.display !== "none"; }
  )) {
    explanationNav.remove();
  }
  function openTargetParents(target) {
    var parent = target.closest("details");
    while (parent) {
      parent.open = true;
      parent = parent.parentElement ? parent.parentElement.closest("details") : null;
    }
  }
  function openCurrentHash() {
    if (!window.location.hash) return;
    var target = document.getElementById(window.location.hash.slice(1));
    if (!target) return;
    openTargetParents(target);
    target.scrollIntoView({block: "center"});
  }
  document.querySelectorAll('a[href^="#"]').forEach(function (link) {
    link.addEventListener("click", function () {
      var target = document.getElementById(link.getAttribute("href").slice(1));
      if (!target) return;
      openTargetParents(target);
    });
  });
  openCurrentHash();
  window.addEventListener("hashchange", openCurrentHash);
})();
</script>
"""
