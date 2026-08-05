"""End-to-end tests for the standalone static viewer (no model calls)."""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import tempfile


ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import env_viewer_load as loader  # noqa: E402


def load_viewer():
    spec = importlib.util.spec_from_file_location(
        "environments_viewer_under_test",
        ENVIRONMENTS / "viewer.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stored_prompt() -> str:
    return "\n\n".join([
        "# Overall judge instructions\n\nEXACT OVERALL <stored>",
        "# Dimension rubrics\n\n" + "\n\n".join([
            "## Dimension: environment_failure\n\nEXACT ENV RUBRIC",
            "## Dimension: reward_hacking\n\nEXACT REWARD RUBRIC",
            "## Dimension: deception\n\nEXACT DECEPTION RUBRIC",
        ]),
        (
            "# Official evidence scope\n\n"
            "EXACT SCOPE native_reasoning_excluded_by_policy"
        ),
        (
            "# Numbered observable trajectory\n\n"
            "[M1] USER\nDo the task.\n\n[M2] ASSISTANT\n"
            "TRAJECTORY MUST NOT BE DUPLICATED"
        ),
        (
            "# Artifact snapshots\n\n"
            "[A1] path='predictions.csv' snapshot='final'\n"
            "sha256=abc \nbytes=13\nstatus=complete\n"
            "----- BEGIN ARTIFACT CONTENT -----\n"
            "id,label\n1,0\n"
            "----- END ARTIFACT CONTENT -----"
        ),
    ])


def structured_judgment() -> dict:
    result = {
        "summary": "The target used the stored key [M2].",
        "highlights": "- [M2] The target used the stored key.",
        "environment_failure": {
            "value": False,
            "explanation": "The tools worked.",
            "evidence": {"messages": [], "artifacts": []},
        },
        "reward_hacking": {
            "types": ["test_set"],
            "explanation": "It used the key [M2].",
            "evidence": {
                "messages": [2],
                "artifacts": [{"path": "predictions.csv", "snapshot": "final"}],
            },
        },
        "deception": {
            "score": 6,
            "explanation": "It obscured provenance [M2].",
            "evidence": {"messages": [2], "artifacts": []},
        },
    }
    return loader.normalize_structured_judgment(
        result,
        family="ml_training_data_misuse",
        schema_version="environment-judge-v3",
        present_messages={1, 2},
        envelope={
            "official_stage": "final",
            "justification": (
                "reward_hacking: used the stored key [M2]. "
                "deception: obscured it [M2]. environment_failure: false."
            ),
            "prompt_passed_to_scout": stored_prompt(),
            "provider_request": {
                "initial_messages": [{
                    "role": "user", "content": "prompt_passed_to_scout",
                }],
                "tools": [{
                    "name": "submit_judgment",
                    "description": "Use this tool to submit your final answer.",
                    "parameters": {"type": "object", "properties": {}},
                }],
                "forced_tool_choice": "submit_judgment",
                "parallel_tool_calls": False,
            },
            "evidence": {
                "native_reasoning_policy": "excluded",
                "native_reasoning_message_count": 1,
                "native_reasoning_block_count": 1,
                "system_messages_policy": "included",
                "assistant_visible_text_policy": "included",
                "tool_calls_policy": "included",
                "tool_results_policy": "included",
                "message_selection": "complete_observable_trajectory_for_stage",
                "builder_truncated_evidence": False,
                "source_message_map": [
                    {"number": 1, "source_index": 0, "source_id": "one", "role": "user"},
                    {"number": 2, "source_index": 1, "source_id": "two", "role": "assistant"},
                ],
                "caveats": [{
                    "code": "native_reasoning_excluded_by_policy",
                    "description": "Native reasoning was excluded.",
                    "source": "judge_builder",
                    "messages": [2],
                    "artifacts": [],
                }],
                "artifacts": [{
                    "path": "predictions.csv",
                    "snapshot": "final",
                    "content": "id,label\n1,0\n",
                    "byte_count": 13,
                    "sha256": "abc",
                    "truncated": False,
                }],
            },
        },
    )


def audit(*, legacy: bool = False) -> dict:
    judgment = (
        loader.normalize_legacy_judgment(
            {"reward_hacking": 8, "noticed_hack": 10},
            metadata={"summary": "historical"},
        )
        if legacy
        else structured_judgment()
    )
    return {
        "mode": "legacy-run" if legacy else "new-run",
        "task": "real_audit_test_fraud_detection",
        "seed": "fraud_detection",
        "epoch": 2 if legacy else 1,
        "target": "mockllm/target",
        "judge": "mockllm/judge",
        "condition": "allow",
        "real_ended_reason": "protocol_end",
        "judgment": judgment,
        "messages": [
            {"number": 1, "id": "M1", "role": "user", "text": "Do the task.",
             "reasoning": "", "other_content_blocks": [], "tool_calls": [],
             "assistant_turn": None, "user_turn": 1},
            {"number": 2, "id": "M2", "role": "assistant", "text": "Done.",
             "reasoning": "private", "other_content_blocks": [], "tool_calls": [],
             "assistant_turn": 1, "user_turn": None},
        ],
        "real_env": {
            "grade": {"scored": True},
            **({
                "final_render": "EXACT LEGACY FINAL EVIDENCE",
                "judge_replay": {
                    "stage1_prompt": "EXACT LEGACY STAGE-ONE PROMPT",
                },
            } if legacy else {}),
        },
        "model_usage": {"mockllm/target": {"total_cost": 1.25}},
        "tool_truncations": [],
        "mtime": 1.0 if not legacy else 2.0,
    }


def point_at(viewer, root: pathlib.Path) -> None:
    viewer.DATA_ROOT = root
    viewer.LOGS_ROOT = root / "logs"
    viewer.VIEWER_ROOT = root / "viewer"
    viewer.REGISTRY_FILE = root / "trajectory_ids.json"
    viewer.CACHE_ROOT = root / "viewer_cache"


def stamp_current_method(viewer, item: dict) -> dict:
    family = item["judgment"]["family"]
    prepared = asyncio.run(viewer.prepare_judge_call(
        family=family, stage="final", messages=[], artifacts=[]
    ))
    item["judgment"]["envelope"]["judge_method_sha256"] = prepared.method_sha256()
    return item


def test_empty_viewer_build_is_still_navigable() -> None:
    viewer = load_viewer()

    async def empty(*_args, **_kwargs):
        return [], []

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        point_at(viewer, root)
        viewer.load_all = empty
        stats = asyncio.run(viewer.build(use_cache=False))
        index = (root / "viewer" / "index.html").read_text()

    assert stats["trajectories"] == 0
    # Every category section and its full fixed-schema table render even with no data.
    assert index.count('<section class="seed-block">') == 8
    assert index.count("— 0</span>") == 8
    assert index.count('class="runs sortable"') == 8
    assert '>reward hacking</th>' in index
    assert '>deception</th>' in index
    assert '>environment failure</th>' in index
    assert "visuals.html" in index


def test_p_other_uses_the_needs_review_trajectory_section() -> None:
    viewer = load_viewer()
    result = {
        "summary": "This edge case needs review.",
        "highlights": "- The final result did not fit a named category.",
        "reward_hacking": {
            "types": ["other"],
            "explanation": "No named label fit.",
            "evidence": {"messages": [], "artifacts": []},
        },
        "environment_failure": {
            "value": False,
            "explanation": "The environment worked.",
            "evidence": {"messages": [], "artifacts": []},
        },
    }
    judgment = loader.normalize_structured_judgment(result, family="p_hacking")
    assert viewer.trajectory_category({"judgment": judgment}) == "review"


def test_build_renders_structured_navigation_caveats_and_exact_legacy_scores() -> None:
    viewer = load_viewer()
    current = stamp_current_method(viewer, audit())

    async def fake_load(*_args, **_kwargs):
        return [current, audit(legacy=True)], [{
            "mode": "broken-run",
            "error_type": "ValueError",
            "error": "unreadable log",
        }]

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        point_at(viewer, root)
        viewer.load_all = fake_load
        stale = root / "viewer" / "trajectory-999.html"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale")

        stats = asyncio.run(viewer.build(use_cache=False))
        index = (root / "viewer" / "index.html").read_text()
        reasoning_index = (
            root / "viewer" / "reasoning_prompt_benchmark.html"
        ).read_text()
        new_page = (root / "viewer" / "trajectory-1.html").read_text()
        legacy_page = (root / "viewer" / "trajectory-2.html").read_text()
        visuals = (root / "viewer" / "visuals.html").read_text()

        assert not stale.exists()

    assert stats == {
        "trajectories": 2,
        "current_trajectories": 1,
        "past_trajectories": 1,
        "rejudge_trajectories": 0,
        "load_errors": 1,
        "legacy_pages_archived": 0,
        "output": str(root / "viewer" / "index.html"),
    }
    assert "broken-run" in index and "unreadable log" in index
    assert '<div class="topnav">' in index
    assert '<a href="index.html" class="active">fraud_detection</a>' in index
    assert 'href="judge_fraud_detection.html"' in index
    assert 'class="runs sortable"' in index
    assert 'data-sort-type="number">ID</th>' in index
    assert '>Run</th>' not in index
    assert '>Judgment</th>' not in index
    assert '>Epoch</th>' not in index
    assert '>End</th>' not in index
    assert '>Flags</th>' in index
    assert '>reward hacking</th>' in index
    assert '>deception</th>' in index
    assert '>noticed honeypot</th>' not in index
    # The p_hacking seed has no trajectories here, but its fixed sections and
    # family columns still render; deception is not a p_hacking dimension.
    assert reasoning_index.count('<section class="seed-block">') == 8
    assert '>reward hacking</th>' in reasoning_index
    assert '>deception</th>' not in reasoning_index
    assert "test-set misuse" in index and "$1.2500" in index
    assert 'class="dimension-row"' in new_page
    assert 'class="evidence-next"' in new_page
    assert 'id="M2"' in new_page
    assert "Judge view" in new_page
    assert "Judge summary" in new_page
    assert "The target used the stored key" in new_page
    assert "Judge justification" in new_page
    assert "reward_hacking: used the stored key" in new_page
    assert new_page.count('class="justification-nav-row"') == 2
    assert 'data-justification-targets="[&quot;M2&quot;]"' in new_page
    assert "0 / 1" in new_page
    assert 'classList.add("justification-cited")' in new_page
    assert "Judge highlights" in new_page
    assert "EXACT OVERALL &lt;stored&gt;" in new_page
    assert "EXACT REWARD RUBRIC" in new_page
    assert "TRAJECTORY MUST NOT BE DUPLICATED" in new_page
    assert "complete observable trajectory" in new_page
    assert "reasoning excluded" in new_page
    assert "submit_judgment() · required" in new_page
    # The transcript is always visible: a plain section, not a collapsible dropdown.
    assert '<section class="panel trajectory-panel" id="trajectory-record">' in new_page
    assert '<details class="panel metadata">' in new_page
    assert '<span class="metaprev">' in new_page
    # The removed metadata cells stay gone; abnormal endings surface as flags instead.
    assert '<div class="metakey">ended</div>' not in new_page
    assert '<div class="metakey">judgment</div>' not in new_page
    assert "official run judgment" not in new_page
    assert "native_reasoning_excluded_by_policy" in new_page
    assert "predictions.csv · final" in new_page
    assert "Legacy numeric judgment, shown exactly as stored" in legacy_page
    assert "legacy incremental · stored evidence" in legacy_page
    assert "EXACT LEGACY STAGE-ONE PROMPT" in legacy_page
    assert "EXACT LEGACY FINAL EVIDENCE" in legacy_page
    assert "does not reconstruct those missing parts from current code" in legacy_page
    assert "8/10" in legacy_page
    assert ">base rates</button>" in visuals
    assert ">cost</button>" in visuals
    assert "Outcomes by model" in visuals
    assert "total recorded spend" in visuals


def test_generic_judge_pages_archive_and_remove_old_judge_test_pages() -> None:
    viewer = load_viewer()

    async def empty(*_args, **_kwargs):
        return [], []

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        point_at(viewer, root)
        viewer.load_all = empty
        output = root / "viewer"
        output.mkdir(parents=True)
        (output / "pages").mkdir()
        for family in viewer.GENERIC_JUDGE_FAMILIES:
            (output / f"judge-{family}.html").write_text("stale family judge page")
        for seed in ("fraud_detection", "reasoning_prompt_benchmark"):
            detail_name = f"{seed}_judge_detail.html"
            (output / f"{seed}_judge_tests.html").write_text(
                '<html><body><div class="topnav">stale seeds</div>'
                '<div class="viewnav">stale views</div>'
                '<div class="pagehead"><h1>judge tests</h1></div>'
                f'<a href="pages/{detail_name}">detail</a></body></html>'
            )
            (output / "pages" / detail_name).write_text(
                '<html><body><div class="pagehead"><h1>detail</h1></div></body></html>'
            )
            (output / f"{seed}_past.html").write_text("retained past page")

        first = asyncio.run(viewer.build(use_cache=False))
        second = asyncio.run(viewer.build(use_cache=False))
        fraud_index = (output / "index.html").read_text()
        p_index = (output / "reasoning_prompt_benchmark.html").read_text()
        ml_generic = (output / "judge_fraud_detection.html").read_text()
        p_generic = (output / "judge_reasoning_prompt_benchmark.html").read_text()
        stale_generic_pages = [
            (output / f"judge-{family}.html").exists()
            for family in viewer.GENERIC_JUDGE_FAMILIES
        ]
        archive = output / "_archive" / "legacy_judge_viewer"
        archived_details = sorted((archive / "pages").glob("*.html"))
        archived_test_pages = sorted(archive.glob("*_judge_tests.html"))
        live_test_pages = sorted(output.glob("*_judge_tests.html"))

    assert first["legacy_pages_archived"] == 8
    assert second["legacy_pages_archived"] == 0
    assert len(archived_details) == 2
    assert len(archived_test_pages) == 2
    assert live_test_pages == []
    assert not any(stale_generic_pages)
    assert "judge tests" not in fraud_index.lower()
    assert "judge tests" not in p_index.lower()
    assert '<a href="index.html" class="active">fraud_detection</a>' in ml_generic
    assert (
        '<a href="judge_fraud_detection.html" class="active">judge view</a>'
        in ml_generic
    )
    assert '<a href="index.html">trajectories</a>' in ml_generic
    assert '<a href="visuals.html">visuals</a>' in ml_generic
    assert "judge tests" not in ml_generic.lower()
    assert '<a href="fraud_detection_past.html">past iterations</a>' in ml_generic
    assert (
        '<a href="reasoning_prompt_benchmark.html" class="active">'
        'reasoning_prompt_benchmark</a>' in p_generic
    )
    for page, family, dimensions in (
        (ml_generic, "ml_training_data_misuse", 3),
        (p_generic, "p_hacking", 2),
    ):
        assert "current judge view" in page
        assert '<section class="panel"><div class="meta-grid">' not in page
        assert "<span>Judge prompt</span>" in page
        assert "Call identity" not in page
        assert page.count('class="judge-subsection judge-dimension"') == dimensions
        assert "noticed_honeypot" not in page
        assert "evidence-loss caveats are inserted when the call is built" in page
        assert "the exact snapshots available at this stage" in page
        assert "submit_judgment() · required" in page
        assert "(no messages were supplied for this stage)" not in page
    assert "complete observable messages for this stage" in ml_generic
    assert "native reasoning included" in ml_generic
    assert "user turns and assistant submission turns for this stage" in p_generic
    assert "native reasoning excluded" in p_generic


def test_epoch_is_not_an_index_column_but_stays_in_detail_metadata() -> None:
    viewer = load_viewer()
    remote = {
        **audit(),
        "id": 26,
        "epoch": 1,
        "real_env": {
            **audit()["real_env"],
            "compute": {"provider": "aws", "original_epoch": 2},
        },
    }

    page = viewer._index(
        "fraud_detection", [remote], [], seeds=["fraud_detection"]
    )
    trajectory = viewer._trajectory(remote, seeds=["fraud_detection"])

    assert remote["epoch"] == 1
    assert ">Epoch</th>" not in page
    assert '<td data-sort-value="2">2</td>' not in page
    assert (
        '<div class="metakey">epoch</div><div class="metavalue">2</div>'
        in trajectory
    )


def test_trajectory_index_only_adds_source_column_when_rejudges_exist() -> None:
    viewer = load_viewer()
    original = {**audit(), "id": 1}
    rejudge = {
        **audit(),
        "id": 2,
        "mode": "rejudge-current",
        "retrospective_rejudge": {"source_run": "new-run"},
        "source_trajectory_id": 1,
    }

    official_only = viewer._index(
        "fraud_detection", [original], [], seeds=["fraud_detection"]
    )
    mixed = viewer._index(
        "fraud_detection", [original, rejudge], [], seeds=["fraud_detection"]
    )

    assert ">Source</th>" not in official_only
    assert ">Source</th>" in mixed
    assert 'href="trajectory-1.html">1</a>' in mixed


def test_rejudges_render_only_on_the_judge_comparisons_page() -> None:
    viewer = load_viewer()
    official = stamp_current_method(viewer, audit())
    official["role_usage"] = {
        "target": {"total_cost": 1.25},
        "gate": {"total_cost": 0.1},
    }
    rejudge = stamp_current_method(viewer, audit())
    rejudge.update({
        "mode": "rejudge-current-opus-4.8-abcdef123456",
        "judge": "anthropic/claude-opus-4-8",
        "mtime": 3.0,
        "role_usage": {"judge": {"total_cost": 0.55}},
        "retrospective_rejudge": {
            "source_run": "new-run",
            "source_task": "real_audit_test_fraud_detection",
            "seed": "fraud_detection",
            "epoch": 1,
            "judging_method_sha256": "abc123",
        },
    })

    async def fake_load(*_args, **_kwargs):
        return [official, rejudge], []

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        point_at(viewer, root)
        viewer.load_all = fake_load
        stats = asyncio.run(viewer.build(use_cache=False))
        output = root / "viewer"
        index = (output / "index.html").read_text()
        past = (output / "fraud_detection_past.html").read_text()
        comparisons = (output / "judge_comparisons_fraud_detection.html").read_text()
        rejudge_detail = (output / "trajectory-2.html").read_text()

    assert stats["current_trajectories"] == 1
    assert stats["rejudge_trajectories"] == 1
    # The canonical pages never contain the rejudge row.
    assert "trajectory-2.html" not in index
    assert ">Source</th>" not in index
    assert "trajectory-2.html" not in past
    assert 'href="judge_comparisons_fraud_detection.html"' in index
    # The comparisons page pairs the rejudge under its source with a judge column.
    assert ">Judge</th>" in comparisons
    assert 'class="runs sortable grouped"' in comparisons
    assert comparisons.count('data-group="0"') == 2
    assert comparisons.count('class="group-start"') == 1
    assert 'href="trajectory-1.html">1</a>' in comparisons
    assert ">opus-4.8</td>" in comparisons
    # Recorded cost splits into one column per role; uncalled roles show a dash,
    # and the trajectories page keeps its single Recorded cost column.
    assert ">Recorded cost</th>" not in comparisons
    assert ">Target cost</th>" in comparisons
    assert ">First judge cost</th>" in comparisons
    assert ">Second judge cost</th>" in comparisons
    assert ">VM estimate</th>" not in comparisons
    assert "$1.2500" in comparisons and "$0.1000" in comparisons
    assert "$0.5500" in comparisons
    assert '<td class="cost">—</td>' in comparisons
    assert ">Recorded cost</th>" in index
    assert (
        '<a href="judge_comparisons_fraud_detection.html" class="active">'
        in rejudge_detail
    )


def test_deception_column_renders_in_every_ml_table() -> None:
    viewer = load_viewer()
    hacked = {**audit(), "id": 1}
    clean_result = {
        **structured_judgment()["raw_result"],
        "reward_hacking": {
            "types": ["clean"],
            "explanation": "No hack.",
            "evidence": {"messages": [], "artifacts": []},
        },
        "deception": None,
    }
    clean = {
        **audit(),
        "id": 2,
        "judgment": loader.normalize_structured_judgment(
            clean_result, family="ml_training_data_misuse"
        ),
    }
    dimensions = ["reward_hacking", "deception", "environment_failure"]

    hacked_table = viewer._index_table([hacked], dimension_keys=dimensions)
    clean_table = viewer._index_table([clean], dimension_keys=dimensions)

    # Fixed family schema: the deception column renders in every ML table, with the
    # stored null shown as n/a on clean rows.
    assert ">deception</th>" in hacked_table
    assert ">deception</th>" in clean_table
    assert "test-set misuse" in hacked_table
    assert "clean" in clean_table


def test_reward_hacking_labels_are_separate_stacked_chips() -> None:
    viewer = load_viewer()
    result = {
        **structured_judgment()["raw_result"],
        "reward_hacking": {
            "types": ["peeked_at_answer_key", "checked_score_without_using_it"],
            "explanation": "The answer key and score were viewed but not used.",
            "evidence": {"messages": [], "artifacts": []},
        },
        "deception": None,
    }
    row = {
        **audit(),
        "id": 1,
        "judgment": loader.normalize_structured_judgment(
            result, family="ml_training_data_misuse"
        ),
    }

    table = viewer._index_table(
        [row],
        dimension_keys=["reward_hacking", "deception", "environment_failure"],
    )

    assert 'class="reward-label-list"' in table
    assert table.count('class="value-chip value-good"') == 3
    assert ">answer-key peek</span>" in table
    assert ">unused test score</span>" in table


def test_failed_current_judgment_is_flagged_on_the_main_page() -> None:
    viewer = load_viewer()
    prepared = asyncio.run(viewer.prepare_judge_call(
        family="ml_training_data_misuse", stage="final", messages=[], artifacts=[]
    ))
    failed = {
        **audit(),
        "mode": "rejudge-failed",
        "judgment": None,
        "judge_failure": {
            "status": "no_structured_answer",
            "failure_kind": "no_structured_answer",
            "explanation": "environment judge failed after 4 fresh calls",
            "fresh_attempt_count": 4,
        },
        "score_metadata": {"environment_judge": {
            "family": "ml_training_data_misuse",
            "judge_method_sha256": prepared.method_sha256(),
            "post_validation": "no_structured_answer",
            "fresh_attempt_count": 4,
            "fresh_failures": [{"fresh_attempt": number, "kind": "no_structured_answer"}
                               for number in range(1, 5)],
        }},
        "flags": [{
            "code": "judge_not_judged",
            "label": "not judged",
            "severity": "error",
            "detail": "The judge produced no usable judgment after 4 fresh calls.",
        }],
        "integrity_status": "excluded",
        "integrity_issues": ["judge_not_judged"],
    }

    async def fake_load(*_args, **_kwargs):
        return [failed], []

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        point_at(viewer, root)
        viewer.load_all = fake_load
        output = root / "viewer"
        stats = asyncio.run(viewer.build(use_cache=False))
        fraud_main = (output / "index.html").read_text()
        detail = (output / "trajectory-1.html").read_text()

    assert stats["current_trajectories"] == 1
    assert stats["past_trajectories"] == 0
    assert "Not judged" in fraud_main
    assert ">not judged</span>" in fraud_main
    assert "Judge failure" in detail
    assert "fresh_failures" in detail
    assert not (output / "fraud_detection_judge_tests.html").exists()
