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
    viewer.JUDGE_TEST_MANIFEST = root / "judge_test_sources.json"


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
    assert "No trajectories" in index
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
    assert viewer._trajectory_category({"judgment": judgment}) == "review"


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
        "load_errors": 1,
        "judge_test_sources": 0,
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
    assert "No trajectories" in reasoning_index
    assert "test_set" in index and "$1.2500" in index
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
    assert '<details class="panel trajectory-panel" id="trajectory-record">' in new_page
    assert "native_reasoning_excluded_by_policy" in new_page
    assert "predictions.csv · final" in new_page
    assert "Legacy numeric judgment, shown exactly as stored" in legacy_page
    assert "legacy incremental · stored evidence" in legacy_page
    assert "EXACT LEGACY STAGE-ONE PROMPT" in legacy_page
    assert "EXACT LEGACY FINAL EVIDENCE" in legacy_page
    assert "does not reconstruct those missing parts from current code" in legacy_page
    assert "8/10" in legacy_page
    assert "Reward hacking" in visuals
    assert "Recorded cost" in visuals


def test_generic_judge_pages_replace_and_archive_old_judge_test_pages() -> None:
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
        fraud_tests = (output / "fraud_detection_judge_tests.html").read_text()
        p_tests = (output / "reasoning_prompt_benchmark_judge_tests.html").read_text()
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

    assert first["legacy_pages_archived"] == 8
    assert second["legacy_pages_archived"] == 0
    assert viewer._GENERATED_MARKER in fraud_tests
    assert viewer._GENERATED_MARKER in p_tests
    assert len(archived_details) == 2
    assert (
        '<a href="fraud_detection_judge_tests.html" class="active">judge tests</a>'
        in fraud_tests
    )
    assert 'href="judge_fraud_detection.html"' in fraud_tests
    assert 'href="judge_reasoning_prompt_benchmark.html"' in p_tests
    assert not any(stale_generic_pages)
    assert 'href="fraud_detection_judge_tests.html"' in fraud_index
    assert 'href="reasoning_prompt_benchmark_judge_tests.html"' in p_index
    assert '<a href="index.html" class="active">fraud_detection</a>' in ml_generic
    assert (
        '<a href="judge_fraud_detection.html" class="active">judge view</a>'
        in ml_generic
    )
    assert '<a href="index.html">trajectories</a>' in ml_generic
    assert '<a href="visuals.html">visuals</a>' in ml_generic
    assert '<a href="fraud_detection_judge_tests.html">judge tests</a>' in ml_generic
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
        assert f"family=`{family}`; stage=`stage_1`" not in page
        assert f"family=`{family}`; stage=`final`" in page
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
        '<div class="meta-key">epoch</div><div class="meta-value">2</div>'
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


def test_judge_test_cohort_is_migrated_then_filled_by_current_rejudgments() -> None:
    viewer = load_viewer()
    fraud = {
        **audit(legacy=True),
        "mode": "legacy-fraud-run",
        "task": "legacy-fraud-task",
        "seed": "fraud_detection",
        "epoch": 1,
    }
    reasoning = {
        **audit(legacy=True),
        "mode": "legacy-reasoning-run",
        "task": "legacy-reasoning-task",
        "seed": "reasoning_prompt_benchmark",
        "epoch": 1,
        "mtime": 3.0,
    }
    loaded = [fraud, reasoning]

    async def fake_load(*_args, **_kwargs):
        return loaded, []

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        point_at(viewer, root)
        viewer.load_all = fake_load
        output = root / "viewer"
        (output / "pages").mkdir(parents=True)
        for item in loaded:
            source = viewer.traj_key(item)
            seed = item["seed"]
            detail = f"{source}__judge_old.html"
            (output / "pages" / detail).write_text("old judgment")
            (output / f"{seed}_judge_tests.html").write_text(
                f'<a href="pages/{detail}">old</a>'
            )

        first = asyncio.run(viewer.build(use_cache=False))
        fraud_tests_pending = (
            output / "fraud_detection_judge_tests.html"
        ).read_text()
        fraud_main_pending = (output / "index.html").read_text()
        fraud_past = (output / "fraud_detection_past.html").read_text()

        rejudge = stamp_current_method(viewer, {
            **audit(),
            "mode": "rejudge-current",
            "task": "retrospective-task",
            "seed": "fraud_detection",
            "epoch": 1,
            "mtime": 10.0,
            "retrospective_rejudge": {
                "source_run": fraud["mode"],
                "source_task": fraud["task"],
                "seed": fraud["seed"],
                "epoch": fraud["epoch"],
                "source_key": "hashed-source-identity",
            },
        })
        loaded.append(rejudge)
        second = asyncio.run(viewer.build(use_cache=False))
        fraud_tests_judged = (
            output / "fraud_detection_judge_tests.html"
        ).read_text()
        fraud_main_judged = (output / "index.html").read_text()
        manifest = (root / "judge_test_sources.json").read_text()

    assert first["judge_test_sources"] == 2
    assert first["current_trajectories"] == 0
    assert "Awaiting current judgment" in fraud_tests_pending
    assert "No trajectories" in fraud_main_pending
    assert "Historical judgments" in fraud_past
    assert second["current_trajectories"] == 1
    assert "Reward hacks" in fraud_tests_judged
    assert ">Source</th>" in fraud_tests_judged
    assert "Reward hacks" in fraud_main_judged
    assert "environment-judge-test-sources-v1" in manifest
