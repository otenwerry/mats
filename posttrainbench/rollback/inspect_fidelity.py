"""Inspect how faithfully the cut-point workspace was rebuilt.

Deterministic checks (free): forward vs. backward file-set + per-file content
(sha) agreement, plus common-sense sanity (test data present and non-trivial,
evaluate.py matches the repo, templates/prompt/timer present). For a heavier
LLM-eyeball pass, point a subagent at the printed build dirs — not wired here.

Usage:
    python -m rollback.inspect_fidelity            # build (skip-bash) + inspect
    python -m rollback.inspect_fidelity --no-build # inspect existing builds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, ptbio, timing
from .workspace import forward, backward, common

EPOCH = 1771300000  # fixed epoch for build-only inspection (timer set by runner)


def _content_diff(a: Path, b: Path, files: set[str]) -> dict:
    same, differ = [], []
    for rel in sorted(files):
        fa, fb = a / rel, b / rel
        if not fa.exists() or not fb.exists():
            differ.append({"file": rel, "reason": "missing on one side"})
            continue
        if common.file_sha(fa) == common.file_sha(fb):
            same.append(rel)
        else:
            differ.append({"file": rel, "reason": "sha mismatch",
                           "bytes": [fa.stat().st_size, fb.stat().st_size]})
    return {"identical": same, "differing": differ}


def sanity_checks(task: Path, traj: config.Trajectory) -> list[dict]:
    out = []

    def check(name, ok, detail=""):
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    # Local test data only exists for benchmarks that ship it in the task def
    # (healthbench). gpqamain pulls the gated HF dataset at eval time instead.
    repo_data = traj.task_def_dir / "evaluation_code"
    if repo_data.exists():
        td = task / "evaluation_code" / "data" / f"{traj.benchmark_id}.jsonl"
        check("test_data_present", td.exists(),
              f"{td.stat().st_size} bytes" if td.exists() else "MISSING")
        check("test_data_nontrivial", td.exists() and td.stat().st_size > 1_000_000,
              "expect MBs (archive strips it; repo overlay restores)")

    if traj.scaffold == "opencode":
        oc = task / "opencode.json"
        check("opencode_json_present", oc.exists())
        if oc.exists():
            from . import scaffold as _scaffold
            check("opencode_json_matches_solve_sh",
                  oc.read_text() == _scaffold.opencode_config_text(traj))

    ev = task / "evaluate.py"
    repo_ev = traj.task_def_dir / "evaluate.py"
    check("evaluate_py_matches_repo",
          ev.exists() and repo_ev.exists()
          and common.file_sha(ev) == common.file_sha(repo_ev))

    tmpl = task / "templates"
    check("templates_present", tmpl.exists() and any(tmpl.glob("*.jinja")),
          f"{len(list(tmpl.glob('*.jinja')))} jinja" if tmpl.exists() else "MISSING")

    check("prompt_present", (task / ".rollback_prompt.txt").exists())

    timer = task / "timer.sh"
    check("timer_present", timer.exists())
    return out


def inspect(traj: config.Trajectory, cut: int, do_build: bool,
            bash_mode: str = "skip") -> dict:
    spec_f = config.ExperimentSpec(traj, "forward", "prompt1", cut)
    spec_b = config.ExperimentSpec(traj, "backward", "prompt1", cut)

    report: dict = {"run_id": traj.run_id, "cut_before_event": cut}
    if do_build:
        report["forward_build"] = forward.build(spec_f, creation_epoch=EPOCH,
                                                bash_mode=bash_mode)
        report["backward_build"] = backward.build(spec_b, creation_epoch=EPOCH)

    tf = spec_f.build_dir / "task"
    tb = spec_b.build_dir / "task"
    pf, pb = common.path_set(tf), common.path_set(tb)

    report["timer"] = timing.reconstruct(traj, cut)

    # Pre-cut bash side-effect files (e.g. inspect-ai eval logs) exist only in
    # the backward build until the forward replay re-executes that bash on the
    # GPU box — an EXPECTED, surfaced divergence, not a fidelity failure. Even
    # on-GPU the regenerated log differs (fresh timestamps/ids), so it stays a
    # known-nondeterministic artifact either way.
    only_b = sorted(pb - pf)
    expected_b = [p for p in only_b if common.name_embedded_epoch(p) is not None]
    report["file_set"] = {
        "forward_count": len(pf), "backward_count": len(pb),
        "only_forward": sorted(pf - pb), "only_backward": only_b,
        "expected_backward_only_bash_side_effects": expected_b,
        "converged": pf == pb,
        "converged_modulo_expected": (pf | set(expected_b)) == pb and not (pf - pb),
    }
    report["content"] = _content_diff(tf, tb, pf & pb)
    report["sanity_forward"] = sanity_checks(tf, traj)
    report["sanity_backward"] = sanity_checks(tb, traj)

    # session reconstruction (into the forward cell's job home, as prepare()
    # would; validates the cut falls on a message/turn boundary)
    if traj.scaffold == "opencode":
        from .session import recon_opencode
        sess = recon_opencode.build_session(spec_f, spec_f.build_dir)
        report["session"] = {k: sess[k] for k in
                             ("session_id", "messages_kept", "messages_dropped",
                              "parts_written", "prompt_chars", "cut_validation")}
    else:
        from .session import recon
        sess = recon.build_session(spec_f, spec_f.build_dir)
        report["session"] = {k: sess[k] for k in
                             ("session_id", "n_turns", "prefix_validation")}

    # rollup verdict (expected/explained divergences don't fail the build, but
    # they are listed above — never silently absorbed)
    sane = all(c["ok"] for c in report["sanity_forward"] + report["sanity_backward"])
    report["verdict"] = {
        "file_set_converged": report["file_set"]["converged"],
        "file_set_converged_modulo_expected":
            report["file_set"]["converged_modulo_expected"],
        "content_identical": not report["content"]["differing"],
        "sanity_passed": sane,
        "overall_ok": report["file_set"]["converged_modulo_expected"]
                      and not report["content"]["differing"] and sane,
    }
    return report


def _print(report: dict) -> None:
    v = report["verdict"]
    fs = report["file_set"]
    print(f"\n=== Workspace fidelity: {report['run_id']} (cut before ev{report['cut_before_event']}) ===")
    print(f"timer: ~{report['timer']['elapsed_seconds']}s elapsed, "
          f"{report['timer']['remaining_seconds']/3600:.2f}h remaining "
          f"({report['timer']['method']})")
    print(f"file sets: forward={fs['forward_count']} backward={fs['backward_count']} "
          f"converged={fs['converged']}")
    if fs["only_forward"]:
        print("  only in forward:", fs["only_forward"])
    if fs["only_backward"]:
        print("  only in backward:", fs["only_backward"])
    if fs["expected_backward_only_bash_side_effects"]:
        print("  ^ expected pre-cut bash side-effects (forward regenerates on "
              "GPU, content will still differ):",
              fs["expected_backward_only_bash_side_effects"])
    cd = report["content"]
    print(f"content: {len(cd['identical'])} identical, {len(cd['differing'])} differing")
    for d in cd["differing"][:10]:
        print("  DIFFER:", d)
    for side in ("sanity_forward", "sanity_backward"):
        bad = [c for c in report[side] if not c["ok"]]
        print(f"{side}: {'all OK' if not bad else 'FAILS: ' + str(bad)}")
    if "session" in report:
        print(f"session: {report['session']}")
    print(f"\nOVERALL: {'✅ OK' if v['overall_ok'] else '❌ check failures above'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=config.CUT_BEFORE_EVENT)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--bash-mode", default="skip", choices=["skip", "safe", "execute"])
    ap.add_argument("--json", action="store_true", help="dump full JSON report")
    args = ap.parse_args()
    rep = inspect(config.TRAJECTORY, args.cut, not args.no_build, args.bash_mode)
    if args.json:
        print(json.dumps(rep, indent=1))
    else:
        _print(rep)


if __name__ == "__main__":
    main()
