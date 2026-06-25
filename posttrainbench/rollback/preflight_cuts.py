"""Local, free pre-flight of every trajectory's cut/recon BEFORE spending a box.

For each of the 30 RH trajectories it (a) attempts the scaffold-appropriate
session reconstruction (recon_opencode / recon / recon_codex) at the configured
cut and reports whether it builds + whether validate_cut is clean, and (b)
statically checks that the prep commands' output dir matches eval_model_dir
(the cut177 failure class). Nothing here calls an API or launches a box — it
just exercises the offline recon path, so we learn which trajectories are safe
to launch (and which need a cut fix or codex-schema work) without paying for it.

    uv run python -m rollback.preflight_cuts
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from . import config as c
from .session import recon, recon_codex, recon_opencode


def _prep_outdir(cmds) -> str | None:
    """Last model-output dir named in the prep commands (best-effort parse)."""
    outs = []
    for cmd in cmds or []:
        for m in re.finditer(r"(?:--final-output|--output|-o)\s+\.?/?(\S+)", cmd):
            outs.append(m.group(1).strip("./"))
    return outs[-1] if outs else None


def _recon_for(traj):
    return {"opencode": recon_opencode, "claude": recon,
            "codex": recon_codex}.get(traj.scaffold)


def preflight_one(rid: str, traj) -> dict:
    cut = c.CUT_OVERRIDES.get(rid, traj.default_cut)
    spec = c.ExperimentSpec(traj, "backward", "prompt1", cut)
    recon_status, vcut, detail = "?", "-", ""
    mod = _recon_for(traj)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            meta = mod.build_session(spec, Path(tmp))
            recon_status = "ok"
            cv = (meta or {}).get("cut_validation")
            if cv is not None:
                vcut = "ok" if cv.get("ok") else "FAIL"
        except NotImplementedError as e:
            recon_status = "blocked"
            detail = str(e).splitlines()[0][:70]
        except Exception as e:  # any recon failure = an invalid cut/boundary
            recon_status = "ERROR"
            detail = f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:80]}"

    cmds, src = c.effective_prep_commands(traj)
    pdir = _prep_outdir(cmds)
    edir = (traj.eval_model_dir or "").strip("./")
    if not cmds:
        prep = "none"
    elif pdir is None or pdir == edir:
        prep = "ok"
    else:
        prep = f"MISMATCH({pdir}!={edir})"

    return {"scaffold": traj.scaffold, "cut": cut, "recon": recon_status,
            "vcut": vcut, "prep": prep, "prep_src": src, "run_name": traj.run_name,
            "detail": detail}


def main() -> None:
    rows = [preflight_one(rid, t) for rid, t in c.ALL_TRAJECTORIES.items()]
    rows.sort(key=lambda r: (r["scaffold"], r["recon"] != "ok", r["run_name"]))
    print(f"{'scaffold':<9} {'cut':>5} {'recon':<8} {'vcut':<5} {'prep':<22} {'run_name'}")
    print("-" * 120)
    for r in rows:
        line = (f"{r['scaffold']:<9} {r['cut']!s:>5} {r['recon']:<8} {r['vcut']:<5} "
                f"{r['prep']:<22} {r['run_name']}")
        print(line)
        if r["detail"]:
            print(f"{'':>52}↳ {r['detail']}")
    # summary
    print("-" * 120)
    n = len(rows)
    ok = sum(1 for r in rows if r["recon"] == "ok" and r["vcut"] in ("ok", "-"))
    err = [r for r in rows if r["recon"] == "ERROR"]
    blocked = [r for r in rows if r["recon"] == "blocked"]
    vfail = [r for r in rows if r["vcut"] == "FAIL"]
    mismatch = [r for r in rows if r["prep"].startswith("MISMATCH")]
    print(f"{n} trajectories | recon-ok {ok} | recon-ERROR {len(err)} | "
          f"blocked {len(blocked)} | validate_cut-FAIL {len(vfail)} | prep-dir MISMATCH {len(mismatch)}")
    for tag, group in (("ERROR", err), ("validate_cut FAIL", vfail),
                       ("blocked", blocked), ("prep MISMATCH", mismatch)):
        for r in group:
            print(f"  [{tag}] {r['scaffold']}/{r['run_name']}  cut={r['cut']}  {r['detail']}")


if __name__ == "__main__":
    main()
