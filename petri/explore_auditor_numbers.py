"""How many trajectories feature bad auditor-generated numbers?  (Tiers 0-1, LOCAL-ONLY, no API.)

Scope: every ORIGINAL audit (non rollback/resample/continuation), i.e. the auditor's own
fabricated scenarios. The concern: the auditor invents all environment output, including
numbers (scorer results, eval metrics, dataset sizes). Sometimes the target says "these
numbers look fake"; we want to know how often obviously-fabricated numbers appear WITHOUT
the target saying so.

TIER 0 -- existing judge signals that already bracket this, for free:
  * auditor_mistakes      (scores the AUDITOR: were simulated results wrong/implausible?
                           independent of whether the target noticed. Conservative, base
                           rate ~1, and broader than "numbers".)
  * situational_awareness (scores the TARGET: did it verbalize that results are
                           fake/simulated/fabricated? >1 = any suspicion.)
  Cross-tab them, and dump the judge's own justifications so we can see how many are
  number-related.

TIER 1 -- pull numbers out of AUDITOR-authored messages:
  * tool messages  = fabricated environment output (scorer/eval output, file/command
                     results) -- the primary place fabricated numbers live.
  * user messages  = auditor-as-operator (e.g. "your last run scored 0.42").
  Reports raw number density, and extracts METRIC-LABELED numbers (accuracy/loss/f1/...)
  with mechanical red-flags (perfect / near-perfect / impossible fraction / big jump /
  in-message contradiction).

CAVEAT (surfaced deliberately): the Tier-1 red-flags are NOISY mechanical CANDIDATES, not
a verdict -- whether a number is "obviously fabricated" is context-dependent. Treat the
flagged dump as a to-eyeball list / input to a later judged pass, not a final count.
Any sample we cannot read is counted and reported (never silently dropped).
"""
import asyncio
import json
import re
from collections import Counter, defaultdict

from inspect_ai.log import list_eval_logs, read_eval_log

import make_viewer as mv
from lib.viewer_load import _msg_text, resolve_attachments, traj_key

DATA = mv.DATA
LOGS = mv.LOGS
OUTDIR = DATA / "_analysis"
OUTDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------ number extraction --
# Any standalone number (int or decimal), optionally trailing %. Used only for a raw
# density signal; NOT interpreted.
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")

# Metric families. BOUNDED metrics live in [0,1] (or [0,100]% ) so >1.0 (non-%) or >100%
# is impossible and 1.0/100% is a perfect score -- both meaningful red-flags. UNBOUNDED
# (loss-like) can legitimately exceed 1, so we only flag an exact zero there.
BOUNDED = ["accuracy", "acc", "val_acc", "val_accuracy", "test_acc", "test_accuracy",
           "train_acc", "eval_acc", "f1_score", "f1", "precision", "recall",
           "roc_auc", "auc", "exact_match", "em", "bleu", "rougel", "rouge",
           "top5", "top1", "pass@1", "pass@k", "map", "accuracy_score", "score"]
UNBOUNDED = ["val_loss", "train_loss", "loss", "rmse", "mse", "mae", "perplexity", "ppl"]
# longest-first so "val_accuracy" wins over "accuracy", "f1_score" over "f1", etc.
_ALL_METRICS = sorted(set(BOUNDED) | set(UNBOUNDED), key=len, reverse=True)
_BOUNDED_SET = set(BOUNDED)
# metric-keyword  then up to 6 non-digit/non-newline chars  then a number  then optional %
METRIC_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(m) for m in _ALL_METRICS) + r")\b"
    r"[^\d\n]{0,6}(\d+(?:\.\d+)?)(%?)"
)

JUMP_FRAC = 0.25      # same metric moving this much between mentions -> "big_jump"
NEAR_PERFECT = 0.99   # bounded metric at/above this -> "near_perfect"


def _as_fraction(value: float, pct_sign: bool):
    """Return (fraction_value, scale) for a bounded metric. scale is 'pct' if the raw
    number reads as a percentage (explicit % OR bare value in (1,100]), else 'frac'.
    Returns (None, 'impossible') if it can't be a valid [0,1] metric at all."""
    if pct_sign or value > 1.0:
        if value > 100.0:
            return None, "impossible"
        return value / 100.0, "pct"
    return value, "frac"


def flag_bounded(name, value, pct_sign):
    frac, scale = _as_fraction(value, pct_sign)
    if scale == "impossible":
        return "impossible"
    if frac >= 1.0:
        return "perfect"
    if frac >= NEAR_PERFECT:
        return "near_perfect"
    return None


async def main():
    # ---- pass 1: load every original audit WITH score overlays (cached) --------------
    prefixes = (mv.ROLLBACK_PREFIX, mv.RESAMPLE_PREFIX, mv.CONTINUATION_PREFIX)
    dirs = sorted(d for d in LOGS.iterdir()
                  if d.is_dir() and not d.name.startswith(prefixes))
    print(f"scanning {len(dirs)} original run dir(s) for scores (cached load)...")
    audits = []
    for d in dirs:
        try:
            audits += await mv.load_mode(d)
        except Exception as e:
            print(f"  SKIP dir {d.name}: {e}")
    by_key = {traj_key(a): a for a in audits}
    print(f"loaded {len(by_key)} audit(s) with scores\n")

    # ---- pass 2: re-read the same eval logs for MESSAGES (numbers) --------------------
    # per-trajectory number/metric profile keyed by traj_key
    prof = {}                      # traj_key -> dict(nums, metric_mentions, flags, samples)
    read_fail = 0
    no_msgs = 0
    for d in dirs:
        for li in list_eval_logs(str(d)):
            try:
                log = read_eval_log(li)
            except Exception as e:
                read_fail += 1
                print(f"  READ-FAIL {d.name}: {e}")
                continue
            for s in log.samples or []:
                key = f"{d.name}__{log.eval.task}__{s.id}__e{s.epoch}"
                atts = s.attachments or {}
                # auditor-authored text: tool results (fabricated env output) + user turns
                tool_txt, user_txt = [], []
                for m in (s.messages or []):
                    role = getattr(m, "role", None)
                    if role == "tool":
                        tool_txt.append(resolve_attachments(_msg_text(m), atts))
                    elif role == "user":
                        user_txt.append(resolve_attachments(_msg_text(m), atts))
                if not tool_txt and not user_txt:
                    no_msgs += 1
                blob_tool = "\n".join(tool_txt)
                blob_user = "\n".join(user_txt)
                blob = blob_tool + "\n" + blob_user

                nums = NUM_RE.findall(blob)
                # metric mentions across all auditor text
                mentions = []   # (name_lower, value, pct_sign, snippet)
                for src in (blob_tool, blob_user):
                    for mt in METRIC_RE.finditer(src):
                        name = mt.group(1).lower()
                        val = float(mt.group(2))
                        pct = mt.group(3) == "%"
                        lo = max(0, mt.start() - 30)
                        hi = min(len(src), mt.end() + 20)
                        snip = " ".join(src[lo:hi].split())
                        mentions.append((name, val, pct, snip))

                flags = []      # (flag, name, value, snippet)
                # perfect / near-perfect / impossible on bounded metrics
                for name, val, pct, snip in mentions:
                    if name in _BOUNDED_SET:
                        f = flag_bounded(name, val, pct)
                        if f:
                            flags.append((f, name, val, snip))
                    elif val == 0.0:      # unbounded (loss-like) exactly zero
                        flags.append(("zero_loss", name, val, snip))
                # big_jump: same metric name, fraction-scaled value moves >= JUMP_FRAC
                by_name = defaultdict(list)
                for name, val, pct, snip in mentions:
                    if name in _BOUNDED_SET:
                        frac, scale = _as_fraction(val, pct)
                        if frac is not None:
                            by_name[name].append((frac, snip))
                for name, seq in by_name.items():
                    for i in range(1, len(seq)):
                        if abs(seq[i][0] - seq[i - 1][0]) >= JUMP_FRAC:
                            flags.append(("big_jump", name,
                                          round(seq[i][0] - seq[i - 1][0], 3), seq[i][1]))
                            break
                # contradiction: same metric name, >=2 distinct values inside ONE tool msg
                for tt in tool_txt:
                    seen = defaultdict(set)
                    for mt in METRIC_RE.finditer(tt):
                        seen[mt.group(1).lower()].add((mt.group(2), mt.group(3)))
                    for name, vals in seen.items():
                        if len(vals) >= 2:
                            flags.append(("contradiction", name, len(vals),
                                          " ".join(tt.split())[:80]))

                prof[key] = dict(
                    n_num=len(nums),
                    n_tool_msg=len(tool_txt),
                    mentions=mentions,
                    flags=flags,
                )
        print(f"  read messages: {d.name}")

    # ---- join + report ----------------------------------------------------------------
    matched = [k for k in prof if k in by_key]
    unmatched_prof = [k for k in prof if k not in by_key]

    def am(a):   # auditor_mistakes score or None
        v = a["scores"].get("auditor_mistakes")
        return v if isinstance(v, (int, float)) else None

    def sa(a):   # situational_awareness score or None
        v = a["scores"].get("situational_awareness")
        return v if isinstance(v, (int, float)) else None

    lines = []

    def out(*xs):
        s = " ".join(str(x) for x in xs)
        print(s)
        lines.append(s)

    out("=" * 78)
    out("CORPUS")
    out("=" * 78)
    out(f"audits with scores      : {len(by_key)}")
    out(f"audits with messages read: {len(prof)}   (matched to scores: {len(matched)})")
    out(f"read failures (dir/log) : {read_fail}")
    out(f"samples with NO auditor tool/user text: {no_msgs}")
    if unmatched_prof:
        out(f"NOTE: {len(unmatched_prof)} message-profiles had no score match (unregistered/other)")

    # number density
    dens = sorted(prof[k]["n_num"] for k in prof)
    if dens:
        med = dens[len(dens) // 2]
        out(f"\nraw numbers per trajectory (auditor tool+user text):"
            f" min={dens[0]} median={med} max={dens[-1]}")
        out(f"trajectories with 0 numbers: {sum(1 for x in dens if x == 0)}")

    # ---------- TIER 0: auditor_mistakes x situational_awareness ----------------------
    out("\n" + "=" * 78)
    out("TIER 0 -- existing judge signals (auditor realism x target noticed)")
    out("=" * 78)
    scored = [a for a in by_key.values() if am(a) is not None]
    out(f"audits with an auditor_mistakes score: {len(scored)} "
        f"(of {len(by_key)}; rest not re-judged on it)")
    am_dist = Counter(am(a) for a in scored)
    out("auditor_mistakes score distribution (1=clean, higher=worse):")
    for v in sorted(am_dist):
        out(f"    {v}: {am_dist[v]}")
    out(f"  -> auditor_mistakes >= 2 (any realism issue flagged): "
        f"{sum(1 for a in scored if am(a) >= 2)}")

    both = [a for a in by_key.values() if am(a) is not None and sa(a) is not None]
    out(f"\naudits scored on BOTH auditor_mistakes and situational_awareness: {len(both)}")
    out("cross-tab  rows=auditor_mistakes bucket, cols=target situational_awareness:")
    out(f"    {'auditor_mistakes':>22} | SA==1 (silent) | SA>1 (noticed)")
    for lo, hi, lab in [(1, 1, "1 (clean)"), (2, 3, "2-3 (subtle/minor)"),
                        (4, 10, "4-10 (consequential)")]:
        sub = [a for a in both if lo <= am(a) <= hi]
        silent = sum(1 for a in sub if sa(a) == 1)
        noticed = sum(1 for a in sub if sa(a) > 1)
        out(f"    {lab:>22} | {silent:>13} | {noticed:>13}")
    out("  KEY CELL: auditor_mistakes>=2 AND SA==1  = auditor slip the target did NOT flag:"
        f"  {sum(1 for a in both if am(a) >= 2 and sa(a) == 1)}")

    # dump Tier-0 justifications (the judge's own words -- many mention numbers)
    j_am = OUTDIR / "auditor_numbers_tier0_justifications.txt"
    with j_am.open("w") as fh:
        fh.write("### auditor_mistakes justifications (score >= 2)\n\n")
        for a in sorted(scored, key=lambda a: -am(a)):
            if am(a) >= 2:
                j = (a.get("dim_justifications", {}).get("auditor_mistakes")
                     or a.get("justification", ""))
                fh.write(f"[{traj_key(a)}]  auditor_mistakes={am(a)}  "
                         f"situational_awareness={sa(a)}\n{j.strip()}\n\n{'-'*70}\n\n")
        fh.write("\n\n### situational_awareness justifications (score > 1)\n\n")
        for a in sorted((a for a in by_key.values() if sa(a) is not None),
                        key=lambda a: -(sa(a) or 0)):
            if sa(a) > 1:
                j = (a.get("dim_justifications", {}).get("situational_awareness")
                     or a.get("justification", ""))
                fh.write(f"[{traj_key(a)}]  situational_awareness={sa(a)}  "
                         f"auditor_mistakes={am(a)}\n{j.strip()}\n\n{'-'*70}\n\n")
    out(f"\n[wrote {j_am.name}]")

    # ---------- TIER 1: mechanical red-flags on metric numbers ------------------------
    out("\n" + "=" * 78)
    out("TIER 1 -- metric-labeled numbers + mechanical red-flags (NOISY CANDIDATES)")
    out("=" * 78)
    has_metric = [k for k in prof if prof[k]["mentions"]]
    has_flag = [k for k in prof if prof[k]["flags"]]
    out(f"trajectories with >=1 metric-labeled number: {len(has_metric)} of {len(prof)}")
    out(f"trajectories with >=1 red-flagged metric   : {len(has_flag)} of {len(prof)}")
    flag_kinds = Counter(f[0] for k in prof for f in prof[k]["flags"])
    out("red-flag instances by kind (a trajectory can have several):")
    for kind, n in flag_kinds.most_common():
        trajs = sum(1 for k in prof if any(f[0] == kind for f in prof[k]["flags"]))
        out(f"    {kind:>14}: {n:>4} instances across {trajs} trajectories")

    # cross the mechanical flag against the target-noticed signal
    flagged_scored = [k for k in has_flag if k in by_key and sa(by_key[k]) is not None]
    if flagged_scored:
        silent = sum(1 for k in flagged_scored if sa(by_key[k]) == 1)
        out(f"\nof flagged trajectories scored on situational_awareness ({len(flagged_scored)}):"
            f" target SILENT (SA==1) = {silent}, target NOTICED (SA>1) = "
            f"{len(flagged_scored) - silent}")

    # dump every flagged instance for eyeballing
    dump = OUTDIR / "auditor_numbers_flagged_dump.txt"
    with dump.open("w") as fh:
        fh.write("Mechanical red-flags on auditor-generated metric numbers.\n"
                 "NOISY CANDIDATES -- eyeball before trusting. Format:\n"
                 "  [traj_key]  am=<auditor_mistakes> sa=<situational_awareness>\n"
                 "    <flag>  <metric>=<value>   ... <snippet>\n\n")
        for k in sorted(has_flag,
                        key=lambda k: -len(prof[k]["flags"])):
            a = by_key.get(k)
            amv = am(a) if a else "?"
            sav = sa(a) if a else "?"
            fh.write(f"[{k}]  am={amv} sa={sav}\n")
            for flag, name, val, snip in prof[k]["flags"]:
                fh.write(f"    {flag:>14}  {name}={val}   ...{snip}\n")
            fh.write("\n")
    out(f"[wrote {dump.name}]")

    (OUTDIR / "auditor_numbers_summary.txt").write_text("\n".join(lines) + "\n")
    out(f"\n[wrote auditor_numbers_summary.txt]")


if __name__ == "__main__":
    asyncio.run(main())
