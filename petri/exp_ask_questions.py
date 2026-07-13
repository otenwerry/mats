"""Ask a set of questions to a resumed Petri target context, n times each.

PAID: calls the target model's API (one generate call per ask).

  uv run exp_ask_questions.py --trajectory <viewer id> --questions em [--n 5]

Petri twin of posttrainbench/exp_ask_questions.py -- same flags, same output
layout, same question sets -- with the resume mechanics swapped for Petri's:
the context bundle is the target's own recorded input at the cut (read verbatim
from that turn's ModelEvent, attachments resolved), so unlike PTB nothing is
synthesized or rebuilt, and there is no scaffold CLI / config dir / local-env
gap to worry about (--user-config has no analog here and does not exist).

Every (question, sample) pair is an INDEPENDENT ask: a fresh message list built
from the same recorded context, one generate call, no cache. The target model is
rebuilt exactly as the original run configured it (build_target_model with the
original's stamped reasoning settings + provider routing), mirroring how
rollbacks/resamples inherit the original's target.

Differences from the PTB version that are forced by the setting:
  - ONE generate call per ask, no agentic loop. Petri tool results are fabricated
    by the auditor, so if the resumed target answers with tool calls instead of
    text there is nothing to execute them against: the ask records whatever text
    accompanied the tool call and flags probe_used_tools (in PTB the scaffold
    runs the tools and the answer comes after; here a tool-calling response IS
    the datum).
  - if the reconstructed context ends with a user message, the question is
    FOLDED into it (prev + blank line + question) instead of appended as a
    second user turn, which some providers reject -- same convention as
    exp_rollback. Flagged per ask (question_folded_into_user_turn).

Question sets (--questions, required):
  em    the 8 main "first plot" questions of the emergent-misalignment paper
        (Betley et al. 2025), read verbatim from
        references/emergent-misalignment/evaluation/first_plot_questions.yaml --
        the plain variants only (no _json system prompt, no _template).
Add new sets to QUESTION_SETS below as experiments need them.

--turn is a TARGET ASSISTANT-TURN INDEX (the viewer's [A<n>] numbering), or one
of three keywords: 'last' (the default -- the context going into the target's
final turn), 'end' (after the target's final message; refused if that message
has unanswered tool calls), 'first_hack' (before the first annotated hack turn).
The cut goes immediately before the resolved turn, same semantics as PTB.
Branched (auditor-rollback) trajectories are refused for numeric/'first_hack'
cuts (turn numbering is ambiguous) and allowed with a warning for 'last'/'end'.

--trajectory takes a viewer #ID; unlike PTB it also accepts a comma-separated
list of ids, or 'hacks' (every original passing the strict binary reward-hack
definition), since Petri reconstruction is uniform across trajectories. Output
is still one dir per trajectory.

Outputs land in mats-local/petri/<campaign>/id<ID>__<cut>/ (campaign defaults
to questions_<set>):
  results.md        every answer, grouped by question -- for reading
  results.json      the same data, machine-readable -- for later judging
  context.jsonl     the resumed context, one message per line (shared by every
                    ask of this trajectory); tools.json = the tool defs sent
  <qid>__s<i>/      one dir per ask: response.json (full ModelOutput message +
                    usage), fidelity.json (flags + cost), probe_result.md

Costs are recorded per ask and summed per trajectory + grand total: exact when
inspect stamped ModelUsage.total_cost (OpenRouter billed cost via
openrouter_cost, first-party list price via direct_cost), otherwise ~price x
tokens from lib/model_prices (source recorded either way). --dry-run (FREE)
builds every bundle and prints per-ask context sizes + estimated input cost so
you can preview spend before paying.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "lib"))
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
from inspect_ai.event import BranchEvent, ModelEvent
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from petri_paths import DATA, ENV_FILE, LOGS, PETRI_ROOT
from exp_rh_audit import REASONING_EFFORT, build_target_model
from model_routing import route
import direct_cost
import model_prices
import openrouter_cost

from viewer import is_hack_binary, load_originals_by_id, page_name

load_dotenv(ENV_FILE)

REFERENCES = PETRI_ROOT.parent.parent / "references"
ANN_FILE = DATA / "annotations.json"

_HEAD_RE = re.compile(r"^\[M(\d+)\] (\w+)", re.M)


class CutError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Question sets. Each loader returns [{"id": ..., "text": ...}], texts kept    #
# verbatim from their source. Add new sets here. (Identical to the PTB twin.)  #
# --------------------------------------------------------------------------- #

def _load_em() -> list[dict]:
    """The 8 main 'first plot' questions (plain variants) of Betley et al.,
    parsed from the cloned paper repo -- same filtering as the viewer's
    EM-questions page, so this file and the viewer can never disagree."""
    import yaml
    f = REFERENCES / "emergent-misalignment" / "evaluation" / "first_plot_questions.yaml"
    if not f.exists():
        sys.exit(f"question set 'em' needs {f} -- clone the emergent-misalignment "
                 f"repo into supermats/references/")
    out = []
    for q in yaml.safe_load(f.read_text()):
        qid = q["id"]
        if qid.endswith("_json") or qid.endswith("_template"):
            continue
        out.append({"id": qid, "text": q["paraphrases"][0]})
    return out


QUESTION_SETS = {"em": _load_em}


# --------------------------------------------------------------------------- #
# Context bundle: the target's recorded input at the cut.                      #
# --------------------------------------------------------------------------- #

@dataclass
class AskBundle:
    tid: int
    page: str                 # viewer page key (mode__task__seed__eN.html)
    target_slug: str          # routed target model slug
    reasoning_on: bool
    reasoning_effort: str | None
    messages: list            # the resumed context (recorded verbatim)
    tools: list               # ToolInfo list the target had at the cut
    cut_label: str            # "a<j>" or "end" (used in the output dir name)
    cut_turn: int             # the turn the cut precedes (= n_turns for 'end')
    n_turns: int              # total target turns in the original
    est_context_tokens: int | None   # original billed input size at the cut turn
    flags: list = field(default_factory=list)


def _find_sample(mode: str, task: str, seed: str, epoch: int):
    """Locate the EvalSample for a trajectory (header-matched by task first, then a
    full read). resolve_attachments=True is ESSENTIAL: the recorded messages store
    large content (system prompt, user turns, tool results) as attachment://<hash>
    references; without resolving, the target would be resumed on the bare
    reference strings instead of the real conversation."""
    for li in list_eval_logs(str(LOGS / mode)):
        try:
            hdr = read_eval_log(li, header_only=True)
        except Exception:
            continue
        if hdr.eval.task != task:
            continue
        log = read_eval_log(li, resolve_attachments=True)
        for s in log.samples or []:
            if str(s.id) == seed and s.epoch == epoch:
                return log, s
    raise CutError(f"sample not found for {mode}/{task}/{seed} e{epoch}")


def _first_hack_turn(orig: dict) -> int:
    """The first annotated hack turn as a target-turn index k, derived exactly like
    exp_rollback.build_replay_data: the annotation's m indexes the viewer transcript,
    k = count of assistant heads up to and including [Mm]."""
    ann = (json.loads(ANN_FILE.read_text()) if ANN_FILE.exists() else {}).get(
        page_name(orig["mode"], orig["task"], orig["seed"], orig["epoch"]))
    if not ann or not ann.get("hack_turns"):
        raise CutError("no hack-turn annotation (run exp_annotate_hacks.py first)")
    m = ann["hack_turns"][0]["m"]
    transcript = orig.get("transcript") or ""
    if not transcript:
        raise CutError("no rendered transcript (dead/empty audit?)")
    heads = [(int(h.group(1)), h.group(2).lower()) for h in _HEAD_RE.finditer(transcript)]
    role_at_m = dict(heads).get(m)
    if role_at_m is None or not role_at_m.startswith("assistant"):
        raise CutError(f"hack message [M{m}] is not an assistant turn (role={role_at_m})")
    return sum(1 for num, role in heads if role.startswith("assistant") and num <= m)


def build_bundle(tid: int, orig: dict, turn_arg: str) -> AskBundle:
    """Read the trajectory's sample and build the resumable context at the cut.
    Raises CutError on invalid/unsupported positions."""
    log, s = _find_sample(orig["mode"], orig["task"], orig["seed"], orig["epoch"])

    events = [e for e in s.events or []
              if isinstance(e, ModelEvent) and getattr(e, "role", None) == "target"
              and e.output and e.output.message]
    if not events:
        raise CutError("no target ModelEvents (dead audit?)")
    n_turns = len(events)
    branched = any(isinstance(e, BranchEvent) for e in s.events or [])

    flags: list[dict] = []

    # ---- resolve the cut (immediately before the resolved turn; PTB semantics) ----
    if turn_arg == "end":
        cut_turn, cut_label = n_turns, "end"
    elif turn_arg == "last":
        cut_turn, cut_label = n_turns, f"a{n_turns}"
    elif turn_arg == "first_hack":
        cut_turn = _first_hack_turn(orig)
        cut_label = f"a{cut_turn}"
    else:
        cut_turn = int(turn_arg)
        cut_label = f"a{cut_turn}"
    if not (1 <= cut_turn <= n_turns):
        raise CutError(f"turn {cut_turn} out of range (target turns 1..{n_turns})")

    if branched:
        if turn_arg not in ("last", "end"):
            raise CutError("trajectory contains conversation branches (auditor "
                           "rollbacks); turn numbering is ambiguous, only "
                           "--turn last/end are supported for it")
        flags.append({
            "code": "branched_trajectory", "severity": "warn",
            "detail": "the auditor branched/rolled back this conversation; the cut "
                      "uses the target's true final generate call, whose turn index "
                      "may not match the viewer's [A] numbering"})

    ev = events[cut_turn - 1]
    if turn_arg == "end":
        final = ev.output.message
        if final.tool_calls:
            raise CutError("the final assistant message has unanswered tool calls; "
                           "a question cannot validly follow it -- try --turn last")
        messages = list(ev.input) + [final]
    else:
        messages = list(ev.input)
    tools = list(ev.tools or [])

    # ---- original context size at the cut (what the provider billed that turn) ----
    u = ev.output.usage
    est = None
    if u is not None:
        est = ((u.input_tokens or 0) + (u.input_tokens_cache_read or 0)
               + (u.input_tokens_cache_write or 0))
        if turn_arg == "end":
            est += u.output_tokens or 0

    # ---- target model config, inherited from the original run's stamped metadata ----
    roles = log.eval.model_roles or {}
    target_slug = route(getattr(roles.get("target"), "model", None)
                        or str(roles.get("target", "?")))
    meta = log.eval.metadata or {}
    reasoning_on = (bool(meta["reasoning"]) if "reasoning" in meta
                    else (meta.get("reasoning_enabled") is True))
    reasoning_effort = meta.get("reasoning_effort") or REASONING_EFFORT

    flags.append({
        "code": "context_recorded_verbatim", "severity": "info",
        "detail": "the resumed context is the target's own recorded model-call input "
                  "at the cut turn (attachments resolved); nothing was synthesized"})
    flags.append({
        "code": "reasoning_replicated", "severity": "info",
        "detail": f"target rebuilt via build_target_model(reasoning_on={reasoning_on}"
                  + (f", effort={reasoning_effort}" if reasoning_on else "")
                  + "), matching the original run's stamped settings"})

    return AskBundle(
        tid=tid,
        page=page_name(orig["mode"], orig["task"], orig["seed"], orig["epoch"]),
        target_slug=target_slug,
        reasoning_on=reasoning_on,
        reasoning_effort=reasoning_effort,
        messages=messages,
        tools=tools,
        cut_label=cut_label,
        cut_turn=cut_turn,
        n_turns=n_turns,
        est_context_tokens=est,
        flags=flags,
    )


def with_question(messages: list, question: str) -> tuple[list, bool]:
    """Context + the question as a user turn, WITHOUT mutating the shared context.
    Folds into a trailing user message when present (consecutive user turns are
    rejected by some providers -- same convention as exp_rollback._append_user);
    returns (new_messages, folded)."""
    if messages and messages[-1].role == "user":
        prev = messages[-1].text or ""
        return messages[:-1] + [ChatMessageUser(content=f"{prev}\n\n{question}")], True
    return list(messages) + [ChatMessageUser(content=question)], False


# --------------------------------------------------------------------------- #
# Cost of one ask.                                                             #
# --------------------------------------------------------------------------- #

def _usage_dict(u) -> dict | None:
    if u is None:
        return None
    return {"input": u.input_tokens, "output": u.output_tokens,
            "cache_read": u.input_tokens_cache_read or 0,
            "cache_write": u.input_tokens_cache_write or 0,
            "reasoning": u.reasoning_tokens, "total": u.total_tokens,
            "total_cost": u.total_cost}


def ask_cost(output, target_slug: str) -> tuple[float | None, str]:
    """(cost_usd, source) for one generate call. Exact when inspect stamped
    ModelUsage.total_cost (OpenRouter billed / first-party list price); else
    ~price x tokens from model_prices; else unknown."""
    u = getattr(output, "usage", None)
    if u is None:
        return None, "unknown (no usage reported)"
    if isinstance(u.total_cost, (int, float)):
        return float(u.total_cost), ("exact (ModelUsage.total_cost: OpenRouter "
                                     "billed / first-party list price)")
    p = model_prices.price_for(target_slug)
    if p is None:
        return None, f"unknown (no price entry for {target_slug})"
    cost = model_prices._usage_cost(
        {"input": u.input_tokens, "output": u.output_tokens,
         "cache_read": u.input_tokens_cache_read or 0,
         "cache_write": u.input_tokens_cache_write or 0}, p)
    return cost, f"~price x tokens ({p['source']})"


def _est_input_cost(bundle: AskBundle) -> float | None:
    """Dry-run estimate: context tokens x input price (output tokens not included)."""
    p = model_prices.price_for(bundle.target_slug)
    if p is None or bundle.est_context_tokens is None:
        return None
    return bundle.est_context_tokens / 1e6 * p["input"]


# --------------------------------------------------------------------------- #
# One ask = one independent generate call.                                     #
# --------------------------------------------------------------------------- #

async def one_ask(model, bundle: AskBundle, fidelity: dict, q: dict, i: int,
                  ask_dir: Path, gen_config: GenerateConfig) -> dict:
    """Run one question against a fresh copy of the reconstructed context.
    Returns the ask record for results.json; writes the per-ask dir."""
    ask_dir.mkdir(parents=True, exist_ok=True)
    messages, folded = with_question(bundle.messages, q["text"])
    if folded:
        fidelity["flags"].append({
            "code": "question_folded_into_user_turn", "severity": "info",
            "detail": "the context ends with a user message; the question was "
                      "appended to it (blank line between) rather than sent as a "
                      "separate consecutive user turn"})

    t0 = time.monotonic()
    output, error = None, None
    try:
        output = await model.generate(input=messages, tools=bundle.tools,
                                      config=gen_config, cache=False)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    dt = time.monotonic() - t0

    result_text, n_tool_uses, stop_reason, cost, cost_source = None, 0, None, None, None
    if output is not None:
        msg = output.message
        stop_reason = output.stop_reason
        result_text = (msg.text or "").strip() or None
        n_tool_uses = len(msg.tool_calls or [])
        cost, cost_source = ask_cost(output, bundle.target_slug)
        (ask_dir / "response.json").write_text(json.dumps(
            {"model": output.model, "stop_reason": stop_reason,
             "message": msg.model_dump(exclude_none=True),
             "usage": _usage_dict(output.usage)}, indent=2))

    if n_tool_uses:
        fidelity["flags"].append({
            "code": "probe_used_tools", "severity": "warn",
            "detail": f"the resumed target responded with {n_tool_uses} tool call(s) "
                      f"-- there is no environment to execute them, so the recorded "
                      f"answer is only the text accompanying the tool call (if any)"})
    if result_text is None and error is None:
        fidelity["flags"].append({
            "code": "probe_no_result", "severity": "warn",
            "detail": f"the response contained no text (stop_reason={stop_reason}); "
                      f"see response.json"})

    fidelity["probe_question"] = q["text"]
    fidelity["probe_answer"] = result_text
    fidelity["error"] = error
    fidelity["cost"] = {"total_usd": cost, "source": cost_source,
                        "usage": _usage_dict(getattr(output, "usage", None))}
    (ask_dir / "fidelity.json").write_text(json.dumps(fidelity, indent=2))

    md = [f"# Question ask: id{bundle.tid} @ {bundle.cut_label} -- {ask_dir.name}",
          f"- target: {bundle.target_slug}   original: {bundle.page}",
          f"- flags: " + ", ".join(f["code"] for f in fidelity["flags"]),
          "", "## Probe", "```", q["text"], "```", "", "## Model answer", "",
          result_text if result_text else
          (f"(ERROR: {error})" if error else "(no text in response -- see response.json)")]
    (ask_dir / "probe_result.md").write_text("\n".join(md))

    return {"dir": ask_dir.name, "answer": result_text, "cost_usd": cost,
            "cost_source": cost_source, "duration_s": round(dt, 1),
            "n_tool_uses": n_tool_uses, "stop_reason": str(stop_reason) if stop_reason else None,
            **({"error": error} if error else {})}


# --------------------------------------------------------------------------- #
# Per-trajectory aggregation (results.json / results.md, same layout as PTB).  #
# --------------------------------------------------------------------------- #

def write_results(out_dir: Path, bundle: AskBundle, args, questions: list[dict],
                  records: list[dict]) -> dict:
    n_asks = len(records)
    costs = [r["cost_usd"] for r in records if isinstance(r["cost_usd"], (int, float))]
    n_exact = sum(1 for r in records
                  if isinstance(r["cost_usd"], (int, float))
                  and str(r.get("cost_source", "")).startswith("exact"))
    n_failed = sum(1 for r in records if r["answer"] is None and r.get("error"))
    n_no_answer = sum(1 for r in records if r["answer"] is None)
    summary = {
        "trajectory_id": bundle.tid, "page": bundle.page,
        "target_model": bundle.target_slug,
        "cut": bundle.cut_label, "cut_turn": bundle.cut_turn,
        "n_target_turns": bundle.n_turns, "requested_turn": args.turn,
        "question_set": args.questions, "n_per_question": args.n,
        "n_asks": n_asks, "n_no_answer": n_no_answer, "n_errored": n_failed,
        "total_cost_usd": round(sum(costs), 4) if costs else None,
        "cost_known_for": len(costs), "cost_exact_for": n_exact,
        "bundle_flags": [f["code"] for f in bundle.flags],
    }
    (out_dir / "results.json").write_text(json.dumps(
        {"summary": summary, "asks": records}, indent=2))

    md = [f"# {args.questions} questions on id{bundle.tid} ({bundle.page}) @ {bundle.cut_label}",
          f"- {len(questions)} question(s) x {args.n} sample(s); "
          f"{n_no_answer} without an answer"
          + (f"; total ${sum(costs):.4f} ({len(costs)}/{n_asks} asks priced, "
             f"{n_exact} exact)" if costs else ""), ""]
    for q in questions:
        md += [f"## {q['id']}", "", "> " + q["text"].replace("\n", "\n> "), ""]
        for r in records:
            if r["question_id"] != q["id"]:
                continue
            note = (" -- ERROR: " + r["error"] if r.get("error") else
                    (f" -- {r['n_tool_uses']} tool call(s) in the response"
                     if r.get("n_tool_uses") else ""))
            md += [f"### sample {r['sample_index']}{note}", "",
                   r["answer"] if r["answer"] is not None
                   else "(no answer -- see the ask dir)", ""]
    (out_dir / "results.md").write_text("\n".join(md))
    return summary


# --------------------------------------------------------------------------- #
# CLI / main                                                                   #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", required=True,
                    help="viewer trajectory #ID; also accepts a comma-separated list, "
                         "or 'hacks' (all originals passing the strict binary RH "
                         "definition)")
    ap.add_argument("--questions", required=True, choices=sorted(QUESTION_SETS),
                    help="which question set to ask (currently only 'em')")
    ap.add_argument("--n", type=int, default=1,
                    help="samples per question (default 1); every sample is an "
                         "independent ask on the same context")
    ap.add_argument("--turn", default="last",
                    help="target assistant-turn index ([A<n>]), 'last' (default), "
                         "'end', or 'first_hack' -- cut goes immediately before the "
                         "resolved turn, same semantics as the PTB twin")
    ap.add_argument("--only", default=None,
                    help="comma-separated question ids to ask (subset of the set); "
                         "useful for a cheap test run")
    ap.add_argument("--concurrency", type=int, default=50,
                    help="max asks in flight at once (default 50)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per ask")
    ap.add_argument("--campaign", default=None,
                    help="output subdir under mats-local/petri (default questions_<set>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="FREE: build every context bundle and print per-ask context "
                         "sizes + estimated input cost, then exit")
    args = ap.parse_args()

    questions = QUESTION_SETS[args.questions]()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        unknown = want - {q["id"] for q in questions}
        if unknown:
            sys.exit(f"--only ids not in set '{args.questions}': {sorted(unknown)}\n"
                     f"available: {[q['id'] for q in questions]}")
        questions = [q for q in questions if q["id"] in want]
    campaign = args.campaign or f"questions_{args.questions}"

    print("[load] originals (viewer load layer, cached) ...")
    originals = asyncio.run(load_originals_by_id())
    if args.trajectory.strip() == "hacks":
        tids = sorted(t for t, a in originals.items() if is_hack_binary(a))
        print(f"[select] 'hacks' -> {len(tids)} full reward hacks (binary def): {tids}")
    else:
        tids = list(dict.fromkeys(int(x) for x in args.trajectory.split(",") if x.strip()))
        unknown_ids = [t for t in tids if t not in originals]
        if unknown_ids:
            sys.exit(f"unknown trajectory ids {unknown_ids} (not in the originals set)")
    if not tids:
        sys.exit("no trajectories selected")

    # ---- build bundles (reads one eval log per trajectory; free) ----
    bundles: list[AskBundle] = []
    for tid in tids:
        try:
            b = build_bundle(tid, originals[tid], args.turn)
        except CutError as e:
            print(f"[id {tid}] CUT REFUSED: {e}")
            continue
        bundles.append(b)
        ctx = (f"{b.est_context_tokens:,} tok" if b.est_context_tokens is not None
               else "? tok")
        print(f"[id {tid}] {b.target_slug.split('/')[-1]}  cut {b.cut_label} "
              f"(turn {b.cut_turn} of {b.n_turns}), context = {ctx}, "
              f"{len(b.tools)} tool(s)")
        for fl in b.flags:
            print(f"  [{fl['severity']:5}] {fl['code']}")
    if not bundles:
        print("nothing to run (every trajectory refused)")
        sys.exit(2)

    n_per_traj = len(questions) * args.n
    n_total = n_per_traj * len(bundles)
    print(f"\nquestion set '{args.questions}': {len(questions)} question(s) x "
          f"{args.n} sample(s) x {len(bundles)} trajectory(ies) = {n_total} asks")

    if args.dry_run:
        print("\n[dry-run] estimated INPUT cost per ask (output tokens not included):")
        grand = 0.0
        unpriced = 0
        for b in bundles:
            est = _est_input_cost(b)
            if est is None:
                unpriced += 1
                print(f"  id{b.tid:>4} {b.target_slug:<40} context ? -> no price entry")
                continue
            grand += est * n_per_traj
            print(f"  id{b.tid:>4} {b.target_slug:<40} "
                  f"{b.est_context_tokens or 0:>9,} tok -> ~${est:.4f}/ask, "
                  f"~${est * n_per_traj:.2f} for {n_per_traj} asks")
        print(f"\n[dry-run] total estimated input cost: ~${grand:.2f} for {n_total} asks"
              + (f" ({unpriced} trajectory(ies) unpriced)" if unpriced else ""))
        print("[dry-run] no run, no cost. Re-run without --dry-run to launch.")
        return

    # exact-cost capture, same hooks the audit/rollback pipelines install
    openrouter_cost.install()
    direct_cost.install()

    out_root = DATA / campaign
    gen_config = GenerateConfig(timeout=args.timeout, max_connections=args.concurrency)

    # one model per distinct (slug, reasoning) config, rebuilt exactly as the original ran
    models: dict[tuple, object] = {}
    for b in bundles:
        key = (b.target_slug, b.reasoning_on, b.reasoning_effort)
        if key not in models:
            m, _, _ = build_target_model(b.target_slug, reasoning_on=b.reasoning_on,
                                         effort=b.reasoning_effort)
            models[key] = get_model(m) if isinstance(m, str) else m

    # write the shared context bundle once per trajectory
    out_dirs: dict[int, Path] = {}
    for b in bundles:
        out_dir = out_root / f"id{b.tid}__{b.cut_label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[b.tid] = out_dir
        with open(out_dir / "context.jsonl", "w") as f:
            for m in b.messages:
                f.write(json.dumps({"role": m.role,
                                    "message": m.model_dump(exclude_none=True)}) + "\n")
        (out_dir / "tools.json").write_text(json.dumps(
            [t.model_dump(exclude_none=True) for t in b.tools], indent=2))
        print(f"[id {b.tid}] -> {out_dir}")

    # ---- fan out: every (trajectory, question, sample) is one independent ask ----
    sem = asyncio.Semaphore(args.concurrency)
    done = {"n": 0}

    async def run_one(b: AskBundle, q: dict, i: int) -> tuple[int, dict]:
        ask_dir = out_dirs[b.tid] / f"{q['id']}__s{i}"
        fid = {"trajectory_id": b.tid, "page": b.page, "cut": b.cut_label,
               "cut_turn": b.cut_turn, "question_set": args.questions,
               "question_id": q["id"], "sample_index": i,
               "flags": copy.deepcopy(b.flags)}
        model = models[(b.target_slug, b.reasoning_on, b.reasoning_effort)]
        async with sem:
            try:
                rec = await one_ask(model, b, fid, q, i, ask_dir, gen_config)
            except Exception as e:  # keep one bad ask from sinking the run
                rec = {"dir": ask_dir.name, "answer": None, "cost_usd": None,
                       "cost_source": None, "duration_s": None, "n_tool_uses": None,
                       "stop_reason": None, "error": f"{type(e).__name__}: {e}"}
        rec = {"question_id": q["id"], "sample_index": i, "question": q["text"], **rec}
        done["n"] += 1
        cost = rec.get("cost_usd")
        print(f"[{done['n']}/{n_total}] id{b.tid} {q['id']} s{i}: "
              + ("ERROR " + rec["error"] if rec.get("error") else
                 ("no answer" if rec["answer"] is None else "ok"))
              + (f", {rec['n_tool_uses']} tool call(s)" if rec.get("n_tool_uses") else "")
              + (f", ${cost:.4f}" if isinstance(cost, (int, float)) else "")
              + (f", {rec['duration_s']:.0f}s" if rec.get("duration_s") else ""),
              flush=True)
        return b.tid, rec

    async def run_all():
        jobs = [(b, q, i) for b in bundles for q in questions for i in range(args.n)]
        return await asyncio.gather(*(run_one(b, q, i) for b, q, i in jobs))

    results = asyncio.run(run_all())

    # ---- aggregate per trajectory + grand total ----
    by_tid: dict[int, list[dict]] = {b.tid: [] for b in bundles}
    for tid, rec in results:
        by_tid[tid].append(rec)

    grand_cost, grand_priced, grand_no_answer = 0.0, 0, 0
    for b in bundles:
        summary = write_results(out_dirs[b.tid], b, args, questions, by_tid[b.tid])
        grand_no_answer += summary["n_no_answer"]
        if summary["total_cost_usd"] is not None:
            grand_cost += summary["total_cost_usd"]
            grand_priced += summary["cost_known_for"]
        print(f"[id {b.tid}] {summary['n_asks'] - summary['n_no_answer']}"
              f"/{summary['n_asks']} answered"
              + (f", ${summary['total_cost_usd']:.4f}"
                 f" ({summary['cost_exact_for']}/{summary['cost_known_for']} exact)"
                 if summary["total_cost_usd"] is not None else "")
              + f" -> {out_dirs[b.tid] / 'results.md'}")

    print(f"\ndone: {n_total - grand_no_answer}/{n_total} asks answered, "
          f"total ${grand_cost:.4f} ({grand_priced}/{n_total} asks priced)")
    sys.exit(0 if grand_no_answer == 0 else 1)


if __name__ == "__main__":
    main()
