"""Build a continuation prefix by running a multi-turn Natural Questions Q&A session.

The target model answers randomly selected questions from the open-domain Natural
Questions set (HF ``google-research-datasets/nq_open``, train split) in ONE growing
conversation, under the exact system prompt and reasoning configuration a real
environment run would use. When the provider-reported context size reaches
``--tokens``, the conversation is saved as an ``environments-continuation-prefix-v1``
payload file, ready for ``exp_continuation_pipeline.py --prefix-files=<path>``.

DETERMINISTIC QUESTIONS, FRESH ANSWERS: question selection and order come from
``--seed`` (fixed default), and no question repeats within a run. Running the same
command three times asks the same questions in the same order and yields three
different prefixes (the model's answers vary), each written to its own
content-addressed file.

Usage (from mats/environments/):
  uv run exp_nq_prefix.py --model=qwen3-32b --tokens=30000
  uv run exp_nq_prefix.py --model=qwen3-32b --tokens=30000 --dry-run   # FREE

Flags:
  --model=<name>       REQUIRED. catalog target name (model_catalog.TARGET_CHOICES).
  --tokens=<N>         REQUIRED. stop once the conversation reaches N tokens, measured
                       as the last call's provider-reported input+output tokens (the
                       true tokenized context size). The final Q&A that crosses the
                       target stays included, so expect a small overshoot.
  --reasoning=yes|no   default yes. Stored in the payload; continuations replicate it.
  --seed=<int>         default 1234. Drives question selection/order only.
  --name=<slug>        payload name; default nq<seed>-<tokens>-<model>-<timestamp>.
                       The default is timestamped so repeated runs never share a name
                       (prefix names must be unique within one continuation run).
  --max-questions=<N>  default 500. Safety cap; if it is hit before --tokens, the
                       payload is still written with reached_target_tokens=false and a
                       loud warning.
  --dry-run            FREE: download/load the dataset, print the selected questions
                       and configuration; no model calls, no files written.

Costs money (one target-provider call per question) unless --dry-run. Token usage and
cost are printed and stored in the payload's source record.
"""

from __future__ import annotations

import asyncio
import pathlib
import random
import re
import sys
from datetime import datetime
from typing import Sequence

_ENVIRONMENTS = pathlib.Path(__file__).resolve().parent
for _p in (str(_ENVIRONMENTS / "lib"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

from cost_tracking import estimate_usage_cost, install_cost_tracking
from exp_real_continuation import (
    PREFIX_FORMAT,
    build_prefix_spec,
    expected_system_prompt,
    store_prefix_payload,
    validate_prefix_payload,
)
from model_catalog import REASONING_EFFORT, TARGET_CHOICES, build_target, resolve_target
from project_paths import DATA_ROOT, ENV_FILE
from protocol_sources import resolve_reasoning


DATASET = "google-research-datasets/nq_open"
SPLIT = "train"
DEFAULT_SEED = 1234
DEFAULT_MAX_QUESTIONS = 500
HF_CACHE = DATA_ROOT / "hf_cache"

_VALUE_FLAGS = {
    "--model", "--tokens", "--reasoning", "--seed", "--name", "--max-questions",
}
_SWITCH_FLAGS = {"--dry-run"}


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for arg in sys.argv[1:]:
        flag, separator, _ = arg.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(f"{flag} requires a value in the form {flag}=<value>")
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {arg!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _int_arg(flag: str, default: int | None) -> int | None:
    raw = _arg(flag)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{flag} must be an integer, got {raw!r}")


def _model_slug_fragment(model_name: str) -> str:
    fragment = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return fragment or "model"


def default_name(model_name: str, tokens: int, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"nq{seed}-{tokens}-{_model_slug_fragment(model_name)}-{stamp}"


def _parse_args() -> dict:
    _validate_cli_args()
    model = _arg("--model")
    if model is None:
        raise SystemExit(f"--model is required; choices: {sorted(TARGET_CHOICES)}")
    if model not in TARGET_CHOICES:
        raise SystemExit(f"unknown --model {model!r}; choices: {sorted(TARGET_CHOICES)}")
    tokens = _int_arg("--tokens", None)
    if tokens is None or tokens < 1:
        raise SystemExit("--tokens is required (positive target context size)")
    seed = _int_arg("--seed", DEFAULT_SEED)
    max_questions = _int_arg("--max-questions", DEFAULT_MAX_QUESTIONS)
    if max_questions < 1:
        raise SystemExit(f"--max-questions must be >= 1, got {max_questions}")
    reasoning = resolve_reasoning(_arg("--reasoning"))
    return {
        "model": model,
        "tokens": tokens,
        "seed": seed,
        "max_questions": max_questions,
        "reasoning": reasoning,
        "name": _arg("--name") or default_name(model, tokens, seed),
        "dry_run": "--dry-run" in sys.argv,
    }


def load_questions() -> list[str]:
    """Download (first use) and read the nq_open train questions."""

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split=SPLIT, cache_dir=str(HF_CACHE))
    return list(dataset["question"])


def select_questions(
    questions: Sequence[str], seed: int, count: int
) -> list[tuple[int, str]]:
    """Deterministic no-repeat sample: same seed -> same questions, same order."""

    rng = random.Random(seed)
    indices = rng.sample(range(len(questions)), k=min(count, len(questions)))
    return [(index, questions[index]) for index in indices]


def _usage_record(usage) -> dict:
    def value(*names: str) -> float:
        for name in names:
            candidate = getattr(usage, name, None)
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return 0.0

    total_cost = getattr(usage, "total_cost", None)
    return {
        "input": value("input_tokens"),
        "output": value("output_tokens"),
        "cache_read": value("input_tokens_cache_read"),
        "cache_write": value("input_tokens_cache_write"),
        "total_cost": total_cost if isinstance(total_cost, (int, float)) else None,
    }


async def run_conversation(cfg: dict, selected: list[tuple[int, str]]) -> dict:
    """Ask questions in one growing conversation until the context target is hit."""

    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    system_prompt = expected_system_prompt(cfg["reasoning"])
    target_slug = resolve_target(cfg["model"])
    build = build_target(
        target_slug, reasoning_on=cfg["reasoning"], effort=REASONING_EFFORT
    )
    model = build.model
    if isinstance(model, str):
        from inspect_ai.model import get_model

        model = get_model(model)

    messages = [ChatMessageSystem(content=system_prompt)]
    asked: list[dict] = []
    totals = {
        "input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0,
    }
    billed_total = 0.0
    all_calls_billed = True
    context_tokens = 0
    measurement = "provider_usage"
    reached = False

    for index, question in selected:
        messages.append(ChatMessageUser(content=question))
        output = await model.generate(input=messages)
        messages.append(output.message)
        usage = getattr(output, "usage", None)
        if usage is not None:
            record = _usage_record(usage)
            for key in totals:
                totals[key] += record[key]
            if record["total_cost"] is None:
                all_calls_billed = False
            else:
                billed_total += record["total_cost"]
            context_tokens = int(record["input"] + record["output"])
        else:
            # No provider usage on this call: fall back to a visible estimate.
            all_calls_billed = False
            measurement = "estimated_chars_over_4"
            context_tokens = sum(len(m.text or "") for m in messages) // 4
        asked.append({"dataset_index": index, "question": question})
        print(
            f"  q{len(asked)}/{len(selected)} [{index}] {question[:70]!r} -> "
            f"context ~{context_tokens:,} tokens"
        )
        if context_tokens >= cfg["tokens"]:
            reached = True
            break

    usage_summary = {
        **{key: int(value) for key, value in totals.items()},
        "total_cost": billed_total if all_calls_billed and asked else None,
    }
    priced = estimate_usage_cost(target_slug, usage_summary)
    return {
        "messages": messages,
        "asked": asked,
        "context_tokens": context_tokens,
        "measurement": measurement,
        "reached_target_tokens": reached,
        "usage": usage_summary,
        "cost": priced,
    }


def build_payload(cfg: dict, result: dict) -> dict:
    payload = {
        "format": PREFIX_FORMAT,
        "name": cfg["name"],
        "target": cfg["model"],
        "reasoning": cfg["reasoning"],
        "source": {
            "kind": "external",
            "description": (
                f"multi-turn open-domain Q&A: {len(result['asked'])} Natural Questions "
                f"({DATASET}, {SPLIT}) answered in one conversation; deterministic "
                f"question order under rng_seed={cfg['seed']}"
            ),
            "generator": "exp_nq_prefix.py",
            "dataset": DATASET,
            "split": SPLIT,
            "rng_seed": cfg["seed"],
            "target_context_tokens": cfg["tokens"],
            "measured_context_tokens": result["context_tokens"],
            "token_measurement": result["measurement"],
            "reached_target_tokens": result["reached_target_tokens"],
            "questions": result["asked"],
            "generated_at": datetime.now().astimezone().isoformat(),
            "generation_usage": result["usage"],
            "generation_cost": result["cost"],
        },
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in result["messages"]
        ],
    }
    return validate_prefix_payload(payload, origin="exp_nq_prefix")


def main() -> None:
    cfg = _parse_args()
    load_dotenv(ENV_FILE)
    print("=" * 72)
    print("NQ PREFIX BUILDER  (multi-turn Q&A -> continuation prefix payload)")
    print("=" * 72)
    print(f"  model={cfg['model']} [reasoning:{'on' if cfg['reasoning'] else 'off'}]  "
          f"target context={cfg['tokens']:,} tokens")
    print(f"  question seed={cfg['seed']}  max questions={cfg['max_questions']}  "
          f"name={cfg['name']}")
    print(f"[dataset] loading {DATASET} ({SPLIT}) into {HF_CACHE} ...")
    questions = load_questions()
    selected = select_questions(questions, cfg["seed"], cfg["max_questions"])
    print(f"[dataset] {len(questions):,} questions; selected {len(selected)} "
          "(deterministic order, no repeats)")

    if cfg["dry_run"]:
        for position, (index, question) in enumerate(selected[:15], start=1):
            print(f"  {position:>3}. [{index}] {question}")
        if len(selected) > 15:
            print(f"  ... and {len(selected) - 15} more in fixed order")
        print("\n[dry-run] no model calls, no files written.")
        return

    install_cost_tracking()
    print(f"[generate] asking questions until ~{cfg['tokens']:,} tokens of context ...")
    result = asyncio.run(run_conversation(cfg, selected))

    if not result["asked"]:
        raise SystemExit("no questions were asked; nothing to save")
    if not result["reached_target_tokens"]:
        print("\n" + "!" * 72)
        print(f"WARNING: hit --max-questions={cfg['max_questions']} at only "
              f"~{result['context_tokens']:,} of the requested {cfg['tokens']:,} "
              "tokens. The payload is written anyway, flagged "
              "reached_target_tokens=false; rerun with a higher --max-questions "
              "for a full-size prefix.")
        print("!" * 72)

    payload = build_payload(cfg, result)
    spec = build_prefix_spec(payload)  # full continuation invariant check, free
    path = store_prefix_payload(payload)
    cost = result["cost"]
    cost_note = (
        f"${cost['cost_usd']:.4f} ({'EXACT' if cost['exact'] else 'estimate'}, "
        f"{cost['source']})"
        if cost["cost_usd"] is not None else "unpriced model"
    )
    print(f"\n[done] {len(result['asked'])} questions, "
          f"~{result['context_tokens']:,} context tokens "
          f"({result['measurement']}), {len(spec.messages)} messages")
    print(f"       generation cost: {cost_note}")
    print(f"       payload: {path}")
    print("\nUse it with:")
    print(f"  uv run exp_continuation_pipeline.py --treatment=<slug> "
          f"--prefix-files={path} \\")
    print("      --seed-dir=<family> --seeds=<members> --epochs=<n>")


if __name__ == "__main__":
    main()
