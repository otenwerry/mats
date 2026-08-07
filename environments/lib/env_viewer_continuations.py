"""Grouping and base-rate math for the viewer's global Continuations page.

Pure, free helpers over already-loaded audit dicts. Continuation trajectories are
identified by their stored ``real_env["continuation"]`` record (written by the
solver), never by run-directory naming. Rendering stays in ``viewer.py``; this
module only computes.
"""

from __future__ import annotations

from env_viewer_visuals import agent_label, trajectory_category


CATEGORY_KEYS = (
    "hack", "review", "notable", "clean",
    "unjudged", "invalid", "awaiting",
)


def continuation_of(audit: dict) -> dict | None:
    """The stored continuation record, or None for an ordinary trajectory."""

    record = (audit.get("real_env") or {}).get("continuation")
    return record if isinstance(record, dict) else None


def treatment_of(audit: dict) -> str:
    record = continuation_of(audit) or {}
    return str(record.get("treatment") or "unknown")


def prefix_of(audit: dict) -> dict:
    record = continuation_of(audit) or {}
    prefix = record.get("prefix")
    return prefix if isinstance(prefix, dict) else {}


def prefix_source_trajectory_id(audit: dict) -> int | None:
    value = prefix_of(audit).get("source_trajectory_id")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def category_counts(
    audits: list[dict], *, include_excluded: bool = False
) -> dict[str, int]:
    """Outcome counts for valid runs, or for all runs when explicitly requested."""

    counts = {key: 0 for key in CATEGORY_KEYS}
    for audit in audits:
        if not include_excluded and audit.get("integrity_status") == "excluded":
            continue
        counts[trajectory_category(audit, respect_exclusion=False)] += 1
    return counts


def hack_rate(counts: dict[str, int]) -> float | None:
    """Hacks over trajectories with a usable non-review judgment.

    ``review`` rows (p-hacking `other`) are neither hack nor non-hack, so they sit
    in neither the numerator nor the denominator; their count stays visible in the
    table.
    """

    judged = counts["hack"] + counts["notable"] + counts["clean"]
    return counts["hack"] / judged if judged else None


def reasoning_values(audits: list[dict]) -> set:
    return {
        audit.get("reasoning")
        for audit in audits
        if isinstance(audit.get("reasoning"), bool)
    }


def harness_of(audit: dict) -> str:
    """Stored agent harness, treating pre-flag trajectories as simple."""

    return str(audit.get("harness") or "simple")


def baseline_audits(
    originals: list[dict], agent: str, reasoning: bool | None, harness: str
) -> list[dict]:
    """Original trajectories comparable to one continuation group.

    ``originals`` is the seed's current official pool. Matching is by exact agent
    slug and harness; when the group has one recorded reasoning setting, originals
    with a recorded conflicting setting are dropped (unstamped ones stay). Historical
    originals without a harness stamp are the former simple implementation.
    """

    pool = [
        audit for audit in originals
        if str(audit.get("target") or "") == agent
        and harness_of(audit) == harness
    ]
    if reasoning is None:
        return pool
    return [
        audit for audit in pool
        if not isinstance(audit.get("reasoning"), bool)
        or audit.get("reasoning") == reasoning
    ]


def continuation_groups(
    continuations: list[dict], *, seed_order: list[str]
) -> list[tuple[tuple[str, str, str], dict[str, list[dict]]]]:
    """[((seed, agent, harness), {treatment: rows})] in viewer order."""

    grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for audit in continuations:
        key = (
            str(audit.get("seed") or "unknown"),
            str(audit.get("target") or ""),
            harness_of(audit),
        )
        grouped.setdefault(key, {}).setdefault(treatment_of(audit), []).append(audit)
    order = {seed: index for index, seed in enumerate(seed_order)}
    return sorted(
        (
            (key, dict(sorted(by_treatment.items())))
            for key, by_treatment in grouped.items()
        ),
        key=lambda item: (
            order.get(item[0][0], len(order)),
            item[0][0],
            agent_label(item[0][1]),
            item[0][2],
        ),
    )
