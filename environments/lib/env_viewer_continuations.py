"""Grouping and base-rate math for the viewer's global Continuations page.

Pure, free helpers over already-loaded audit dicts. Continuation trajectories are
identified by their stored ``real_env["continuation"]`` record (written by the
solver), never by run-directory naming. Rendering stays in ``viewer.py``; this
module only computes.
"""

from __future__ import annotations

from env_viewer_visuals import target_label, trajectory_category


CATEGORY_KEYS = (
    "hack", "review", "notable", "clean",
    "unjudged", "invalid", "awaiting", "excluded",
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


def category_counts(audits: list[dict]) -> dict[str, int]:
    counts = {key: 0 for key in CATEGORY_KEYS}
    for audit in audits:
        counts[trajectory_category(audit)] += 1
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


def baseline_audits(
    originals: list[dict], target: str, reasoning: bool | None
) -> list[dict]:
    """Original trajectories comparable to one continuation group.

    ``originals`` is the seed's current official pool. Matching is by exact target
    slug; when the group has one recorded reasoning setting, originals with a
    recorded conflicting setting are dropped (unstamped ones stay).
    """

    pool = [
        audit for audit in originals
        if str(audit.get("target") or "") == target
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
) -> list[tuple[tuple[str, str], dict[str, list[dict]]]]:
    """[( (seed, target), {treatment: rows} )] ordered by seed nav order, target."""

    grouped: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for audit in continuations:
        key = (str(audit.get("seed") or "unknown"), str(audit.get("target") or ""))
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
            target_label(item[0][1]),
        ),
    )
