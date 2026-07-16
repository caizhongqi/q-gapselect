"""Failure attribution for fixed-cap matched Q-GapSelect campaigns."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    exact_rate = _rate(rows, "exact")
    certified_exact_rate = _rate(rows, "certified_exact")
    return {
        "attempts": len(rows),
        "exact_rate": exact_rate,
        "certified_exact_rate": certified_exact_rate,
        "certification_loss_rate": exact_rate - certified_exact_rate,
        "complete_rate": _rate(rows, "complete"),
        "timeout_rate": _rate(rows, "timeout"),
        "cleanup_failure_rate": sum(row["cleanup_passed"] is False for row in rows)
        / len(rows),
        "mean_actual_queries": _mean(rows, "actual_queries"),
        "mean_selection_queries": _mean(rows, "selection_queries"),
        "mean_verification_queries": _mean(rows, "verification_queries"),
    }


def analyze_matched_failure(
    records: Sequence[Mapping[str, object]],
    *,
    candidate_method_id: str,
    baseline_method_ids: Sequence[str],
) -> dict[str, object]:
    """Attribute candidate loss without dropping attempts or choosing subgroups."""

    if not records:
        raise ValueError("records must be non-empty")
    baseline_ids = tuple(dict.fromkeys(str(value) for value in baseline_method_ids))
    if not baseline_ids or candidate_method_id in baseline_ids:
        raise ValueError("baseline ids must be non-empty and exclude the candidate")

    grouped: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    methods: set[str] = set()
    for record in records:
        family = str(record["family_id"])
        cap = int(record["query_cap"])
        method = str(record["method_id"])
        methods.add(method)
        grouped[(family, cap, method)].append(record)
    required = {candidate_method_id, *baseline_ids}
    missing = sorted(required - methods)
    if missing:
        raise ValueError(f"records are missing methods: {missing}")

    cells: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    families = sorted({key[0] for key in grouped})
    caps = sorted({key[1] for key in grouped})
    for family in families:
        for cap in caps:
            summaries: dict[str, dict[str, object]] = {}
            for method in sorted(required):
                rows = grouped.get((family, cap, method), [])
                if not rows:
                    raise ValueError(f"incomplete cell: {(family, cap, method)}")
                summary = _group_summary(rows)
                summaries[method] = summary
                cells.append(
                    {
                        "family_id": family,
                        "query_cap": cap,
                        "method_id": method,
                        **summary,
                    }
                )

            candidate = summaries[candidate_method_id]
            strongest = max(
                baseline_ids,
                key=lambda method: (
                    float(summaries[method]["certified_exact_rate"]),
                    float(summaries[method]["exact_rate"]),
                    method,
                ),
            )
            baseline = summaries[strongest]
            observed = float(candidate["certified_exact_rate"])
            idealized = float(candidate["exact_rate"])
            baseline_rate = float(baseline["certified_exact_rate"])
            comparisons.append(
                {
                    "family_id": family,
                    "query_cap": cap,
                    "candidate_method_id": candidate_method_id,
                    "strongest_baseline_method_id": strongest,
                    "candidate_certified_exact_rate": observed,
                    "candidate_exact_upper_envelope": idealized,
                    "candidate_certification_recovery_potential": idealized - observed,
                    "strongest_baseline_certified_exact_rate": baseline_rate,
                    "observed_risk_difference": observed - baseline_rate,
                    "idealized_candidate_risk_difference": idealized - baseline_rate,
                    "candidate_observed_dominated": observed < baseline_rate,
                    "candidate_still_dominated_after_perfect_certification": (
                        idealized < baseline_rate
                    ),
                }
            )

    maximum_cap = max(caps)
    max_cap_rows = [row for row in comparisons if row["query_cap"] == maximum_cap]
    return {
        "candidate_method_id": candidate_method_id,
        "baseline_method_ids": list(baseline_ids),
        "family_count": len(families),
        "query_cap_count": len(caps),
        "attempt_count": len(records),
        "all_attempts_retained": True,
        "post_hoc_subgroup_selection_performed": False,
        "cells": cells,
        "comparisons": comparisons,
        "maximum_cap_summary": {
            "query_cap": maximum_cap,
            "family_count": len(max_cap_rows),
            "observed_dominated_family_count": sum(
                bool(row["candidate_observed_dominated"]) for row in max_cap_rows
            ),
            "idealized_dominated_family_count": sum(
                bool(row["candidate_still_dominated_after_perfect_certification"])
                for row in max_cap_rows
            ),
            "mean_observed_risk_difference": fmean(
                float(row["observed_risk_difference"]) for row in max_cap_rows
            ),
            "mean_idealized_risk_difference": fmean(
                float(row["idealized_candidate_risk_difference"]) for row in max_cap_rows
            ),
            "mean_certification_recovery_potential": fmean(
                float(row["candidate_certification_recovery_potential"])
                for row in max_cap_rows
            ),
        },
        "claim_boundary": (
            "diagnostic attribution only; exact recovery is an optimistic envelope, "
            "not an executed algorithm or quantum-advantage result"
        ),
    }
