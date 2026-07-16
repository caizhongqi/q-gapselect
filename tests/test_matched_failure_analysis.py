from __future__ import annotations

import pytest

from qgapselect.matched_failure_analysis import analyze_matched_failure


def _row(family: str, cap: int, method: str, *, exact: bool, certified: bool) -> dict:
    return {
        "family_id": family,
        "query_cap": cap,
        "method_id": method,
        "exact": exact,
        "certified_exact": exact and certified,
        "complete": exact,
        "timeout": False,
        "cleanup_passed": certified,
        "actual_queries": 10,
        "selection_queries": 8,
        "verification_queries": 2,
    }


def test_failure_analysis_separates_certification_and_selection_gaps() -> None:
    rows = [
        _row("f", 100, "candidate", exact=True, certified=False),
        _row("f", 100, "candidate", exact=True, certified=True),
        _row("f", 100, "baseline", exact=True, certified=True),
        _row("f", 100, "baseline", exact=True, certified=True),
    ]
    result = analyze_matched_failure(
        rows,
        candidate_method_id="candidate",
        baseline_method_ids=("baseline",),
    )
    comparison = result["comparisons"][0]
    assert comparison["candidate_certified_exact_rate"] == 0.5
    assert comparison["candidate_exact_upper_envelope"] == 1.0
    assert comparison["candidate_certification_recovery_potential"] == 0.5
    assert comparison["candidate_observed_dominated"] is True
    assert comparison["candidate_still_dominated_after_perfect_certification"] is False
    assert result["all_attempts_retained"] is True


def test_failure_analysis_rejects_missing_methods() -> None:
    with pytest.raises(ValueError, match="missing methods"):
        analyze_matched_failure(
            [_row("f", 1, "candidate", exact=True, certified=True)],
            candidate_method_id="candidate",
            baseline_method_ids=("baseline",),
        )
