from __future__ import annotations

import json

import pytest

from qgapselect.attack_oracles import CandidateEdge
from qgapselect.frozen_selector_benchmarking import (
    CLAIM_SCOPE,
    DEFAULT_SELECTOR_IDS,
    SelectorBudget,
    SelectorLandscape,
    run_frozen_selector_benchmark,
)


def _landscapes() -> tuple[SelectorLandscape, ...]:
    edges = (
        CandidateEdge("a", "b"),
        CandidateEdge("b", "a"),
        CandidateEdge("b", "c"),
        CandidateEdge("c", "b"),
        CandidateEdge("c", "d"),
        CandidateEdge("d", "c"),
    )
    return (
        SelectorLandscape.from_means(
            "wide-gap",
            {"a": 0.05, "b": 0.85, "c": 0.65, "d": 0.15},
            candidate_costs={"a": 0.5, "b": 1.0, "c": 1.5, "d": 0.75},
            edges=edges,
        ),
        SelectorLandscape.from_means(
            "near-boundary",
            {"a": 0.45, "b": 0.55, "c": 0.52, "d": 0.48},
            candidate_costs=1.0,
            edges=edges,
        ),
    )


def _run_report(master_seed: int = 23):
    return run_frozen_selector_benchmark(
        landscapes=_landscapes(),
        budgets=(
            SelectorBudget("q8", max_queries=8, max_cost=12.0),
            SelectorBudget("q16", max_queries=16, max_cost=24.0),
        ),
        trials=3,
        k=2,
        samples_per_candidate=20,
        master_seed=master_seed,
    )


def test_multi_landscape_budget_trial_matrix_is_complete() -> None:
    report = _run_report()

    assert len(report.runs) == 2 * 2 * 3 * len(DEFAULT_SELECTOR_IDS)
    assert len(report.aggregates) == 2 * 2 * len(DEFAULT_SELECTOR_IDS)
    assert len(report.fixture_manifest_hashes) == 2 * 3
    assert len(report.manifest_hash) == 64
    assert report.claim_scope == CLAIM_SCOPE
    assert all(aggregate.run_count == 3 for aggregate in report.aggregates)


def test_all_selectors_receive_the_same_fixture_and_fair_budget_per_panel() -> None:
    report = _run_report()
    panels: dict[tuple[str, str, int], list[object]] = {}
    for run in report.runs:
        panels.setdefault((run.landscape_id, run.budget_id, run.trial), []).append(run)

    for panel in panels.values():
        assert {run.selector_id for run in panel} == set(DEFAULT_SELECTOR_IDS)
        assert len({run.fixture_manifest_hash for run in panel}) == 1
        assert len({run.query_budget for run in panel}) == 1
        assert len({run.cost_budget for run in panel}) == 1
        assert all(run.queries_used <= run.query_budget for run in panel)
        assert all(run.cost_used <= run.cost_budget + 1e-12 for run in panel)
        assert all(sum(run.sample_counts.values()) == run.queries_used for run in panel)


def test_each_budget_run_has_an_independent_oracle_cursor() -> None:
    landscape = SelectorLandscape.from_means(
        "one",
        {"a": 0.2, "b": 0.8},
        candidate_costs=1.0,
        edges=(CandidateEdge("a", "b"), CandidateEdge("b", "a")),
    )
    report = run_frozen_selector_benchmark(
        landscapes=(landscape,),
        budgets=(SelectorBudget("q4", 4, 4.0),),
        trials=1,
        k=1,
        samples_per_candidate=4,
        master_seed=5,
        selector_ids=("uniform", "successive_halving"),
    )

    assert [run.queries_used for run in report.runs] == [4, 4]
    assert [sum(run.sample_counts.values()) for run in report.runs] == [4, 4]
    assert report.runs[0].fixture_manifest_hash == report.runs[1].fixture_manifest_hash


def test_report_is_bitwise_reproducible_and_json_serializable() -> None:
    first = _run_report()
    second = _run_report()

    assert first == second
    assert first.as_dict() == second.as_dict()
    assert json.loads(json.dumps(first.as_dict(), sort_keys=True)) == first.as_dict()


def test_manifest_changes_when_master_seed_changes() -> None:
    first = _run_report(master_seed=1)
    second = _run_report(master_seed=2)

    assert first.manifest_hash != second.manifest_hash
    assert first.fixture_manifest_hashes != second.fixture_manifest_hashes


def test_evaluation_metrics_are_post_selection_and_well_formed() -> None:
    report = _run_report()

    for run in report.runs:
        assert len(run.selected_candidates) == 2
        assert len(run.reference_top_k) == 2
        assert 0.0 <= run.precision_at_k <= 1.0
        assert 0.0 <= run.recall_at_k <= 1.0
        assert run.top_k_regret >= 0.0
        assert run.optimal_expected_reward >= run.selected_expected_reward
        assert run.claim_scope == CLAIM_SCOPE
    for aggregate in report.aggregates:
        assert 0.0 <= aggregate.exact_match_rate <= 1.0
        assert aggregate.exact_match_successes <= aggregate.run_count
        assert aggregate.exact_match_wilson_lower <= aggregate.exact_match_rate
        assert aggregate.exact_match_wilson_upper >= aggregate.exact_match_rate
        assert aggregate.std_top_k_regret >= 0.0


def test_random_selector_consumes_no_oracle_resources() -> None:
    report = run_frozen_selector_benchmark(
        landscapes=(_landscapes()[0],),
        budgets=(SelectorBudget("q8", 8, 12.0),),
        trials=2,
        k=2,
        samples_per_candidate=8,
        selector_ids=("random",),
    )

    assert all(run.queries_used == 0 and run.cost_used == 0.0 for run in report.runs)
    assert all(set(run.sample_counts.values()) == {0} for run in report.runs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"landscapes": ()}, "landscapes"),
        ({"budgets": ()}, "budgets"),
        ({"trials": 0}, "trials"),
        ({"k": 0}, "k"),
        ({"k": 5}, "k cannot exceed"),
        ({"samples_per_candidate": 15}, "largest query budget"),
        ({"selector_ids": ()}, "selector_ids"),
        ({"selector_ids": ("unknown",)}, "unknown selector"),
    ],
)
def test_runner_rejects_invalid_designs(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "landscapes": _landscapes(),
        "budgets": (
            SelectorBudget("q8", 8, 12.0),
            SelectorBudget("q16", 16, 24.0),
        ),
        "trials": 1,
        "k": 2,
        "samples_per_candidate": 16,
        "selector_ids": DEFAULT_SELECTOR_IDS,
    }
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        run_frozen_selector_benchmark(**arguments)  # type: ignore[arg-type]


def test_landscape_freezes_caller_owned_mappings() -> None:
    means = {"a": 0.2, "b": 0.8}
    costs = {"a": 1.0, "b": 2.0}
    landscape = SelectorLandscape.from_means("frozen", means, candidate_costs=costs)
    means["a"] = 1.0
    costs["a"] = 99.0

    assert landscape.means == {"a": 0.2, "b": 0.8}
    assert landscape.candidate_costs == {"a": 1.0, "b": 2.0}
