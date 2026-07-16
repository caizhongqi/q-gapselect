from __future__ import annotations

import json

import pytest

from qgapselect.attack_oracles import FrozenCandidateGraph, freeze_source_streams
from qgapselect.frozen_coherent_oracle import FrozenEmpiricalCoherentOracle
from qgapselect.independent_iae_topk import (
    CLAIM_SCOPE,
    IndependentIAETopKReference,
    run_independent_iae_topk,
)
from qgapselect.models import IAEConfig
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


def _config() -> IAEConfig:
    return IAEConfig(
        target_angular_precision=0.1,
        confidence=0.05,
        shots_per_round=32,
        max_rounds=3,
        max_grover_power=3,
        grid_points=1025,
    )


def _endpoint_oracle(seed: int = 7) -> FrozenEmpiricalCoherentOracle:
    graph = FrozenCandidateGraph.from_ids(("zero-a", "one-a", "one-b", "zero-b"))
    fixture = freeze_source_streams(
        graph,
        reward_streams={
            "zero-a": [0, 0, 0, 0],
            "one-a": [1, 1, 1, 1],
            "one-b": [1, 1, 1, 1],
            "zero-b": [0, 0, 0, 0],
        },
        cost_streams={candidate: [1.0] * 4 for candidate in graph.candidate_ids},
    )
    return FrozenEmpiricalCoherentOracle(fixture, measurement_seed=seed)


def test_independent_iae_certifies_strictly_separated_endpoint_top_k() -> None:
    oracle = _endpoint_oracle()
    result = run_independent_iae_topk(
        oracle,
        2,
        config=_config(),
        confidence=0.05,
        target_angular_precision=0.1,
    )

    assert result.selected == (1, 2)
    assert result.certified
    assert result.status == "certified_strict_interval_separation"
    assert result.certificate_boundary is not None
    selected_lower, outside_upper = result.certificate_boundary
    assert selected_lower > outside_upper
    assert result.claim_scope == CLAIM_SCOPE
    assert result.claim_scope == ("analytic_independent_iae_topk_reference_no_hardware_claim")
    assert not result.hardware_claimable


def test_equal_best_arms_remain_explicitly_unresolved() -> None:
    graph = FrozenCandidateGraph.from_ids(("one-a", "one-b", "zero"))
    fixture = freeze_source_streams(
        graph,
        reward_streams={
            "one-a": [1] * 4,
            "one-b": [1] * 4,
            "zero": [0] * 4,
        },
        cost_streams={candidate: [1.0] * 4 for candidate in graph.candidate_ids},
    )
    oracle = FrozenEmpiricalCoherentOracle(fixture, measurement_seed=11)
    result = run_independent_iae_topk(
        oracle,
        1,
        config=_config(),
        confidence=0.05,
        target_angular_precision=0.1,
    )

    assert result.selected == (0,)
    assert not result.certified
    assert result.heuristic_output_only
    assert result.status == "unresolved_heuristic_ranking_only"
    assert result.unresolved_reason is not None
    assert result.intervals[0] == result.intervals[1]


def test_certification_is_exactly_the_strict_interval_predicate() -> None:
    result = run_independent_iae_topk(
        _endpoint_oracle(seed=19),
        2,
        config=_config(),
        confidence=0.1,
        target_angular_precision=0.1,
    )
    selected = set(result.selected)
    outside = set(result.ranking) - selected
    predicate = all(
        result.intervals[arm][0] > max(result.intervals[other][1] for other in outside)
        for arm in selected
    )

    assert result.certified is predicate


def test_query_and_per_arm_call_ledgers_are_exact() -> None:
    oracle = _endpoint_oracle(seed=3)
    result = run_independent_iae_topk(
        oracle,
        2,
        config=_config(),
        confidence=0.05,
        target_angular_precision=0.1,
    )

    assert result.query_counts == oracle.query_snapshot().flat()
    assert sum(result.per_arm_calls.values()) == result.oracle_queries
    assert result.per_arm_estimator_calls == {0: 1, 1: 1, 2: 1, 3: 1}
    assert set(result.per_arm_query_counts) == {0, 1, 2, 3}
    assert all(record.oracle_calls > 0 for record in result.trace)
    assert all(record.grover_experiments >= 1 for record in result.trace)


class _ThreeMethodOnlyOracle:
    """Capability spy with no public mean, fixture, or evaluator access."""

    def __init__(self) -> None:
        self._delegate = CanonicalBernoulliOracleSimulator((0.0, 1.0), seed=5)

    @property
    def n_arms(self) -> int:
        return self._delegate.n_arms

    @property
    def means(self) -> object:
        raise AssertionError("algorithm attempted to read hidden means")

    def query_snapshot(self):
        return self._delegate.query_snapshot()

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool = False,
        tag: str | None = None,
    ) -> int:
        return self._delegate.run_grover_experiment(
            arm,
            grover_power,
            shots,
            controlled=controlled,
            tag=tag,
        )


def test_algorithm_uses_only_the_three_method_coherent_capability() -> None:
    result = run_independent_iae_topk(
        _ThreeMethodOnlyOracle(),
        1,
        config=_config(),
        confidence=0.05,
        target_angular_precision=0.1,
    )

    assert result.selected == (1,)
    assert result.certified


def test_all_arms_selection_is_vacuously_certified_but_still_estimated() -> None:
    result = run_independent_iae_topk(
        _endpoint_oracle(),
        4,
        config=_config(),
        confidence=0.05,
        target_angular_precision=0.1,
    )

    assert result.certified
    assert result.status == "certified_all_arms_selected"
    assert result.certificate_boundary is None
    assert set(result.selected) == {0, 1, 2, 3}
    assert len(result.trace) == 4
    assert result.oracle_queries > 0


def test_result_is_json_serializable_without_exposing_an_oracle() -> None:
    result = run_independent_iae_topk(
        _endpoint_oracle(),
        2,
        config=_config(),
        confidence=0.05,
        target_angular_precision=0.1,
    )

    document = result.as_dict()
    assert json.loads(json.dumps(document, sort_keys=True)) == document
    assert "oracle" not in document


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"k": 0}, ValueError, "k"),
        ({"k": 5}, ValueError, "k cannot exceed"),
        ({"confidence": 0.0}, ValueError, "confidence"),
        ({"confidence": 1.0}, ValueError, "confidence"),
        ({"target_angular_precision": 0.0}, ValueError, "target_angular_precision"),
        ({"config": object()}, TypeError, "IAEConfig"),
        ({"oracle": object()}, TypeError, "oracle"),
    ],
)
def test_invalid_inputs_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "oracle": _endpoint_oracle(),
        "k": 2,
        "config": _config(),
        "confidence": 0.05,
        "target_angular_precision": 0.1,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        IndependentIAETopKReference(**arguments)  # type: ignore[arg-type]
