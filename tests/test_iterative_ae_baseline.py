from __future__ import annotations

import inspect

import pytest

import qgapselect.iterative_ae_baseline as baseline_module
from qgapselect.iterative_ae_baseline import IterativeAEThresholdScan
from qgapselect.models import IAEConfig
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


def _config() -> IAEConfig:
    return IAEConfig(
        target_angular_precision=0.1,
        confidence=0.05,
        shots_per_round=64,
        max_rounds=4,
        max_grover_power=7,
        grid_points=1025,
    )


@pytest.mark.parametrize(
    ("relation", "expected"),
    [("above", 0), ("below", 1)],
)
def test_iterative_ae_scan_classifies_endpoint_arms_with_charged_queries(
    relation: str,
    expected: int,
) -> None:
    oracle = CanonicalBernoulliOracleSimulator([1.0, 0.0], seed=3)
    result = IterativeAEThresholdScan(
        oracle,
        0.5,
        1,
        relation=relation,
        confidence=0.05,
        target_angular_precision=0.1,
        config=_config(),
        seed=4,
    ).run()

    assert result.complete and result.verified
    assert result.outputs == (expected,)
    assert result.status == "complete_analytic_iae_angular_intervals"
    assert result.per_arm_confidence == pytest.approx(0.025)
    assert result.resources.oracle_queries > 0
    assert result.resources.grover_experiments >= 1
    assert result.resources.measurement_shots >= 64
    assert result.resources.query_counts == oracle.query_snapshot().flat()
    assert result.resources.backend == "analytic_iterative_ae_measurement_law"
    assert "no_hardware" in result.claim_status


def test_iterative_ae_unresolved_scan_makes_no_absence_claim() -> None:
    oracle = CanonicalBernoulliOracleSimulator([0.5], seed=7)
    result = IterativeAEThresholdScan(
        oracle,
        0.5,
        1,
        target_angular_precision=0.1,
        config=_config(),
        seed=8,
    ).run()

    assert not result.complete
    assert result.outputs == ()
    assert result.status == "scan_exhausted_without_target"
    assert result.trace[0].status == "unresolved"
    assert not result.absence_certified


def test_zero_target_completes_without_query() -> None:
    oracle = CanonicalBernoulliOracleSimulator([0.5], seed=1)
    result = IterativeAEThresholdScan(oracle, 0.5, 0, config=_config()).run()

    assert result.complete and result.outputs == ()
    assert result.resources.oracle_queries == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_count": True},
        {"expected_count": 2},
        {"expected_count": 1, "relation": "equal"},
        {"expected_count": 1, "target_angular_precision": True},
    ],
)
def test_iterative_ae_scan_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    oracle = CanonicalBernoulliOracleSimulator([0.5])
    with pytest.raises((TypeError, ValueError)):
        IterativeAEThresholdScan(oracle, 0.5, config=_config(), **kwargs)


def test_baseline_does_not_reach_hidden_oracle_parameters() -> None:
    source = inspect.getsource(baseline_module)

    assert "__means" not in source
    assert "._Canonical" not in source
    assert ".estimate(" in source
    assert ".query_snapshot(" in source
