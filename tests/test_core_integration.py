from __future__ import annotations

from qgapselect.gapselect import QGapSelect
from qgapselect.models import GapSelectConfig, TopKInstance
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


def test_executable_driver_keeps_queries_and_candidate_accounting_separate() -> None:
    instance = TopKInstance.from_sequence((0.9, 0.8, 0.2, 0.1), 2)
    oracle = CanonicalBernoulliOracleSimulator(instance.means, seed=17)
    config = GapSelectConfig(
        confidence=0.1,
        initial_angular_epsilon=0.25,
        max_rounds=3,
        shots_per_iae_round=24,
        iae_max_rounds=3,
        iae_grid_points=1025,
    )
    result = QGapSelect(config).run(oracle, instance.k)

    assert len(result.selected) == instance.k
    assert result.executed_query_counts["total"] > 0
    assert result.candidate_theory_accounting.proof_status == (
        "conjectural_not_a_query_bound"
    )
    assert all(
        charge.proof_status == "conjectural_not_a_query_bound"
        for charge in result.candidate_theory_accounting.charges
    )
    assert any("not executed queries" in warning for warning in result.warnings)
