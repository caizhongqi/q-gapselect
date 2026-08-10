from math import isinf

from qgapselect.qcollide import (
    evaluate_instance,
    packed_claw,
    packing_statistics,
    pair_oracle_negative,
)


def test_f0_pair_oracle_refuses_structured_claw_claim() -> None:
    instance = pair_oracle_negative(n=16, marked_pairs=7, seed=11)
    result = evaluate_instance(instance)
    assert result["packing"]["edge_count"] == 7
    assert result["structured_claw_admissible"] is False
    assert isinf(result["structured_claw_cost"])


def test_f1_packed_claw_has_exact_matching() -> None:
    instance = packed_claw(n=64, matching_size=13, seed=23)
    stats = packing_statistics(instance)
    assert stats.edge_count == 13
    assert stats.matching_size == 13
    assert stats.independence_ratio == 1.0
    assert stats.max_left_degree == 1
    assert stats.max_right_degree == 1
