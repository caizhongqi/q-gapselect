from qgapselect.qcollide import packing_statistics, random_range


def test_random_range_matching_never_exceeds_edges_or_domain() -> None:
    instance = random_range(n=128, range_size=256, seed=20260810)
    stats = packing_statistics(instance)
    assert 0 <= stats.matching_size <= 128
    assert stats.matching_size <= stats.edge_count
    assert 0.0 <= stats.independence_ratio <= 1.0
