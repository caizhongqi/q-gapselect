import numpy as np

from qgapselect.qcollide.overlap_bounds import bipartite_capacity_envelope
from qgapselect.qcollide.overlap_capacity import (
    critical_retention_probability,
    estimate_overlap_capacity,
    make_dense_graph,
    make_hub_graph,
    make_matching_graph,
    make_sparse_overlap_graph,
    packed_survival_probability,
)


def test_matching_fixture_has_exact_capacity_and_degree_one() -> None:
    graph = make_matching_graph(n=32, matching_size=7, seed=1)
    assert graph.matching_size == 7
    assert graph.edge_count == 7
    assert graph.maximum_degree == 1


def test_hub_fixture_separates_edges_from_matching_capacity() -> None:
    graph = make_hub_graph(n=32, spokes=20, seed=2)
    assert graph.edge_count == 20
    assert graph.matching_size == 1
    assert graph.maximum_degree == 20


def test_sparse_overlap_respects_requested_degree_bound() -> None:
    graph = make_sparse_overlap_graph(n=40, matching_size=10, maximum_degree=4, seed=3)
    assert graph.matching_size >= 10
    assert graph.maximum_degree <= 4


def test_dense_graph_is_nontrivial() -> None:
    graph = make_dense_graph(n=24, density=0.25, seed=4)
    assert graph.edge_count > graph.matching_size
    assert graph.matching_size > 0


def test_packed_survival_formula_matches_monte_carlo() -> None:
    graph = make_matching_graph(n=64, matching_size=16, seed=5)
    q = critical_retention_probability(graph.matching_size)
    estimate = estimate_overlap_capacity(
        graph,
        target_q=q,
        trials=12000,
        seed=6,
        degree_aware=False,
    )
    expected = packed_survival_probability(graph.matching_size, q)
    assert np.isclose(estimate.survival_probability, expected, atol=0.025)
    assert estimate.absolute_error_packed_inversion < 2.0


def test_hub_is_negative_control_for_packed_inversion() -> None:
    graph = make_hub_graph(n=64, spokes=32, seed=7)
    q = 0.5
    estimate = estimate_overlap_capacity(
        graph,
        target_q=q,
        trials=8000,
        seed=8,
        degree_aware=False,
    )
    assert graph.matching_size == 1
    assert estimate.packed_inversion_estimate > 1.5


def test_degree_aware_sampling_reduces_hub_survival_bias() -> None:
    graph = make_hub_graph(n=64, spokes=32, seed=9)
    uniform = estimate_overlap_capacity(
        graph,
        target_q=0.5,
        trials=8000,
        seed=10,
        degree_aware=False,
    )
    corrected = estimate_overlap_capacity(
        graph,
        target_q=0.5,
        trials=8000,
        seed=11,
        degree_aware=True,
    )
    assert corrected.survival_probability < uniform.survival_probability


def test_capacity_envelope_contains_truth_for_all_fixture_families() -> None:
    graphs = (
        make_matching_graph(n=48, matching_size=12, seed=12),
        make_sparse_overlap_graph(n=48, matching_size=12, maximum_degree=4, seed=13),
        make_hub_graph(n=48, spokes=32, seed=14),
        make_dense_graph(n=48, density=0.15, seed=15),
    )
    for graph in graphs:
        envelope = bipartite_capacity_envelope(graph)
        assert envelope.lower_bound <= graph.matching_size <= envelope.upper_bound


def test_capacity_envelope_is_exact_on_matching_and_star_extremes() -> None:
    matching = bipartite_capacity_envelope(
        make_matching_graph(n=32, matching_size=9, seed=16)
    )
    hub = bipartite_capacity_envelope(make_hub_graph(n=32, spokes=24, seed=17))
    assert matching.lower_bound == matching.true_matching_size == matching.upper_bound
    assert hub.lower_bound == hub.true_matching_size == 1
