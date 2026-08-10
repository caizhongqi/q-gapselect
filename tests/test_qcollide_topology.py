import numpy as np

from qgapselect.qcollide.topology import collision_filtration, collision_topology_metrics


def test_disjoint_claws_have_maximal_component_entropy_and_capacity() -> None:
    metrics = collision_topology_metrics(
        ((0,), (1,), (2,)),
        3,
        edge_vectors=np.eye(3),
    )
    assert metrics.beta0_active == 3
    assert metrics.beta1_active == 0
    assert np.isclose(metrics.normalized_component_edge_entropy, 1.0)
    assert np.isclose(metrics.effective_component_count, 3.0)
    assert metrics.matching_size == 3
    assert metrics.capacity_fraction == 1.0
    assert metrics.displacement_rank_95 == 3
    assert np.isclose(metrics.displacement_entropy_rank, 3.0)


def test_star_and_cycle_are_topologically_distinct() -> None:
    star = collision_topology_metrics(((0,), (0,), (0,)), 3)
    cycle = collision_topology_metrics(((0, 1), (0, 1)), 2)
    assert star.beta0_active == 1
    assert star.beta1_active == 0
    assert star.matching_size == 1
    assert np.isclose(star.capacity_fraction, 1.0 / 3.0)
    assert star.normalized_component_edge_entropy == 0.0
    assert cycle.beta0_active == 1
    assert cycle.beta1_active == 1
    assert cycle.matching_size == 2


def test_control_threshold_filtration_is_edge_and_capacity_monotone() -> None:
    control = np.array([[0.1, 0.8], [0.7, 0.2]])
    payload = np.ones((2, 2))
    behavior = np.ones((2, 2))
    displacements = np.ones((2, 2, 3))
    points = collision_filtration(
        control,
        payload,
        behavior,
        (0.15, 0.25, 0.75, 1.0),
        payload_delta=0.5,
        behavior_gamma=0.5,
        edge_displacements=displacements,
    )
    edges = [point.metrics.edge_count for point in points]
    capacities = [point.metrics.matching_size for point in points]
    assert edges == sorted(edges)
    assert capacities == sorted(capacities)
    assert edges == [1, 2, 3, 4]
    assert capacities[-1] == 2


def test_duplicate_and_misaligned_edges_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        collision_topology_metrics(((0, 0),), 1)
    with np.testing.assert_raises(ValueError):
        collision_topology_metrics(((0,),), 1, edge_vectors=np.ones((2, 3)))
