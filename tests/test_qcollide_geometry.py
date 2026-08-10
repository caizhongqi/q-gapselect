import numpy as np

from qgapselect.qcollide import (
    geometry_to_packing,
    packing_statistics,
    targeted_control_closure,
    tunnel_certificate,
    tunnel_geometry,
)


def test_geometry_detects_control_null_behavior_direction() -> None:
    jacobian = np.array([[1.0, 0.0, 0.0]])
    gradient = np.array([0.0, 1.0, 0.0])
    geometry = tunnel_geometry(jacobian, gradient)
    assert geometry.effective_rank == 1
    assert geometry.tunnel_dimension == 2
    assert np.isclose(geometry.openness, 1.0)


def test_targeted_closure_removes_the_selected_open_direction() -> None:
    jacobian = np.array([[1.0, 0.0, 0.0]])
    gradient = np.array([0.0, 1.0, 0.0])
    closed = targeted_control_closure(jacobian, gradient)
    before = tunnel_geometry(jacobian, gradient)
    after = tunnel_geometry(closed, gradient)
    assert before.openness == 1.0
    assert after.openness < 1e-10
    assert after.tunnel_dimension == before.tunnel_dimension - 1


def test_positive_certificate_generates_packed_geometry_fixture() -> None:
    instance = geometry_to_packing(
        manifold_dimension=8,
        control_rank=3,
        openness=1.0,
        anchors=6,
        control_epsilon=0.20,
        payload_delta=0.05,
        behavior_gamma=0.05,
        control_curvature=0.05,
        behavior_curvature=0.05,
        payload_curvature=0.05,
        local_radius=1.0,
        seed=99,
    )
    stats = packing_statistics(instance)
    certificates = instance.metadata["certificates"]
    assert all(item["certified"] for item in certificates)
    assert stats.matching_size == 6


def test_full_rank_control_has_no_tunnel_certificate() -> None:
    jacobian = np.eye(4)
    gradient = np.ones(4)
    certificate = tunnel_certificate(
        jacobian,
        gradient,
        np.eye(4),
        control_epsilon=0.1,
        payload_delta=0.01,
        behavior_gamma=0.01,
        control_curvature=0.1,
        behavior_curvature=0.1,
        payload_curvature=0.1,
        local_radius=1.0,
    )
    assert certificate.geometry.tunnel_dimension == 0
    assert not certificate.certified


def test_row_space_gradient_does_not_create_numerical_tunnel() -> None:
    rng = np.random.default_rng(314159)
    control = rng.normal(size=(3, 8))
    gradient = control.T @ rng.normal(size=3)
    geometry = tunnel_geometry(control, gradient)
    assert geometry.tunnel_dimension == 5
    assert geometry.openness == 0.0
    assert np.linalg.norm(geometry.direction) == 0.0
