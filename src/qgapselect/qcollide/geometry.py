"""Local collision-tunnel geometry and causal control-rank interventions."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt

import numpy as np


@dataclass(frozen=True)
class TunnelGeometry:
    manifold_dimension: int
    effective_rank: int
    tunnel_dimension: int
    openness: float
    direction: tuple[float, ...]
    projected_gradient_norm: float


@dataclass(frozen=True)
class TunnelCertificate:
    geometry: TunnelGeometry
    gamma_rate: float
    beta_rate: float
    required_step: float
    safe_step: float
    margin: float
    certified: bool


def _as_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return array


def _as_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return array


def numerical_rank(matrix: np.ndarray, relative_tolerance: float = 1e-10) -> int:
    matrix = _as_matrix(matrix, name="matrix")
    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return 0
    threshold = relative_tolerance * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold))


def nullspace_basis(matrix: np.ndarray, relative_tolerance: float = 1e-10) -> np.ndarray:
    matrix = _as_matrix(matrix, name="matrix")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if len(singular_values) == 0 or singular_values[0] == 0.0:
        rank = 0
    else:
        rank = int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))
    return vh[rank:].T.copy()


def tunnel_geometry(
    control_jacobian: np.ndarray,
    behavior_gradient: np.ndarray,
    relative_tolerance: float = 1e-10,
) -> TunnelGeometry:
    jacobian = _as_matrix(control_jacobian, name="control_jacobian")
    gradient = _as_vector(behavior_gradient, name="behavior_gradient")
    if jacobian.shape[1] != gradient.shape[0]:
        raise ValueError("Jacobian and gradient dimensions must agree")
    manifold_dimension = gradient.size
    rank = numerical_rank(jacobian, relative_tolerance)
    basis = nullspace_basis(jacobian, relative_tolerance)
    projected = basis @ (basis.T @ gradient) if basis.size else np.zeros_like(gradient)
    projected_norm = float(np.linalg.norm(projected))
    gradient_norm = float(np.linalg.norm(gradient))
    openness = projected_norm / gradient_norm if gradient_norm > 0.0 else 0.0
    if projected_norm > 0.0:
        direction = projected / projected_norm
    else:
        direction = np.zeros_like(gradient)
    return TunnelGeometry(
        manifold_dimension=manifold_dimension,
        effective_rank=rank,
        tunnel_dimension=manifold_dimension - rank,
        openness=float(openness),
        direction=tuple(float(x) for x in direction),
        projected_gradient_norm=projected_norm,
    )


def tunnel_certificate(
    control_jacobian: np.ndarray,
    behavior_gradient: np.ndarray,
    payload_jacobian: np.ndarray,
    *,
    control_epsilon: float,
    payload_delta: float,
    behavior_gamma: float,
    control_curvature: float,
    behavior_curvature: float,
    payload_curvature: float,
    local_radius: float,
    relative_tolerance: float = 1e-10,
) -> TunnelCertificate:
    if min(control_epsilon, payload_delta, behavior_gamma) < 0.0:
        raise ValueError("thresholds must be non-negative")
    if min(control_curvature, behavior_curvature, payload_curvature) < 0.0:
        raise ValueError("curvature bounds must be non-negative")
    if local_radius <= 0.0:
        raise ValueError("local_radius must be positive")

    geometry = tunnel_geometry(control_jacobian, behavior_gradient, relative_tolerance)
    direction = np.asarray(geometry.direction, dtype=float)
    payload = _as_matrix(payload_jacobian, name="payload_jacobian")
    if payload.shape[1] != direction.size:
        raise ValueError("payload Jacobian and tangent dimensions must agree")

    gradient = np.asarray(behavior_gradient, dtype=float)
    gamma_rate = float(abs(gradient @ direction))
    beta_rate = float(np.linalg.norm(payload @ direction))

    if gamma_rate == 0.0 or beta_rate == 0.0:
        required = inf
    else:
        required = max(2.0 * behavior_gamma / gamma_rate, 2.0 * payload_delta / beta_rate)

    control_bound = (
        inf
        if control_curvature == 0.0
        else sqrt(2.0 * control_epsilon / control_curvature)
    )
    behavior_bound = inf if behavior_curvature == 0.0 else gamma_rate / behavior_curvature
    payload_bound = inf if payload_curvature == 0.0 else beta_rate / payload_curvature
    safe = min(control_bound, behavior_bound, payload_bound, local_radius)
    margin = safe - required
    return TunnelCertificate(
        geometry=geometry,
        gamma_rate=gamma_rate,
        beta_rate=beta_rate,
        required_step=float(required),
        safe_step=float(safe),
        margin=float(margin),
        certified=bool(margin >= 0.0 and geometry.tunnel_dimension > 0),
    )


def targeted_control_closure(
    control_jacobian: np.ndarray,
    behavior_gradient: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Append the most behavior-relevant control-null direction as a control row."""

    jacobian = _as_matrix(control_jacobian, name="control_jacobian")
    geometry = tunnel_geometry(jacobian, behavior_gradient, tolerance)
    direction = np.asarray(geometry.direction, dtype=float)
    if np.linalg.norm(direction) <= tolerance:
        return jacobian.copy()
    return np.vstack([jacobian, direction])
