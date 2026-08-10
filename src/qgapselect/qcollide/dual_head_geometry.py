"""Collision construction and rank-matched interventions for the dual-head model."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np

from .contracts import CollisionCriteria, CollisionInstance, EndpointRecord
from .costs import classical_packed_cost, product_johnson_cost
from .dual_head_model import DualHeadModel
from .geometry import nullspace_basis, tunnel_geometry
from .graph import packing_statistics


def _standardize_projection(projection: np.ndarray, hidden_support: np.ndarray) -> np.ndarray:
    output_scales = np.maximum((hidden_support @ projection.T).std(axis=0), 1e-8)
    return projection / output_scales[:, None]


def _control_vector(
    model: DualHeadModel,
    latent: np.ndarray,
    standardized_projection: np.ndarray,
) -> np.ndarray:
    return standardized_projection @ model.hidden(latent) / sqrt(
        standardized_projection.shape[0]
    )


def _calibrate_epsilon(
    model: DualHeadModel,
    anchors: np.ndarray,
    projection: np.ndarray,
    hidden_support: np.ndarray,
    *,
    seed: int,
    benign_radius: float,
    benign_repetitions: int,
    benign_quantile: float,
) -> tuple[np.ndarray, float, float]:
    standardized = _standardize_projection(projection, hidden_support)
    rng = np.random.default_rng(seed)
    distances: list[float] = []
    for anchor in anchors:
        anchor_control = _control_vector(model, anchor, standardized)
        for _ in range(benign_repetitions):
            direction = rng.normal(size=model.latent_dimension)
            direction /= np.linalg.norm(direction)
            perturbed = np.clip(anchor + benign_radius * direction, -1.0, 1.0)
            distances.append(
                float(
                    np.linalg.norm(
                        _control_vector(model, perturbed, standardized) - anchor_control
                    )
                )
            )
    epsilon = float(np.quantile(distances, benign_quantile))
    acceptance = float(np.mean(np.asarray(distances) <= epsilon))
    return standardized, epsilon, acceptance


def _generate_attacks(
    model: DualHeadModel,
    anchors: np.ndarray,
    visible_rank: int,
    standardized_projection: np.ndarray,
    epsilon: float,
    *,
    payload_delta: float,
    behavior_gamma: float,
    local_radius: float,
    attack_steps: int,
    latent_bound: float,
) -> tuple[dict[str, Any] | None, ...]:
    projection = model.control_head[:, :visible_rank].T
    candidates: list[dict[str, Any] | None] = []
    for anchor_index, anchor in enumerate(anchors):
        hidden_jacobian = model.hidden_jacobian(anchor)
        geometry = tunnel_geometry(
            projection @ hidden_jacobian,
            model.main_head @ hidden_jacobian,
        )
        direction = np.asarray(geometry.direction, dtype=float)
        selected: dict[str, Any] | None = None
        if np.linalg.norm(direction) > 0.0:
            anchor_control = _control_vector(model, anchor, standardized_projection)
            anchor_behavior = float(model.main(anchor))
            for step in np.linspace(payload_delta, local_radius, attack_steps):
                for sign in (1.0, -1.0):
                    latent = anchor + sign * step * direction
                    if np.any(latent < -latent_bound) or np.any(latent > latent_bound):
                        continue
                    control_distance = float(
                        np.linalg.norm(
                            _control_vector(model, latent, standardized_projection)
                            - anchor_control
                        )
                    )
                    behavior_distance = abs(float(model.main(latent)) - anchor_behavior)
                    if control_distance <= epsilon and behavior_distance >= behavior_gamma:
                        selected = {
                            "anchor_index": anchor_index,
                            "latent": latent,
                            "step": float(step),
                            "control_distance": control_distance,
                            "behavior_distance": behavior_distance,
                            "openness": geometry.openness,
                            "tunnel_dimension": geometry.tunnel_dimension,
                        }
                        break
                if selected is not None:
                    break
        candidates.append(selected)
    return tuple(candidates)


def _orthonormal_columns(vectors: np.ndarray, maximum_columns: int) -> np.ndarray:
    if vectors.shape[1] == 0:
        return np.zeros((vectors.shape[0], 0), dtype=float)
    q, r = np.linalg.qr(vectors)
    diagonal = np.abs(np.diag(r))
    tolerance = 1e-10 * max(float(diagonal.max()) if diagonal.size else 0.0, 1.0)
    rank = int(np.count_nonzero(diagonal > tolerance))
    return q[:, : min(maximum_columns, rank)]


def _rowspace_basis(matrix: np.ndarray) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = float(singular_values[0]) if singular_values.size else 1.0
    rank = int(np.count_nonzero(singular_values > 1e-10 * max(scale, 1.0)))
    return vh[:rank].T.copy()


def _append_rows(projection: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return projection.copy() if columns.shape[1] == 0 else np.vstack([projection, columns.T])


def _interventions(
    model: DualHeadModel,
    projection: np.ndarray,
    calibration_anchors: np.ndarray,
    calibration_attacks: Sequence[dict[str, Any] | None],
    hidden_support: np.ndarray,
    *,
    closure_rank: int,
    seed: int,
) -> dict[str, np.ndarray]:
    hidden_nullspace = nullspace_basis(projection)
    displacements: list[np.ndarray] = []
    for anchor, attack in zip(calibration_anchors, calibration_attacks, strict=True):
        if attack is None:
            continue
        displacement = model.hidden(attack["latent"]) - model.hidden(anchor)
        if hidden_nullspace.size:
            displacement = hidden_nullspace @ (hidden_nullspace.T @ displacement)
        displacements.append(displacement)
    if displacements:
        left_vectors, _, _ = np.linalg.svd(
            np.stack(displacements, axis=1),
            full_matrices=False,
        )
        targeted = left_vectors[:, : min(closure_rank, left_vectors.shape[1])]
    else:
        targeted = np.zeros((model.hidden_dimension, 0), dtype=float)

    rng = np.random.default_rng(seed)
    if hidden_nullspace.shape[1] > 0:
        random_null = _orthonormal_columns(
            hidden_nullspace
            @ rng.normal(size=(hidden_nullspace.shape[1], closure_rank)),
            closure_rank,
        )
        centered = hidden_support - hidden_support.mean(axis=0)
        covariance = centered.T @ centered / hidden_support.shape[0]
        restricted = hidden_nullspace.T @ covariance @ hidden_nullspace
        eigenvalues, eigenvectors = np.linalg.eigh(restricted)
        order = np.argsort(eigenvalues)[::-1]
        variance_null = _orthonormal_columns(
            hidden_nullspace @ eigenvectors[:, order[:closure_rank]],
            closure_rank,
        )
    else:
        random_null = np.zeros((model.hidden_dimension, 0), dtype=float)
        variance_null = np.zeros((model.hidden_dimension, 0), dtype=float)
    rowspace = _rowspace_basis(projection)
    row_sham = rowspace[:, : min(closure_rank, rowspace.shape[1])]
    return {
        "baseline": projection.copy(),
        "targeted": _append_rows(projection, targeted),
        "random_null": _append_rows(projection, random_null),
        "variance_null": _append_rows(projection, variance_null),
        "row_sham": _append_rows(projection, row_sham),
    }


def _evaluate_projection(
    model: DualHeadModel,
    anchors: np.ndarray,
    attacks: Sequence[dict[str, Any] | None],
    projection: np.ndarray,
    hidden_support: np.ndarray,
    *,
    intervention: str,
    seed: int,
    benign_radius: float,
    benign_repetitions: int,
    benign_quantile: float,
    payload_delta: float,
    behavior_gamma: float,
) -> dict[str, Any]:
    standardized, epsilon, benign_acceptance = _calibrate_epsilon(
        model,
        anchors,
        projection,
        hidden_support,
        seed=seed,
        benign_radius=benign_radius,
        benign_repetitions=benign_repetitions,
        benign_quantile=benign_quantile,
    )
    left: list[EndpointRecord] = []
    right: list[EndpointRecord] = []
    for index, (anchor, attack) in enumerate(zip(anchors, attacks, strict=True)):
        right.append(
            EndpointRecord(
                index=index,
                side="B",
                signature=(index,),
                prefixes=(),
                control=tuple(_control_vector(model, anchor, standardized)),
                payload=tuple(float(value) for value in anchor),
                behavior=float(model.main(anchor)),
                valid=True,
            )
        )
        attack_latent = anchor if attack is None else np.asarray(attack["latent"])
        left.append(
            EndpointRecord(
                index=index,
                side="A",
                signature=(index,),
                prefixes=(),
                control=tuple(_control_vector(model, attack_latent, standardized)),
                payload=tuple(float(value) for value in attack_latent),
                behavior=float(model.main(attack_latent)),
                valid=attack is not None,
            )
        )
    instance = CollisionInstance(
        name=f"dual_head_{intervention}",
        left=tuple(left),
        right=tuple(right),
        criteria=CollisionCriteria(epsilon, payload_delta, behavior_gamma),
        endpoint_local=True,
        metadata={"intervention": intervention},
    )
    statistics = packing_statistics(instance)
    matching = statistics.matching_size
    quantum = product_johnson_cost(len(anchors), matching).total_cost if matching else None
    return {
        "intervention": intervention,
        "projection_rows": int(projection.shape[0]),
        "epsilon": epsilon,
        "benign_acceptance": benign_acceptance,
        "edge_count": statistics.edge_count,
        "matching_size": matching,
        "packing_fraction": matching / len(anchors),
        "classical_cost": classical_packed_cost(len(anchors), matching)
        if matching
        else None,
        "quantum_proxy": quantum,
    }
