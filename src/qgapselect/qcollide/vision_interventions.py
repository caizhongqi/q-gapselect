"""Dangerous-mode spectra and rank-matched control interventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .vision_linear import nullspace_basis, row_basis
from .vision_types import AnchorPanel, AttackCandidate


def dangerous_displacements(
    attacks: Sequence[AttackCandidate | None],
    anchors: AnchorPanel,
    projection: np.ndarray,
) -> np.ndarray:
    nullspace = nullspace_basis(projection)
    columns: list[np.ndarray] = []
    for attack in attacks:
        if attack is None:
            continue
        displacement = attack.hidden - anchors.hidden[attack.source_index]
        projected = (
            nullspace @ (nullspace.T @ displacement)
            if nullspace.shape[1]
            else np.zeros_like(displacement)
        )
        if np.linalg.norm(projected) > 1e-10:
            columns.append(projected)
    if not columns:
        return np.zeros((anchors.hidden.shape[1], 0), dtype=float)
    return np.stack(columns, axis=1)


def dangerous_hidden_gradients(
    model: torch.nn.Module,
    anchors: AnchorPanel,
    projection: np.ndarray,
) -> np.ndarray:
    """Control-null hidden target gradients learned only from calibration anchors."""

    main_weight = model.main_head.weight.detach().cpu().numpy()
    nullspace = nullspace_basis(projection)
    columns: list[np.ndarray] = []
    for true_label, target_label in zip(anchors.labels, anchors.targets, strict=True):
        gradient = main_weight[int(target_label)] - main_weight[int(true_label)]
        projected = (
            nullspace @ (nullspace.T @ gradient)
            if nullspace.shape[1]
            else np.zeros_like(gradient)
        )
        if np.linalg.norm(projected) > 1e-10:
            columns.append(projected)
    if not columns:
        return np.zeros((anchors.hidden.shape[1], 0), dtype=float)
    return np.stack(columns, axis=1)


def dangerous_spectrum(directions: np.ndarray) -> dict[str, Any]:
    values = np.asarray(directions, dtype=float)
    if values.shape[1] == 0:
        return {
            "singular_values": [],
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "total_energy": 0.0,
            "basis": np.zeros((values.shape[0], 0), dtype=float),
        }
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    energy = singular_values**2
    cumulative = np.cumsum(energy) / energy.sum()

    def rank_at(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    return {
        "singular_values": [float(value) for value in singular_values],
        "rank_90": rank_at(0.90),
        "rank_95": rank_at(0.95),
        "rank_99": rank_at(0.99),
        "total_energy": float(energy.sum()),
        "basis": left,
    }


def intervention_bases(
    projection: np.ndarray,
    dangerous_basis: np.ndarray,
    hidden_support: np.ndarray,
    *,
    maximum_rank: int,
    seed: int,
) -> dict[str, np.ndarray]:
    hidden_dimension = hidden_support.shape[1]
    nullspace = nullspace_basis(projection)
    targeted = dangerous_basis[:, : min(maximum_rank, dangerous_basis.shape[1])]
    rng = np.random.default_rng(seed)
    if nullspace.shape[1]:
        random_columns = nullspace @ rng.normal(size=(nullspace.shape[1], maximum_rank))
        random_null = row_basis(random_columns.T).T
        centered = hidden_support - hidden_support.mean(axis=0)
        covariance = centered.T @ centered / len(centered)
        restricted = nullspace.T @ covariance @ nullspace
        eigenvalues, eigenvectors = np.linalg.eigh(restricted)
        order = np.argsort(eigenvalues)[::-1]
        variance_null = nullspace @ eigenvectors[:, order[:maximum_rank]]
    else:
        random_null = np.zeros((hidden_dimension, 0), dtype=float)
        variance_null = np.zeros((hidden_dimension, 0), dtype=float)
    rowspace = row_basis(projection).T
    return {
        "targeted": targeted,
        "random_null": random_null,
        "variance_null": variance_null,
        "row_sham": rowspace,
    }


def residual_dangerous_energy(
    dangerous: np.ndarray,
    added_columns: np.ndarray,
    closure_rank: int,
) -> float:
    if dangerous.shape[1] == 0:
        return 0.0
    selected = added_columns[:, : min(closure_rank, added_columns.shape[1])]
    if selected.shape[1] == 0:
        return float(np.linalg.norm(dangerous, ord="fro") ** 2)
    basis = row_basis(selected.T).T
    residual = dangerous - basis @ (basis.T @ dangerous)
    return float(np.linalg.norm(residual, ord="fro") ** 2)


__all__ = [
    "dangerous_displacements",
    "dangerous_hidden_gradients",
    "dangerous_spectrum",
    "intervention_bases",
    "residual_dangerous_energy",
]
