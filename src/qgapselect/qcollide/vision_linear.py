"""Linear-algebra utilities for neural control projections."""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch


def row_basis(matrix: np.ndarray, *, tolerance: float = 1e-9) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape[0] == 0:
        return np.zeros((0, values.shape[1]), dtype=float)
    _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    return vh[:rank].copy()


def nullspace_basis(matrix: np.ndarray, *, tolerance: float = 1e-9) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    _, singular_values, vh = np.linalg.svd(values, full_matrices=True)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    return vh[rank:].T.copy()


def standardized_projection(projection: np.ndarray, hidden_support: np.ndarray) -> np.ndarray:
    """Return a covariance-whitened, row-space-invariant control map.

    Per-coordinate standardization depends on which orthonormal basis happens
    to represent a subspace. Whitening the projected support covariance instead
    makes Euclidean control distances invariant under orthogonal basis changes.
    """

    basis = row_basis(projection)
    if basis.shape[0] == 0:
        return basis
    support = np.asarray(hidden_support, dtype=float)
    outputs = support @ basis.T
    centered = outputs - outputs.mean(axis=0, keepdims=True)
    denominator = max(len(centered) - 1, 1)
    covariance = centered.T @ centered / denominator
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(eigenvalues.max()) if eigenvalues.size else 0.0, 1.0)
    floor = 1e-8 * largest
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    return inverse_root @ basis


def combine_projection(projection: np.ndarray, added_columns: np.ndarray) -> np.ndarray:
    """Add genuinely new control directions while preserving a redundant sham."""

    base = row_basis(projection)
    columns = np.asarray(added_columns, dtype=float)
    if columns.ndim != 2:
        raise ValueError("added_columns must be a matrix")
    if columns.shape[1] == 0:
        return base
    combined = row_basis(np.vstack([base, columns.T]))
    return base if combined.shape[0] == base.shape[0] else combined


def control_output(hidden: np.ndarray, standardized: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=float)
    if standardized.shape[0] == 0:
        return np.zeros(values.shape[:-1] + (0,), dtype=float)
    return values @ standardized.T / sqrt(standardized.shape[0])


def residual_hidden(hidden: np.ndarray, projection: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=float)
    basis = row_basis(projection)
    if basis.shape[0] == 0:
        return values.copy()
    return values - (values @ basis.T) @ basis


def visible_control_projection(model: torch.nn.Module, visible_rank: int) -> np.ndarray:
    control_weight = model.control_head.weight.detach().cpu().numpy()
    _, singular_values, vh = np.linalg.svd(control_weight, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    numerical_rank = int(np.count_nonzero(singular_values > 1e-9 * scale))
    if not 1 <= visible_rank <= numerical_rank:
        raise ValueError(
            f"visible_rank={visible_rank} exceeds the numerical control-head rank "
            f"{numerical_rank}"
        )
    return vh[:visible_rank].copy()


__all__ = [
    "combine_projection",
    "control_output",
    "nullspace_basis",
    "residual_hidden",
    "row_basis",
    "standardized_projection",
    "visible_control_projection",
]
