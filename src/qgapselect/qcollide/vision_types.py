"""Immutable data contracts for the real-image Q-COLLIDE experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import CollisionTopologyMetrics
from .topology_persistence import CollisionFiltrationProfile


@dataclass(frozen=True)
class AnchorPanel:
    images: np.ndarray
    labels: np.ndarray
    hidden: np.ndarray
    logits: np.ndarray
    hidden_jacobians: np.ndarray
    targets: np.ndarray

    @property
    def size(self) -> int:
        return len(self.images)


@dataclass(frozen=True)
class BenignPanel:
    anchor_hidden: np.ndarray
    perturbed_hidden: np.ndarray


@dataclass(frozen=True)
class ProjectionCalibration:
    projection: np.ndarray
    standardized_projection: np.ndarray
    control_epsilon: float
    payload_delta: float
    control_acceptance: float
    payload_exceedance: float


@dataclass(frozen=True)
class AttackCandidate:
    source_index: int
    source_label: int
    target_label: int
    image: np.ndarray
    hidden: np.ndarray
    logits: np.ndarray
    input_l2: float
    source_control_distance: float
    source_payload_distance: float
    openness: float
    tunnel_dimension: int


@dataclass(frozen=True)
class PackingResult:
    edge_count: int
    matching_size: int
    packing_fraction: float
    candidate_fraction: float
    mean_input_l2: float | None
    topology: CollisionTopologyMetrics | None = None
    filtration: CollisionFiltrationProfile | None = None


__all__ = [
    "AnchorPanel",
    "AttackCandidate",
    "BenignPanel",
    "PackingResult",
    "ProjectionCalibration",
]
