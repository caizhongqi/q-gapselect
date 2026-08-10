"""Change-point analysis for collision-topology trajectories.

A detected breakpoint is a statistical candidate, not by itself evidence of a
thermodynamic phase transition. The caller must establish finite-size scaling,
replication, and robustness across thresholds before using phase-transition
language.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .topology import CollisionTopologyMetrics

TopologyMetricName = Literal[
    "capacity_fraction",
    "normalized_component_edge_entropy",
    "displacement_entropy_rank",
    "beta1_density",
]


@dataclass(frozen=True)
class TopologySnapshot:
    step: float
    capacity_fraction: float
    normalized_component_edge_entropy: float
    displacement_entropy_rank: float | None
    beta1_density: float


@dataclass(frozen=True)
class TopologyChangePoint:
    metric: str
    split_index: int
    transition_step: float
    one_segment_sse: float
    two_segment_sse: float
    relative_sse_improvement: float
    slope_before: float
    slope_after: float
    value_jump: float


def topology_snapshot(step: float, metrics: CollisionTopologyMetrics) -> TopologySnapshot:
    if not np.isfinite(step):
        raise ValueError("step must be finite")
    return TopologySnapshot(
        step=float(step),
        capacity_fraction=metrics.capacity_fraction,
        normalized_component_edge_entropy=metrics.normalized_component_edge_entropy,
        displacement_entropy_rank=metrics.displacement_entropy_rank,
        beta1_density=(metrics.beta1_active / metrics.edge_count) if metrics.edge_count else 0.0,
    )


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    matrix = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ coefficients
    return float(coefficients[1]), float(coefficients[0]), float(residual @ residual)


def detect_topology_change_point(
    snapshots: Sequence[TopologySnapshot],
    *,
    metric: TopologyMetricName = "capacity_fraction",
    minimum_segment: int = 2,
) -> TopologyChangePoint:
    """Fit the best two-line breakpoint against a one-line null model."""

    if minimum_segment < 2:
        raise ValueError("minimum_segment must be at least two")
    if len(snapshots) < 2 * minimum_segment:
        raise ValueError("trajectory is too short for the requested segment size")
    ordered = sorted(snapshots, key=lambda item: item.step)
    steps = np.asarray([item.step for item in ordered], dtype=float)
    if np.any(np.diff(steps) <= 0.0):
        raise ValueError("snapshot steps must be unique")
    raw_values = [getattr(item, metric) for item in ordered]
    if any(value is None for value in raw_values):
        raise ValueError(f"metric {metric!r} is unavailable in at least one snapshot")
    values = np.asarray(raw_values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory metric values must be finite")

    _, _, one_sse = _linear_fit(steps, values)
    best: tuple[float, int, float, float, float] | None = None
    for split in range(minimum_segment, len(ordered) - minimum_segment + 1):
        before_slope, before_intercept, before_sse = _linear_fit(
            steps[:split],
            values[:split],
        )
        after_slope, after_intercept, after_sse = _linear_fit(
            steps[split:],
            values[split:],
        )
        total_sse = before_sse + after_sse
        transition = 0.5 * (steps[split - 1] + steps[split])
        before_value = before_intercept + before_slope * transition
        after_value = after_intercept + after_slope * transition
        candidate = (
            total_sse,
            split,
            before_slope,
            after_slope,
            after_value - before_value,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    two_sse, split, before_slope, after_slope, jump = best
    improvement = 0.0 if one_sse <= 1e-15 else 1.0 - two_sse / one_sse
    return TopologyChangePoint(
        metric=metric,
        split_index=split,
        transition_step=float(0.5 * (steps[split - 1] + steps[split])),
        one_segment_sse=one_sse,
        two_segment_sse=two_sse,
        relative_sse_improvement=float(improvement),
        slope_before=before_slope,
        slope_after=after_slope,
        value_jump=float(jump),
    )


def analyze_topology_trajectory(
    snapshots: Sequence[TopologySnapshot],
    *,
    minimum_segment: int = 2,
) -> dict[str, TopologyChangePoint]:
    """Analyze every topology metric available across a trajectory."""

    names: tuple[TopologyMetricName, ...] = (
        "capacity_fraction",
        "normalized_component_edge_entropy",
        "displacement_entropy_rank",
        "beta1_density",
    )
    output: dict[str, TopologyChangePoint] = {}
    for name in names:
        if name == "displacement_entropy_rank" and any(
            item.displacement_entropy_rank is None for item in snapshots
        ):
            continue
        output[name] = detect_topology_change_point(
            snapshots,
            metric=name,
            minimum_segment=minimum_segment,
        )
    return output


__all__ = [
    "TopologyChangePoint",
    "TopologyMetricName",
    "TopologySnapshot",
    "analyze_topology_trajectory",
    "detect_topology_change_point",
    "topology_snapshot",
]
