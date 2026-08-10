"""Aggregate graph-level collision-topology outputs from visual campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TopologyResultCell:
    dataset: str
    architecture: str
    visible_rank: int
    model_count: int
    openness: float
    capacity_fraction: float
    beta0_active: float
    beta1_density: float
    normalized_component_entropy: float
    effective_component_count: float
    largest_component_edge_fraction: float
    independence_ratio: float
    displacement_entropy_rank: float
    displacement_stable_rank: float


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size < 2 or float(lhs.std()) == 0.0 or float(rhs.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _baseline_rows(artifact: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(artifact.get("rows"), "rows")
    selected: list[Mapping[str, object]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"rows[{index}]")
        if row.get("intervention") == "baseline" and row.get("closure_rank") == 0:
            selected.append(row)
    if not selected:
        raise ValueError("artifact contains no baseline topology rows")
    return tuple(selected)


def _row_metric(row: Mapping[str, object], key: str, index: int) -> float:
    if key not in row:
        raise ValueError(f"baseline row {index} lacks topology field {key!r}")
    return _number(row[key], f"baseline row {index} {key}")


def extract_topology_result_cells(
    artifact: Mapping[str, object],
) -> tuple[TopologyResultCell, ...]:
    """Aggregate baseline topology metrics by architecture and visible rank."""

    claim_scope = _mapping(artifact.get("claim_scope"), "claim_scope")
    if claim_scope.get("full_collision_topology_emitted") is not True:
        raise ValueError("artifact was not produced by the graph-instrumented pipeline")
    dataset = claim_scope.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("claim_scope.dataset must be a non-empty string")
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for index, row in enumerate(_baseline_rows(artifact)):
        architecture = row.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError(f"baseline row {index} architecture must be non-empty")
        rank = _integer(row.get("visible_rank"), f"baseline row {index} rank", minimum=1)
        groups.setdefault((architecture, rank), []).append(row)

    cells: list[TopologyResultCell] = []
    for (architecture, rank), rows in sorted(groups.items()):
        arrays: dict[str, np.ndarray] = {}
        keys = (
            "mean_openness",
            "packing_fraction",
            "collision_beta0_active",
            "collision_beta1_active",
            "edge_count",
            "collision_normalized_component_entropy",
            "collision_effective_component_count",
            "collision_largest_component_edge_fraction",
            "collision_independence_ratio",
            "collision_displacement_entropy_rank",
            "collision_displacement_stable_rank",
        )
        for key in keys:
            arrays[key] = np.asarray(
                [_row_metric(row, key, index) for index, row in enumerate(rows)],
                dtype=float,
            )
        beta1_density = np.divide(
            arrays["collision_beta1_active"],
            arrays["edge_count"],
            out=np.zeros_like(arrays["collision_beta1_active"]),
            where=arrays["edge_count"] > 0.0,
        )
        cells.append(
            TopologyResultCell(
                dataset=dataset,
                architecture=architecture,
                visible_rank=rank,
                model_count=len(rows),
                openness=float(arrays["mean_openness"].mean()),
                capacity_fraction=float(arrays["packing_fraction"].mean()),
                beta0_active=float(arrays["collision_beta0_active"].mean()),
                beta1_density=float(beta1_density.mean()),
                normalized_component_entropy=float(
                    arrays["collision_normalized_component_entropy"].mean()
                ),
                effective_component_count=float(
                    arrays["collision_effective_component_count"].mean()
                ),
                largest_component_edge_fraction=float(
                    arrays["collision_largest_component_edge_fraction"].mean()
                ),
                independence_ratio=float(
                    arrays["collision_independence_ratio"].mean()
                ),
                displacement_entropy_rank=float(
                    arrays["collision_displacement_entropy_rank"].mean()
                ),
                displacement_stable_rank=float(
                    arrays["collision_displacement_stable_rank"].mean()
                ),
            )
        )
    return tuple(cells)


def _monotonicity(cells: Sequence[TopologyResultCell]) -> list[dict[str, object]]:
    metrics = (
        "openness",
        "capacity_fraction",
        "beta1_density",
        "displacement_entropy_rank",
    )
    output: list[dict[str, object]] = []
    for architecture in sorted({cell.architecture for cell in cells}):
        group = sorted(
            [cell for cell in cells if cell.architecture == architecture],
            key=lambda cell: cell.visible_rank,
        )
        comparisons = max(0, len(group) - 1)
        row: dict[str, object] = {
            "architecture": architecture,
            "cell_count": len(group),
        }
        for metric in metrics:
            passes = sum(
                getattr(right, metric) <= getattr(left, metric) + 1e-12
                for left, right in zip(group, group[1:], strict=False)
            )
            row[f"{metric}_nonincreasing_fraction"] = (
                passes / comparisons if comparisons else 1.0
            )
        output.append(row)
    return output


def _architecture_contrasts(cells: Sequence[TopologyResultCell]) -> list[dict[str, object]]:
    architectures = sorted({cell.architecture for cell in cells})
    if len(architectures) != 2:
        return []
    first, second = architectures
    lookup = {(cell.architecture, cell.visible_rank): cell for cell in cells}
    ranks = sorted(
        {cell.visible_rank for cell in cells if (first, cell.visible_rank) in lookup}
        & {cell.visible_rank for cell in cells if (second, cell.visible_rank) in lookup}
    )
    metrics = (
        "capacity_fraction",
        "beta1_density",
        "normalized_component_entropy",
        "displacement_entropy_rank",
    )
    output: list[dict[str, object]] = []
    for rank in ranks:
        left = lookup[(first, rank)]
        right = lookup[(second, rank)]
        row: dict[str, object] = {
            "visible_rank": rank,
            "reference_architecture": first,
            "contrasted_architecture": second,
        }
        for metric in metrics:
            row[f"delta_{metric}"] = getattr(right, metric) - getattr(left, metric)
        output.append(row)
    return output


def build_collision_topology_run_summary(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Build a descriptive, graph-level topology summary for one visual run."""

    cells = extract_topology_result_cells(artifact)
    capacity = np.asarray([cell.capacity_fraction for cell in cells], dtype=float)
    variables = {
        "openness": np.asarray([cell.openness for cell in cells], dtype=float),
        "beta1_density": np.asarray([cell.beta1_density for cell in cells], dtype=float),
        "normalized_component_entropy": np.asarray(
            [cell.normalized_component_entropy for cell in cells],
            dtype=float,
        ),
        "effective_component_count": np.asarray(
            [cell.effective_component_count for cell in cells],
            dtype=float,
        ),
        "largest_component_edge_fraction": np.asarray(
            [cell.largest_component_edge_fraction for cell in cells],
            dtype=float,
        ),
        "independence_ratio": np.asarray(
            [cell.independence_ratio for cell in cells],
            dtype=float,
        ),
        "displacement_entropy_rank": np.asarray(
            [cell.displacement_entropy_rank for cell in cells],
            dtype=float,
        ),
    }
    return {
        "artifact_type": "qcollide_collision_topology_run_summary",
        "schema_version": 1,
        "dataset": cells[0].dataset,
        "architectures": sorted({cell.architecture for cell in cells}),
        "cell_count": len(cells),
        "baseline_cells": [asdict(cell) for cell in cells],
        "capacity_associations": {
            name: _pearson(values, capacity) for name, values in variables.items()
        },
        "monotonicity": _monotonicity(cells),
        "architecture_contrasts": _architecture_contrasts(cells),
        "claim_scope": {
            "descriptive_graph_topology": True,
            "training_phase_transition_claimed": False,
            "persistent_homology_claimed": False,
            "dataset_invariant_law_claimed": False,
            "quantum_reconstruction_advantage_claimed": False,
        },
    }


__all__ = [
    "TopologyResultCell",
    "build_collision_topology_run_summary",
    "extract_topology_result_cells",
]
