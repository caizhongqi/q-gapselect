"""Cross-dataset meta-analysis for Q-COLLIDE capacity summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import log2, sqrt

import numpy as np


@dataclass(frozen=True)
class CapacityCell:
    dataset: str
    architecture: str
    visible_rank: int
    openness: float
    packing_fraction: float
    standard_error: float | None


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


def _baseline_rows(document: Mapping[str, object]) -> Sequence[object]:
    for key in ("baseline_and_closure", "baseline_and_rank6_closure"):
        if key in document:
            return _sequence(document[key], key)
    raise ValueError(
        "summary must contain baseline_and_closure or baseline_and_rank6_closure"
    )


def extract_capacity_cells(document: Mapping[str, object]) -> tuple[CapacityCell, ...]:
    """Extract one baseline capacity cell per dataset/architecture/rank."""

    claim_scope = _mapping(document.get("claim_scope"), "claim_scope")
    dataset = claim_scope.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("claim_scope.dataset must be a non-empty string")
    raw_rows = _baseline_rows(document)
    cells: list[CapacityCell] = []
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, f"baseline row {index}")
        architecture = row.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError(f"row {index} architecture must be non-empty")
        rank_value = row.get("visible_rank")
        if isinstance(rank_value, bool) or not isinstance(rank_value, int):
            raise ValueError(f"row {index} visible_rank must be an integer")
        if rank_value <= 0:
            raise ValueError(f"row {index} visible_rank must be positive")
        openness = _number(row.get("baseline_openness"), f"row {index} openness")
        packing = _number(row.get("baseline_packing"), f"row {index} packing")
        if not 0.0 <= openness <= 1.0:
            raise ValueError(f"row {index} openness must lie in [0,1]")
        if not 0.0 <= packing <= 1.0:
            raise ValueError(f"row {index} packing must lie in [0,1]")
        key = (architecture, rank_value)
        if key in seen:
            raise ValueError(f"duplicate baseline cell for {key}")
        seen.add(key)
        standard_error = row.get("baseline_se")
        cells.append(
            CapacityCell(
                dataset=dataset,
                architecture=architecture,
                visible_rank=rank_value,
                openness=openness,
                packing_fraction=packing,
                standard_error=(
                    None
                    if standard_error is None
                    else _number(standard_error, f"row {index} standard_error")
                ),
            )
        )
    if not cells:
        raise ValueError("baseline rows must be non-empty")
    return tuple(cells)


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size < 2 or float(lhs.std()) == 0.0 or float(rhs.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _fit_linear(
    cells: Sequence[CapacityCell],
    *,
    include_dataset: bool,
    include_architecture: bool,
    include_log_rank: bool,
) -> dict[str, object]:
    datasets = sorted({cell.dataset for cell in cells})
    architectures = sorted({cell.architecture for cell in cells})
    columns = ["intercept", "openness"]
    columns.extend(f"dataset:{value}" for value in datasets[1:] if include_dataset)
    columns.extend(
        f"architecture:{value}" for value in architectures[1:] if include_architecture
    )
    if include_log_rank:
        columns.append("log2_visible_rank")
    design: list[list[float]] = []
    response: list[float] = []
    for cell in cells:
        row = [1.0, cell.openness]
        if include_dataset:
            row.extend(1.0 if cell.dataset == value else 0.0 for value in datasets[1:])
        if include_architecture:
            row.extend(
                1.0 if cell.architecture == value else 0.0
                for value in architectures[1:]
            )
        if include_log_rank:
            row.append(log2(cell.visible_rank))
        design.append(row)
        response.append(cell.packing_fraction)
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(response, dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    predicted = matrix @ coefficients
    residual = target - predicted
    total = target - target.mean()
    residual_sum = float(residual @ residual)
    total_sum = float(total @ total)
    return {
        "columns": columns,
        "coefficients": {
            name: float(value) for name, value in zip(columns, coefficients, strict=True)
        },
        "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0,
        "rmse": sqrt(float(np.mean(residual * residual))),
        "sample_count": len(cells),
        "design_rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
    }


def _leave_one_dataset_out(cells: Sequence[CapacityCell]) -> dict[str, object]:
    results: dict[str, object] = {}
    datasets = sorted({cell.dataset for cell in cells})
    architectures = sorted({cell.architecture for cell in cells})

    def row(cell: CapacityCell) -> list[float]:
        values = [1.0, cell.openness]
        values.extend(
            1.0 if cell.architecture == value else 0.0
            for value in architectures[1:]
        )
        values.append(log2(cell.visible_rank))
        return values

    for held_out in datasets:
        train = [cell for cell in cells if cell.dataset != held_out]
        test = [cell for cell in cells if cell.dataset == held_out]
        if not train or not test:
            continue
        x_train = np.asarray([row(cell) for cell in train], dtype=float)
        y_train = np.asarray([cell.packing_fraction for cell in train], dtype=float)
        coefficients = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        x_test = np.asarray([row(cell) for cell in test], dtype=float)
        y_test = np.asarray([cell.packing_fraction for cell in test], dtype=float)
        residual = y_test - x_test @ coefficients
        results[held_out] = {
            "cell_count": len(test),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": sqrt(float(np.mean(residual * residual))),
        }
    return results


def _monotonicity(cells: Sequence[CapacityCell]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in sorted({cell.dataset for cell in cells}):
        for architecture in sorted({cell.architecture for cell in cells}):
            group = sorted(
                [
                    cell
                    for cell in cells
                    if cell.dataset == dataset and cell.architecture == architecture
                ],
                key=lambda cell: cell.visible_rank,
            )
            if not group:
                continue
            comparisons = max(0, len(group) - 1)
            packing_passes = sum(
                right.packing_fraction <= left.packing_fraction + 1e-12
                for left, right in zip(group, group[1:], strict=False)
            )
            openness_passes = sum(
                right.openness <= left.openness + 1e-12
                for left, right in zip(group, group[1:], strict=False)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "cell_count": len(group),
                    "packing_nonincreasing_fraction": (
                        packing_passes / comparisons if comparisons else 1.0
                    ),
                    "openness_nonincreasing_fraction": (
                        openness_passes / comparisons if comparisons else 1.0
                    ),
                }
            )
    return rows


def build_capacity_meta_summary(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the frozen cross-dataset Geometry-to-Packing meta summary."""

    if len(documents) < 2:
        raise ValueError("at least two dataset summaries are required")
    cells = tuple(cell for document in documents for cell in extract_capacity_cells(document))
    datasets = sorted({cell.dataset for cell in cells})
    if len(datasets) != len(documents):
        raise ValueError("each input document must represent a unique dataset")
    openness = np.asarray([cell.openness for cell in cells], dtype=float)
    packing = np.asarray([cell.packing_fraction for cell in cells], dtype=float)
    return {
        "artifact_type": "qcollide_cross_dataset_capacity_meta_summary",
        "schema_version": 1,
        "datasets": datasets,
        "architectures": sorted({cell.architecture for cell in cells}),
        "cell_count": len(cells),
        "baseline_cells": [asdict(cell) for cell in cells],
        "associations": {
            "pearson_openness_vs_packing": _pearson(openness, packing),
        },
        "linear_models": {
            "openness_dataset_architecture": _fit_linear(
                cells,
                include_dataset=True,
                include_architecture=True,
                include_log_rank=False,
            ),
            "openness_dataset_architecture_log2_rank": _fit_linear(
                cells,
                include_dataset=True,
                include_architecture=True,
                include_log_rank=True,
            ),
        },
        "leave_one_dataset_out": _leave_one_dataset_out(cells),
        "monotonicity": _monotonicity(cells),
        "claim_scope": {
            "supports_cross_dataset_geometry_to_packing": True,
            "universal_dataset_invariant_scalar_curve_claimed": False,
            "production_scale_claimed": False,
            "coherent_quantum_execution": False,
        },
    }


__all__ = ["CapacityCell", "build_capacity_meta_summary", "extract_capacity_cells"]
