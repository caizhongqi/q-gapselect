"""Audit which collision-topology quantities are identifiable in legacy summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TopologyEvidenceCell:
    dataset: str
    architecture: str
    visible_rank: int
    openness: float
    capacity_fraction: float
    displacement_rank_95: float
    gradient_rank_95: float


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
    raise ValueError("summary lacks frozen baseline rows")


def _spectrum_values(row: Mapping[str, object], index: int) -> tuple[float, float]:
    displacement = row.get("disp_rank95", row.get("mean_displacement_rank_95"))
    gradient = row.get("grad_rank95", row.get("mean_gradient_rank_95"))
    return (
        _number(displacement, f"spectrum {index} displacement rank"),
        _number(gradient, f"spectrum {index} gradient rank"),
    )


def extract_topology_evidence_cells(
    document: Mapping[str, object],
) -> tuple[TopologyEvidenceCell, ...]:
    """Extract capacity and spectral-dimension evidence from a compact summary."""

    claim_scope = _mapping(document.get("claim_scope"), "claim_scope")
    dataset = claim_scope.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("claim_scope.dataset must be a non-empty string")
    spectra = _sequence(document.get("static_spectra"), "static_spectra")
    spectrum_lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for index, raw in enumerate(spectra):
        row = _mapping(raw, f"spectrum {index}")
        architecture = row.get("architecture")
        rank = row.get("visible_rank")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError(f"spectrum {index} architecture must be non-empty")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"spectrum {index} visible_rank must be positive")
        key = (architecture, rank)
        if key in spectrum_lookup:
            raise ValueError(f"duplicate spectrum cell {key}")
        spectrum_lookup[key] = _spectrum_values(row, index)

    cells: list[TopologyEvidenceCell] = []
    for index, raw in enumerate(_baseline_rows(document)):
        row = _mapping(raw, f"baseline {index}")
        architecture = row.get("architecture")
        rank = row.get("visible_rank")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError(f"baseline {index} architecture must be non-empty")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"baseline {index} visible_rank must be positive")
        key = (architecture, rank)
        if key not in spectrum_lookup:
            raise ValueError(f"missing spectrum for baseline cell {key}")
        displacement_rank, gradient_rank = spectrum_lookup[key]
        cells.append(
            TopologyEvidenceCell(
                dataset=dataset,
                architecture=architecture,
                visible_rank=rank,
                openness=_number(row.get("baseline_openness"), f"baseline {index} openness"),
                capacity_fraction=_number(
                    row.get("baseline_packing"),
                    f"baseline {index} capacity",
                ),
                displacement_rank_95=displacement_rank,
                gradient_rank_95=gradient_rank,
            )
        )
    return tuple(cells)


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size < 2 or float(lhs.std()) == 0.0 or float(rhs.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def build_topology_evidence_audit(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reanalyze legacy experiments without inventing unavailable graph topology."""

    if len(documents) < 2:
        raise ValueError("at least two independent dataset summaries are required")
    cells = tuple(
        cell for document in documents for cell in extract_topology_evidence_cells(document)
    )
    datasets = sorted({cell.dataset for cell in cells})
    if len(datasets) != len(documents):
        raise ValueError("each summary must represent a unique dataset")
    capacity = np.asarray([cell.capacity_fraction for cell in cells], dtype=float)
    openness = np.asarray([cell.openness for cell in cells], dtype=float)
    displacement_rank = np.asarray(
        [cell.displacement_rank_95 for cell in cells],
        dtype=float,
    )
    gradient_rank = np.asarray([cell.gradient_rank_95 for cell in cells], dtype=float)
    return {
        "artifact_type": "qcollide_legacy_collision_topology_evidence_audit",
        "schema_version": 1,
        "datasets": datasets,
        "cell_count": len(cells),
        "cells": [asdict(cell) for cell in cells],
        "associations": {
            "openness_vs_capacity": _pearson(openness, capacity),
            "displacement_rank95_vs_capacity": _pearson(displacement_rank, capacity),
            "openness_vs_displacement_rank95": _pearson(openness, displacement_rank),
            "gradient_rank95_vs_capacity": _pearson(gradient_rank, capacity),
        },
        "metric_identifiability": {
            "K_C_matching_capacity": "identified",
            "D_C_displacement_spectral_proxy": "identified",
            "H_C_component_entropy": "not_identifiable_without_adjacency",
            "beta_0_active_components": "not_identifiable_without_adjacency",
            "beta_1_collision_cycles": "not_identifiable_without_adjacency",
        },
        "claim_scope": {
            "supports_full_collision_topology": False,
            "supports_partial_capacity_dimension_profile": True,
            "retrospective_entropy_reconstruction_claimed": False,
            "phase_transition_claimed": False,
        },
    }


__all__ = [
    "TopologyEvidenceCell",
    "build_topology_evidence_audit",
    "extract_topology_evidence_cells",
]
