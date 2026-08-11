"""Pre-registered small-sample statistics for Q-COLLIDE main experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Literal

import numpy as np

Alternative = Literal["greater", "less", "two-sided"]


def exact_paired_sign_flip_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
) -> dict[str, object]:
    """Exact paired randomization test by enumerating every sign flip.

    This is intended for the five-seed confirmatory experiments. It does not
    treat control ranks or thresholds as independent replicates.
    """

    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.ndim != 1 or rhs.ndim != 1 or lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("left and right must be aligned nonempty vectors")
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("left and right must be finite")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("unsupported alternative")
    if lhs.size > 20:
        raise ValueError("exact enumeration is restricted to at most 20 pairs")

    differences = lhs - rhs
    observed = float(differences.mean())
    randomized = np.asarray(
        [
            float(np.mean(differences * np.asarray(signs, dtype=float)))
            for signs in product((-1.0, 1.0), repeat=lhs.size)
        ],
        dtype=float,
    )
    tolerance = 1e-15
    if alternative == "greater":
        extreme = randomized >= observed - tolerance
    elif alternative == "less":
        extreme = randomized <= observed + tolerance
    else:
        extreme = np.abs(randomized) >= abs(observed) - tolerance
    return {
        "pair_count": int(lhs.size),
        "alternative": alternative,
        "mean_left": float(lhs.mean()),
        "mean_right": float(rhs.mean()),
        "mean_difference": observed,
        "median_difference": float(np.median(differences)),
        "minimum_difference": float(differences.min()),
        "maximum_difference": float(differences.max()),
        "same_sign_fraction": float(
            max(np.mean(differences >= 0.0), np.mean(differences <= 0.0))
        ),
        "exact_p_value": float(np.mean(extreme)),
        "differences": differences.tolist(),
    }


def _matched_observation_index(
    artifact: Mapping[str, object],
) -> dict[tuple[str, int, int], Mapping[str, object]]:
    if str(artifact.get("artifact_type")) != "qcollide_cifar_performance_matched_main_table":
        raise ValueError("unexpected CIFAR main-table artifact type")
    output: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for row in artifact["matched_observations"]:
        key = (
            str(row["architecture"]),
            int(row["model_seed"]),
            int(row["visible_rank"]),
        )
        if key in output:
            raise ValueError(f"duplicate matched observation {key}")
        output[key] = row
    return output


def paired_architecture_contrast(
    artifact: Mapping[str, object],
    *,
    left_architecture: str,
    right_architecture: str,
    visible_rank: int,
    metric: str,
    alternative: Alternative = "two-sided",
) -> dict[str, object]:
    """Compare two architectures using matched model-seed replicates only."""

    index = _matched_observation_index(artifact)
    seeds_left = {
        seed
        for architecture, seed, rank in index
        if architecture == left_architecture and rank == visible_rank
    }
    seeds_right = {
        seed
        for architecture, seed, rank in index
        if architecture == right_architecture and rank == visible_rank
    }
    if seeds_left != seeds_right or not seeds_left:
        raise ValueError("architectures do not share a complete matched seed set")
    seeds = sorted(seeds_left)
    left = [float(index[(left_architecture, seed, visible_rank)][metric]) for seed in seeds]
    right = [
        float(index[(right_architecture, seed, visible_rank)][metric]) for seed in seeds
    ]
    result = exact_paired_sign_flip_test(left, right, alternative=alternative)
    return {
        "left_architecture": left_architecture,
        "right_architecture": right_architecture,
        "visible_rank": visible_rank,
        "metric": metric,
        "model_seeds": seeds,
        **result,
    }


def basin_capacity_decomposition(
    artifact: Mapping[str, object],
    *,
    left_architecture: str,
    right_architecture: str,
    visible_rank: int,
) -> dict[str, object]:
    """Symmetrically decompose C=B*M into basin-density and multiplicity effects."""

    index = _matched_observation_index(artifact)
    seeds_left = {
        seed
        for architecture, seed, rank in index
        if architecture == left_architecture and rank == visible_rank
    }
    seeds_right = {
        seed
        for architecture, seed, rank in index
        if architecture == right_architecture and rank == visible_rank
    }
    if seeds_left != seeds_right or not seeds_left:
        raise ValueError("architectures do not share a complete matched seed set")

    rows: list[dict[str, float | int]] = []
    for seed in sorted(seeds_left):
        left = index[(left_architecture, seed, visible_rank)]
        right = index[(right_architecture, seed, visible_rank)]
        capacity_left = float(left["capacity_fraction"])
        capacity_right = float(right["capacity_fraction"])
        basin_left = float(left["basin_density"])
        basin_right = float(right["basin_density"])
        multiplicity_left = capacity_left / basin_left if basin_left > 0.0 else 0.0
        multiplicity_right = capacity_right / basin_right if basin_right > 0.0 else 0.0
        if basin_left == 0.0 and capacity_left != 0.0:
            raise ValueError("positive capacity with zero basin density")
        if basin_right == 0.0 and capacity_right != 0.0:
            raise ValueError("positive capacity with zero basin density")
        basin_contribution = 0.5 * (basin_left - basin_right) * (
            multiplicity_left + multiplicity_right
        )
        multiplicity_contribution = 0.5 * (
            multiplicity_left - multiplicity_right
        ) * (basin_left + basin_right)
        total_difference = capacity_left - capacity_right
        closure_error = abs(
            total_difference - basin_contribution - multiplicity_contribution
        )
        if closure_error > 1e-10:
            raise RuntimeError("capacity decomposition failed to close")
        rows.append(
            {
                "model_seed": seed,
                "capacity_difference": total_difference,
                "basin_density_contribution": basin_contribution,
                "within_basin_multiplicity_contribution": multiplicity_contribution,
                "closure_error": closure_error,
            }
        )

    mean_gap = float(np.mean([row["capacity_difference"] for row in rows]))
    mean_basin = float(np.mean([row["basin_density_contribution"] for row in rows]))
    mean_multiplicity = float(
        np.mean([row["within_basin_multiplicity_contribution"] for row in rows])
    )
    denominator = abs(mean_gap)
    return {
        "left_architecture": left_architecture,
        "right_architecture": right_architecture,
        "visible_rank": visible_rank,
        "per_seed": rows,
        "mean_capacity_difference": mean_gap,
        "mean_basin_density_contribution": mean_basin,
        "mean_within_basin_multiplicity_contribution": mean_multiplicity,
        "basin_contribution_fraction_of_absolute_gap": (
            mean_basin / denominator if denominator > 0.0 else None
        ),
        "multiplicity_contribution_fraction_of_absolute_gap": (
            mean_multiplicity / denominator if denominator > 0.0 else None
        ),
    }


def preregistered_cifar_contrasts(
    artifact: Mapping[str, object],
    *,
    primary_rank: int = 8,
) -> dict[str, object]:
    """Run the frozen primary and robustness contrasts for one CIFAR dataset."""

    primary = paired_architecture_contrast(
        artifact,
        left_architecture="vit_tiny",
        right_architecture="resnet18",
        visible_rank=primary_rank,
        metric="capacity_fraction",
        alternative="greater",
    )
    mechanism = paired_architecture_contrast(
        artifact,
        left_architecture="vit_tiny",
        right_architecture="resnet18",
        visible_rank=primary_rank,
        metric="basin_density",
        alternative="greater",
    )
    robustness = [
        paired_architecture_contrast(
            artifact,
            left_architecture="vit_tiny",
            right_architecture="resnet18",
            visible_rank=rank,
            metric="capacity_fraction",
            alternative="greater",
        )
        for rank in (1, 32)
        if rank in set(int(value) for value in artifact["visible_ranks"])
    ]
    decomposition = basin_capacity_decomposition(
        artifact,
        left_architecture="vit_tiny",
        right_architecture="resnet18",
        visible_rank=primary_rank,
    )
    return {
        "artifact_type": "qcollide_cifar_preregistered_statistics",
        "schema_version": 1,
        "dataset": artifact["dataset"],
        "primary_rank": primary_rank,
        "primary_confirmatory_contrast": primary,
        "mechanism_secondary_contrast": mechanism,
        "rank_robustness_contrasts": robustness,
        "capacity_gap_decomposition": decomposition,
        "claim_boundary": {
            "control_ranks_treated_as_independent_replicates": False,
            "threshold_filtration_points_treated_as_independent_replicates": False,
            "primary_alternative_selected_after_results": False,
            "primary_confirmatory_metric": "capacity_fraction",
            "primary_confirmatory_pair": "vit_tiny > resnet18",
        },
    }


__all__ = [
    "basin_capacity_decomposition",
    "exact_paired_sign_flip_test",
    "paired_architecture_contrast",
    "preregistered_cifar_contrasts",
]
