"""Dangerous-direction spectrum and closure-rank dose response for Q-COLLIDE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import exp, log, sqrt
from typing import Any

import numpy as np

from .dual_head_geometry import (
    _calibrate_epsilon,
    _evaluate_projection,
    _generate_attacks,
    _interventions,
    _rowspace_basis,
)
from .dual_head_model import (
    _model_seeds,
    _seed,
    train_dual_head_model,
)
from .geometry import nullspace_basis, tunnel_geometry
from .scaling_common import int_values, number, positive_int


def _nonnegative_int_values(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    output: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
        output.append(item)
    if not output:
        raise ValueError(f"{name} must be non-empty")
    return tuple(output)


def _dangerous_displacements(
    model: Any,
    projection: np.ndarray,
    anchors: np.ndarray,
    attacks: Sequence[dict[str, Any] | None],
) -> np.ndarray:
    hidden_nullspace = nullspace_basis(projection)
    displacements: list[np.ndarray] = []
    for anchor, attack in zip(anchors, attacks, strict=True):
        if attack is None:
            continue
        displacement = model.hidden(attack["latent"]) - model.hidden(anchor)
        if hidden_nullspace.size:
            displacement = hidden_nullspace @ (hidden_nullspace.T @ displacement)
        if np.linalg.norm(displacement) > 1e-12:
            displacements.append(displacement)
    if not displacements:
        return np.zeros((model.hidden_dimension, 0), dtype=float)
    return np.stack(displacements, axis=1)


def dangerous_direction_spectrum(displacements: np.ndarray) -> dict[str, object]:
    """Return singular energy, stable rank, entropy rank, and capture curve."""

    matrix = np.asarray(displacements, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("displacements must be a matrix")
    if matrix.shape[1] == 0 or np.linalg.norm(matrix) == 0.0:
        return {
            "sample_count": int(matrix.shape[1]),
            "singular_values": [],
            "normalized_energy": [],
            "cumulative_energy": [],
            "stable_rank": 0.0,
            "entropy_effective_rank": 0.0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
        }
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energy = singular_values * singular_values
    probabilities = energy / energy.sum()
    cumulative = np.cumsum(probabilities)
    entropy = -float(
        sum(probability * log(probability) for probability in probabilities if probability > 0.0)
    )

    def capture_rank(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold, side="left") + 1)

    return {
        "sample_count": int(matrix.shape[1]),
        "singular_values": [float(value) for value in singular_values],
        "normalized_energy": [float(value) for value in probabilities],
        "cumulative_energy": [float(value) for value in cumulative],
        "stable_rank": float(energy.sum() / energy.max()),
        "entropy_effective_rank": float(exp(entropy)),
        "rank_90": capture_rank(0.90),
        "rank_95": capture_rank(0.95),
        "rank_99": capture_rank(0.99),
    }


def residual_dangerous_energy(
    displacements: np.ndarray,
    projection: np.ndarray,
) -> float:
    matrix = np.asarray(displacements, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("displacements must be a matrix")
    denominator = float(np.sum(matrix * matrix))
    if denominator == 0.0:
        return 0.0
    rowspace = _rowspace_basis(np.asarray(projection, dtype=float))
    residual = matrix
    if rowspace.shape[1]:
        residual = matrix - rowspace @ (rowspace.T @ matrix)
    return float(np.sum(residual * residual) / denominator)


def _mean_geometry(model: Any, anchors: np.ndarray, projection: np.ndarray) -> tuple[float, float]:
    geometries = []
    for anchor in anchors:
        hidden_jacobian = model.hidden_jacobian(anchor)
        geometries.append(
            tunnel_geometry(
                projection @ hidden_jacobian,
                model.main_head @ hidden_jacobian,
            )
        )
    return (
        float(np.mean([item.openness for item in geometries])),
        float(np.mean([item.tunnel_dimension for item in geometries])),
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                int(row["visible_rank"]),
                int(row["closure_rank"]),
                str(row["intervention"]),
            )
            for row in rows
        }
    )
    for visible_rank, closure_rank, intervention in keys:
        cell = [
            row
            for row in rows
            if row["visible_rank"] == visible_rank
            and row["closure_rank"] == closure_rank
            and row["intervention"] == intervention
        ]
        packing = np.asarray([row["packing_fraction"] for row in cell], dtype=float)
        output.append(
            {
                "visible_rank": visible_rank,
                "closure_rank": closure_rank,
                "intervention": intervention,
                "run_count": len(cell),
                "mean_packing_fraction": float(packing.mean()),
                "packing_standard_error": float(packing.std(ddof=1) / sqrt(len(packing)))
                if len(packing) > 1
                else 0.0,
                "mean_candidate_fraction": float(
                    np.mean([row["candidate_fraction"] for row in cell])
                ),
                "mean_residual_dangerous_energy": float(
                    np.mean([row["residual_dangerous_energy"] for row in cell])
                ),
                "mean_openness": float(np.mean([row["mean_openness"] for row in cell])),
                "mean_tunnel_dimension": float(
                    np.mean([row["mean_tunnel_dimension"] for row in cell])
                ),
                "mean_effective_added_rank": float(
                    np.mean([row["effective_added_rank"] for row in cell])
                ),
                "mean_benign_acceptance": float(
                    np.mean([row["benign_acceptance"] for row in cell])
                ),
            }
        )
    return output


def _minimum_closure_ranks(
    rows: Sequence[Mapping[str, Any]],
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    model_seeds = sorted({int(row["model_seed"]) for row in rows})
    visible_ranks = sorted({int(row["visible_rank"]) for row in rows})
    for threshold in thresholds:
        for model_seed in model_seeds:
            for visible_rank in visible_ranks:
                cell = sorted(
                    (
                        row
                        for row in rows
                        if row["model_seed"] == model_seed
                        and row["visible_rank"] == visible_rank
                        and row["intervention"] == "targeted"
                    ),
                    key=lambda row: int(row["closure_rank"]),
                )
                successful = [
                    int(row["closure_rank"])
                    for row in cell
                    if float(row["packing_fraction"]) <= threshold
                ]
                output.append(
                    {
                        "threshold": threshold,
                        "model_seed": model_seed,
                        "visible_rank": visible_rank,
                        "minimum_targeted_closure_rank": min(successful)
                        if successful
                        else None,
                    }
                )
    return output


def _correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    residual = np.asarray([row["residual_dangerous_energy"] for row in rows], dtype=float)
    packing = np.asarray([row["packing_fraction"] for row in rows], dtype=float)
    openness = np.asarray([row["mean_openness"] for row in rows], dtype=float)

    def pearson(first: np.ndarray, second: np.ndarray) -> float | None:
        if first.std() == 0.0 or second.std() == 0.0:
            return None
        return float(np.corrcoef(first, second)[0, 1])

    return {
        "residual_energy_packing_pearson": pearson(residual, packing),
        "openness_packing_pearson": pearson(openness, packing),
    }


def run_closure_spectrum_campaign(config: Mapping[str, object]) -> dict[str, object]:
    """Run equal-rank targeted and control closure sweeps with adaptive attacks."""

    schema_version = positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("only schema_version=1 is supported")
    master_seed = positive_int(config.get("master_seed"), "master_seed")
    model_seeds = _model_seeds(config.get("model_seeds"))
    latent_dimension = positive_int(config.get("latent_dimension"), "latent_dimension")
    hidden_dimension = positive_int(config.get("hidden_dimension"), "hidden_dimension")
    visible_ranks = int_values(config.get("visible_ranks"), "visible_ranks")
    closure_ranks = _nonnegative_int_values(config.get("closure_ranks"), "closure_ranks")
    if 0 not in closure_ranks:
        raise ValueError("closure_ranks must include zero")
    if any(rank > latent_dimension for rank in visible_ranks):
        raise ValueError("visible_ranks must not exceed latent_dimension")
    integer_names = (
        "training_samples",
        "training_steps",
        "batch_size",
        "calibration_anchors",
        "evaluation_anchors",
        "support_samples",
        "benign_repetitions",
        "attack_steps",
    )
    integers = {name: positive_int(config.get(name), name) for name in integer_names}
    numeric_names = (
        "learning_rate",
        "minimum_latent_scale",
        "anchor_bound",
        "latent_bound",
        "benign_radius",
        "benign_quantile",
        "payload_delta",
        "behavior_gamma",
        "local_radius",
    )
    numbers = {name: number(config.get(name), name) for name in numeric_names}
    thresholds = tuple(
        number(item, f"packing_thresholds[{index}]")
        for index, item in enumerate(config.get("packing_thresholds", (0.05, 0.1)))
    )
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("packing thresholds must lie in [0,1]")

    training_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for model_seed in model_seeds:
        model, diagnostics = train_dual_head_model(
            seed=_seed(master_seed, "model", model_seed),
            latent_dimension=latent_dimension,
            hidden_dimension=hidden_dimension,
            training_samples=integers["training_samples"],
            training_steps=integers["training_steps"],
            batch_size=integers["batch_size"],
            learning_rate=numbers["learning_rate"],
            minimum_latent_scale=numbers["minimum_latent_scale"],
        )
        training_rows.append({"model_seed": model_seed, **diagnostics})
        rng = np.random.default_rng(_seed(master_seed, "anchors", model_seed))
        calibration_anchors = rng.uniform(
            -numbers["anchor_bound"],
            numbers["anchor_bound"],
            size=(integers["calibration_anchors"], latent_dimension),
        )
        evaluation_anchors = rng.uniform(
            -numbers["anchor_bound"],
            numbers["anchor_bound"],
            size=(integers["evaluation_anchors"], latent_dimension),
        )
        support_latent = rng.uniform(
            -1.0,
            1.0,
            size=(integers["support_samples"], latent_dimension),
        )
        hidden_support = model.hidden(support_latent)

        for visible_rank in visible_ranks:
            baseline_projection = model.control_head[:, :visible_rank].T
            calibration_standardized, calibration_epsilon, _ = _calibrate_epsilon(
                model,
                calibration_anchors,
                baseline_projection,
                hidden_support,
                seed=_seed(master_seed, "spectrum-calibration", model_seed, visible_rank),
                benign_radius=numbers["benign_radius"],
                benign_repetitions=integers["benign_repetitions"],
                benign_quantile=numbers["benign_quantile"],
            )
            calibration_attacks = _generate_attacks(
                model,
                calibration_anchors,
                baseline_projection,
                calibration_standardized,
                calibration_epsilon,
                payload_delta=numbers["payload_delta"],
                behavior_gamma=numbers["behavior_gamma"],
                local_radius=numbers["local_radius"],
                attack_steps=integers["attack_steps"],
                latent_bound=numbers["latent_bound"],
            )
            displacements = _dangerous_displacements(
                model,
                baseline_projection,
                calibration_anchors,
                calibration_attacks,
            )
            spectrum_rows.append(
                {
                    "model_seed": model_seed,
                    "visible_rank": visible_rank,
                    "baseline_candidate_fraction": float(
                        np.mean([item is not None for item in calibration_attacks])
                    ),
                    **dangerous_direction_spectrum(displacements),
                }
            )
            maximum_closure_rank = max(0, latent_dimension - visible_rank)
            admissible_closure_ranks = [
                rank for rank in closure_ranks if rank <= maximum_closure_rank
            ]
            for closure_rank in admissible_closure_ranks:
                if closure_rank == 0:
                    projections = {"baseline": baseline_projection}
                else:
                    all_projections = _interventions(
                        model,
                        baseline_projection,
                        calibration_anchors,
                        calibration_attacks,
                        hidden_support,
                        closure_rank=closure_rank,
                        seed=_seed(
                            master_seed,
                            "spectrum-intervention",
                            model_seed,
                            visible_rank,
                            closure_rank,
                        ),
                    )
                    projections = {
                        name: all_projections[name]
                        for name in (
                            "targeted",
                            "random_null",
                            "variance_null",
                            "row_sham",
                        )
                    }
                common_benign_seed = _seed(
                    master_seed,
                    "spectrum-benign",
                    model_seed,
                    visible_rank,
                    closure_rank,
                )
                baseline_effective_rank = _rowspace_basis(baseline_projection).shape[1]
                for intervention, projection in projections.items():
                    standardized, epsilon, benign_acceptance = _calibrate_epsilon(
                        model,
                        evaluation_anchors,
                        projection,
                        hidden_support,
                        seed=common_benign_seed,
                        benign_radius=numbers["benign_radius"],
                        benign_repetitions=integers["benign_repetitions"],
                        benign_quantile=numbers["benign_quantile"],
                    )
                    attacks = _generate_attacks(
                        model,
                        evaluation_anchors,
                        projection,
                        standardized,
                        epsilon,
                        payload_delta=numbers["payload_delta"],
                        behavior_gamma=numbers["behavior_gamma"],
                        local_radius=numbers["local_radius"],
                        attack_steps=integers["attack_steps"],
                        latent_bound=numbers["latent_bound"],
                    )
                    result = _evaluate_projection(
                        model,
                        evaluation_anchors,
                        attacks,
                        projection,
                        standardized,
                        epsilon,
                        benign_acceptance,
                        intervention=intervention,
                        attack_mode="adaptive",
                        payload_delta=numbers["payload_delta"],
                        behavior_gamma=numbers["behavior_gamma"],
                    )
                    mean_openness, mean_tunnel_dimension = _mean_geometry(
                        model,
                        evaluation_anchors,
                        projection,
                    )
                    effective_rank = _rowspace_basis(projection).shape[1]
                    rows.append(
                        {
                            "model_seed": model_seed,
                            "visible_rank": visible_rank,
                            "closure_rank": closure_rank,
                            "intervention": intervention,
                            "effective_added_rank": effective_rank - baseline_effective_rank,
                            "candidate_fraction": float(
                                np.mean([item is not None for item in attacks])
                            ),
                            "residual_dangerous_energy": residual_dangerous_energy(
                                displacements,
                                projection,
                            ),
                            "mean_openness": mean_openness,
                            "mean_tunnel_dimension": mean_tunnel_dimension,
                            **result,
                        }
                    )

    summary = _summary(rows)
    minimum_ranks = _minimum_closure_ranks(rows, thresholds)
    targeted_rows = [row for row in rows if row["intervention"] == "targeted"]
    controls = [
        row
        for row in rows
        if row["intervention"] in {"random_null", "variance_null"}
    ]
    targeted_is_energy_optimal = []
    for row in targeted_rows:
        peers = [
            peer
            for peer in controls
            if peer["model_seed"] == row["model_seed"]
            and peer["visible_rank"] == row["visible_rank"]
            and peer["closure_rank"] == row["closure_rank"]
        ]
        if peers:
            targeted_is_energy_optimal.append(
                float(row["residual_dangerous_energy"])
                <= min(float(peer["residual_dangerous_energy"]) for peer in peers) + 1e-12
            )
    return {
        "artifact_type": "qcollide_dangerous_direction_closure_spectrum",
        "schema_version": schema_version,
        "master_seed": master_seed,
        "claim_scope": {
            "trained_neural_network": True,
            "synthetic_task_only": True,
            "adaptive_reoptimization_included": True,
            "rank_dose_response_included": True,
            "spectral_optimality_is_linear_algebraic": True,
            "pretrained_model_claimed": False,
            "real_world_attack_claimed": False,
            "quantum_values_are": "analytic endpoint-query proxies",
            "coherent_quantum_execution": False,
            "new_lower_bound_claimed": False,
        },
        "config": dict(config),
        "training": training_rows,
        "spectra": spectrum_rows,
        "rows": rows,
        "summary": summary,
        "minimum_closure_ranks": minimum_ranks,
        "diagnostics": {
            **_correlations(rows),
            "targeted_energy_optimal_fraction": float(
                np.mean(targeted_is_energy_optimal)
            )
            if targeted_is_energy_optimal
            else None,
            "benign_acceptance_mean": float(
                np.mean([row["benign_acceptance"] for row in rows])
            ),
            "benign_acceptance_maximum_deviation": float(
                np.max(
                    np.abs(
                        np.asarray([row["benign_acceptance"] for row in rows])
                        - numbers["benign_quantile"]
                    )
                )
            ),
        },
    }


def compact_closure_spectrum_summary(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    summary = artifact["summary"]
    targeted = [row for row in summary if row["intervention"] == "targeted"]
    return {
        "artifact_type": artifact["artifact_type"],
        "schema_version": artifact["schema_version"],
        "master_seed": artifact["master_seed"],
        "claim_scope": artifact["claim_scope"],
        "diagnostics": artifact["diagnostics"],
        "spectra": artifact["spectra"],
        "targeted_summary": targeted,
        "minimum_closure_ranks": artifact["minimum_closure_ranks"],
    }


__all__ = [
    "compact_closure_spectrum_summary",
    "dangerous_direction_spectrum",
    "residual_dangerous_energy",
    "run_closure_spectrum_campaign",
]
