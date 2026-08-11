"""Prospective collision-targeted representation repair on real CIFAR models.

The intervention basis is learned only from a calibration collision graph. All
repair effects are evaluated on a disjoint evaluation collision graph under the
*pre-intervention* collision thresholds. This prevents evaluation-edge leakage
and prevents post-intervention threshold recalibration from hiding the effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from typing import Any

import numpy as np

from .cifar_models import derived_seed, load_cifar_data, train_cifar_model
from .cifar_topology import _device, _encode, _make_benign_images
from .topology_persistence import collision_filtration_profile
from .vision_linear import (
    control_output,
    nullspace_basis,
    residual_hidden,
    row_basis,
    standardized_projection,
    visible_control_projection,
)


def _correct_two_way_split(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select two disjoint correctly classified groups per class."""

    predictions = np.asarray(logits).argmax(axis=1)
    rng = np.random.default_rng(seed)
    first: list[int] = []
    second: list[int] = []
    for label in range(class_count):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) < 2 * per_class:
            raise RuntimeError(
                f"class {label} has {len(candidates)} correct examples; "
                f"{2 * per_class} are required for the prospective repair split"
            )
        candidates = candidates.copy()
        rng.shuffle(candidates)
        first.extend(int(value) for value in candidates[:per_class])
        second.extend(int(value) for value in candidates[per_class : 2 * per_class])
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def _pairwise(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    return np.linalg.norm(difference, axis=2)


def _logits_from_hidden(model, hidden: np.ndarray) -> np.ndarray:
    weight = model.main_head.weight.detach().cpu().numpy()
    bias_parameter = model.main_head.bias
    bias = (
        np.zeros(weight.shape[0], dtype=float)
        if bias_parameter is None
        else bias_parameter.detach().cpu().numpy()
    )
    return np.asarray(hidden, dtype=float) @ weight.T + bias


def _fixed_thresholds(
    *,
    projection: np.ndarray,
    standardized: np.ndarray,
    anchor_hidden: np.ndarray,
    anchor_logits: np.ndarray,
    benign_hidden: np.ndarray,
    benign_logits: np.ndarray,
    benign_anchor_indices: np.ndarray,
    epsilon_multipliers: Sequence[float],
    control_quantile: float,
    payload_quantile: float,
    behavior_quantile: float,
) -> dict[str, object]:
    anchor_control = control_output(anchor_hidden, standardized)
    benign_control = control_output(benign_hidden, standardized)
    benign_control_distance = np.linalg.norm(
        benign_control - anchor_control[benign_anchor_indices],
        axis=1,
    )
    nominal_epsilon = max(
        float(np.quantile(benign_control_distance, control_quantile)),
        1e-8,
    )
    thresholds = tuple(
        sorted(
            {
                max(1e-10, float(multiplier) * nominal_epsilon)
                for multiplier in epsilon_multipliers
            }
        )
    )

    anchor_residual = residual_hidden(anchor_hidden, projection)
    benign_residual = residual_hidden(benign_hidden, projection)
    benign_payload = np.linalg.norm(
        benign_residual - anchor_residual[benign_anchor_indices],
        axis=1,
    )
    payload_delta = float(np.quantile(benign_payload, payload_quantile))
    benign_behavior = np.linalg.norm(
        benign_logits - anchor_logits[benign_anchor_indices],
        axis=1,
    ) / sqrt(anchor_logits.shape[1])
    behavior_gamma = float(np.quantile(benign_behavior, behavior_quantile))
    return {
        "nominal_epsilon": nominal_epsilon,
        "control_thresholds": thresholds,
        "payload_delta": payload_delta,
        "behavior_gamma": behavior_gamma,
    }


def _profile(
    *,
    projection: np.ndarray,
    standardized: np.ndarray,
    anchor_hidden: np.ndarray,
    anchor_logits: np.ndarray,
    anchor_labels: np.ndarray,
    candidate_hidden: np.ndarray,
    candidate_logits: np.ndarray,
    candidate_labels: np.ndarray,
    thresholds: Mapping[str, object],
) -> tuple[dict[str, object], np.ndarray]:
    anchor_control = control_output(anchor_hidden, standardized)
    candidate_control = control_output(candidate_hidden, standardized)
    anchor_residual = residual_hidden(anchor_hidden, projection)
    candidate_residual = residual_hidden(candidate_hidden, projection)

    control_distances = _pairwise(candidate_control, anchor_control)
    payload_distances = _pairwise(candidate_residual, anchor_residual)
    behavior_distances = _pairwise(candidate_logits, anchor_logits) / sqrt(
        anchor_logits.shape[1]
    )
    candidate_predictions = candidate_logits.argmax(axis=1)
    anchor_predictions = anchor_logits.argmax(axis=1)
    valid_pairs = (
        (candidate_labels[:, None] != anchor_labels[None, :])
        & (candidate_predictions[:, None] != anchor_predictions[None, :])
    )
    edge_displacements = candidate_residual[:, None, :] - anchor_residual[None, :, :]
    profile = collision_filtration_profile(
        control_distances,
        payload_distances,
        behavior_distances,
        thresholds["control_thresholds"],
        nominal_epsilon=float(thresholds["nominal_epsilon"]),
        payload_delta=float(thresholds["payload_delta"]),
        behavior_gamma=float(thresholds["behavior_gamma"]),
        valid_pairs=valid_pairs,
        edge_displacements=edge_displacements,
    )
    nominal_index = int(
        np.argmin(
            np.abs(
                np.asarray(
                    [point.control_epsilon for point in profile.points],
                    dtype=float,
                )
                - float(thresholds["nominal_epsilon"])
            )
        )
    )
    nominal_point = profile.points[nominal_index]
    collision_mask = (
        valid_pairs
        & (control_distances <= float(thresholds["nominal_epsilon"]))
        & (payload_distances >= float(thresholds["payload_delta"]))
        & (behavior_distances >= float(thresholds["behavior_gamma"]))
    )
    return (
        {
            "nominal": asdict(nominal_point.metrics),
            "summary": asdict(profile.summary),
            "persistence": asdict(profile.basin_persistence),
        },
        collision_mask,
    )


def _dangerous_basis(
    *,
    projection: np.ndarray,
    anchor_hidden: np.ndarray,
    candidate_hidden: np.ndarray,
    collision_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    anchor_residual = residual_hidden(anchor_hidden, projection)
    candidate_residual = residual_hidden(candidate_hidden, projection)
    left, right = np.nonzero(collision_mask)
    if len(left) == 0:
        hidden_dimension = anchor_hidden.shape[1]
        return np.zeros((hidden_dimension, 0), dtype=float), {
            "collision_edge_count": 0,
            "singular_values": [],
            "rank_90": 0,
            "rank_95": 0,
        }
    displacements = candidate_residual[left] - anchor_residual[right]
    _, singular_values, vh = np.linalg.svd(displacements, full_matrices=False)
    energy = singular_values * singular_values
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-30)

    def rank_at(level: float) -> int:
        return int(np.searchsorted(cumulative, level, side="left") + 1)

    return vh.T.copy(), {
        "collision_edge_count": int(len(left)),
        "singular_values": [float(value) for value in singular_values],
        "rank_90": rank_at(0.90),
        "rank_95": rank_at(0.95),
    }


def _support_removed_energy(
    hidden_support: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    rank: int,
) -> float:
    selected = basis[:, : min(rank, basis.shape[1])]
    if selected.shape[1] == 0:
        return 0.0
    coordinates = (hidden_support - center) @ selected
    return float(np.mean(np.sum(coordinates * coordinates, axis=1)))


def _apply_closure(
    hidden: np.ndarray,
    *,
    center: np.ndarray,
    basis: np.ndarray,
    rank: int,
) -> np.ndarray:
    selected = basis[:, : min(rank, basis.shape[1])]
    if selected.shape[1] == 0:
        return np.asarray(hidden, dtype=float).copy()
    centered = np.asarray(hidden, dtype=float) - center
    return np.asarray(hidden, dtype=float) - (centered @ selected) @ selected.T


def _variance_null_basis(
    projection: np.ndarray,
    hidden_support: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    nullspace = nullspace_basis(projection)
    if nullspace.shape[1] == 0:
        return np.zeros((hidden_support.shape[1], 0), dtype=float)
    centered = hidden_support - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    restricted = nullspace.T @ covariance @ nullspace
    eigenvalues, eigenvectors = np.linalg.eigh(restricted)
    order = np.argsort(eigenvalues)[::-1]
    return nullspace @ eigenvectors[:, order]


def _energy_matched_random_basis(
    *,
    projection: np.ndarray,
    hidden_support: np.ndarray,
    center: np.ndarray,
    target_basis: np.ndarray,
    rank: int,
    candidates: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    nullspace = nullspace_basis(projection)
    effective_rank = min(rank, target_basis.shape[1], nullspace.shape[1])
    if effective_rank == 0:
        return np.zeros((hidden_support.shape[1], 0), dtype=float), {
            "target_removed_energy": 0.0,
            "random_removed_energy": 0.0,
            "relative_energy_mismatch": 0.0,
        }
    target_energy = _support_removed_energy(
        hidden_support,
        center,
        target_basis,
        effective_rank,
    )
    rng = np.random.default_rng(seed)
    best_basis: np.ndarray | None = None
    best_energy = 0.0
    best_mismatch = float("inf")
    for _ in range(candidates):
        raw = nullspace @ rng.normal(size=(nullspace.shape[1], effective_rank))
        basis = row_basis(raw.T).T
        if basis.shape[1] < effective_rank:
            continue
        energy = _support_removed_energy(
            hidden_support,
            center,
            basis,
            effective_rank,
        )
        mismatch = abs(energy - target_energy) / max(target_energy, 1e-12)
        if mismatch < best_mismatch:
            best_basis = basis
            best_energy = energy
            best_mismatch = mismatch
    if best_basis is None:
        raise RuntimeError("failed to construct an energy-matched random null basis")
    return best_basis, {
        "target_removed_energy": target_energy,
        "random_removed_energy": best_energy,
        "relative_energy_mismatch": best_mismatch,
    }


def _intervention_result(
    *,
    name: str,
    basis: np.ndarray,
    closure_rank: int,
    center: np.ndarray,
    model,
    projection: np.ndarray,
    standardized: np.ndarray,
    thresholds: Mapping[str, object],
    anchor_hidden: np.ndarray,
    anchor_labels: np.ndarray,
    candidate_hidden: np.ndarray,
    candidate_labels: np.ndarray,
    full_evaluation_hidden: np.ndarray,
    full_evaluation_labels: np.ndarray,
    baseline_profile: Mapping[str, object],
    baseline_accuracy: float,
) -> dict[str, object]:
    modified_anchor = _apply_closure(
        anchor_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
    )
    modified_candidate = _apply_closure(
        candidate_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
    )
    modified_evaluation = _apply_closure(
        full_evaluation_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
    )
    anchor_logits = _logits_from_hidden(model, modified_anchor)
    candidate_logits = _logits_from_hidden(model, modified_candidate)
    evaluation_logits = _logits_from_hidden(model, modified_evaluation)
    evaluation_accuracy = float(
        np.mean(evaluation_logits.argmax(axis=1) == full_evaluation_labels)
    )
    profile, _ = _profile(
        projection=projection,
        standardized=standardized,
        anchor_hidden=modified_anchor,
        anchor_logits=anchor_logits,
        anchor_labels=anchor_labels,
        candidate_hidden=modified_candidate,
        candidate_logits=candidate_logits,
        candidate_labels=candidate_labels,
        thresholds=thresholds,
    )
    baseline_capacity = float(baseline_profile["nominal"]["capacity_fraction"])
    capacity = float(profile["nominal"]["capacity_fraction"])
    baseline_auc = float(baseline_profile["summary"]["capacity_auc"])
    capacity_auc = float(profile["summary"]["capacity_auc"])
    baseline_control = control_output(full_evaluation_hidden, standardized)
    modified_control = control_output(modified_evaluation, standardized)
    control_drift = float(
        np.sqrt(np.mean((modified_control - baseline_control) ** 2))
    )
    return {
        "intervention": name,
        "closure_rank": closure_rank,
        "effective_rank": min(closure_rank, basis.shape[1]),
        "evaluation_accuracy": evaluation_accuracy,
        "accuracy_change": evaluation_accuracy - baseline_accuracy,
        "accuracy_loss": max(0.0, baseline_accuracy - evaluation_accuracy),
        "capacity_fraction": capacity,
        "capacity_reduction": baseline_capacity - capacity,
        "capacity_auc": capacity_auc,
        "capacity_auc_reduction": baseline_auc - capacity_auc,
        "basin_density": float(profile["nominal"]["beta0_active"])
        / max(
            1,
            min(
                int(profile["nominal"]["n_left"]),
                int(profile["nominal"]["n_right"]),
            ),
        ),
        "control_drift_rms": control_drift,
        "profile": profile,
    }


def run_cifar_repair_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Run one prospective calibration-to-evaluation repair experiment."""

    allowed_architectures = tuple(str(value) for value in config["architectures"])
    allowed_seeds = tuple(int(value) for value in config["model_seeds"])
    if architecture not in allowed_architectures:
        raise ValueError(f"architecture {architecture!r} is not configured")
    if model_seed not in allowed_seeds:
        raise ValueError(f"model_seed={model_seed} is not configured")

    master_seed = int(config["master_seed"])
    visible_rank = int(config["visible_rank"])
    closure_ranks = tuple(int(value) for value in config["closure_ranks"])
    per_class = int(config["repair_examples_per_class_per_side"])
    data = load_cifar_data(
        dataset=str(config["dataset"]),
        dataset_seed=int(config["dataset_seed"]),
        data_root=str(config["data_root"]),
        train_samples=int(config["train_samples"]),
        calibration_samples=int(config["calibration_samples"]),
        evaluation_samples=int(config["evaluation_samples"]),
        control_grid_size=int(config["control_grid_size"]),
    )
    training_seed = derived_seed(master_seed, architecture, model_seed, "training")
    model, training, _ = train_cifar_model(
        architecture,
        data,
        seed=training_seed,
        hidden_dimension=int(config["hidden_dimension"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        control_loss_weight=float(config["control_loss_weight"]),
        control_grid_size=int(config["control_grid_size"]),
        checkpoint_epochs=(),
        device_name=str(config.get("device", "auto")),
    )
    device = _device(str(config.get("device", "auto")))
    model.to(device)
    model.eval()

    calibration_hidden, calibration_logits = _encode(
        model,
        data.calibration_images,
        device=device,
    )
    evaluation_hidden, evaluation_logits = _encode(
        model,
        data.evaluation_images,
        device=device,
    )
    baseline_accuracy = float(
        np.mean(evaluation_logits.argmax(axis=1) == data.evaluation_labels)
    )
    calibration_anchor_indices, calibration_candidate_indices = _correct_two_way_split(
        calibration_logits,
        data.calibration_labels,
        class_count=data.class_count,
        per_class=per_class,
        seed=derived_seed(master_seed, architecture, model_seed, "repair-design-split"),
    )
    evaluation_anchor_indices, evaluation_candidate_indices = _correct_two_way_split(
        evaluation_logits,
        data.evaluation_labels,
        class_count=data.class_count,
        per_class=per_class,
        seed=derived_seed(master_seed, architecture, model_seed, "repair-heldout-split"),
    )

    support_count = int(config["support_samples"])
    support_rng = np.random.default_rng(
        derived_seed(master_seed, architecture, model_seed, "repair-support")
    )
    support_indices = np.sort(
        support_rng.choice(len(data.train_images), size=support_count, replace=False)
    )
    support_hidden, _ = _encode(
        model,
        data.train_images[support_indices],
        device=device,
    )
    center = support_hidden.mean(axis=0)
    projection = visible_control_projection(model, visible_rank)
    standardized = standardized_projection(projection, support_hidden)

    design_anchor_hidden = calibration_hidden[calibration_anchor_indices]
    design_anchor_logits = calibration_logits[calibration_anchor_indices]
    design_anchor_labels = data.calibration_labels[calibration_anchor_indices]
    design_candidate_hidden = calibration_hidden[calibration_candidate_indices]
    design_candidate_logits = calibration_logits[calibration_candidate_indices]
    design_candidate_labels = data.calibration_labels[calibration_candidate_indices]

    benign_images, benign_anchor_indices = _make_benign_images(
        data.calibration_images[calibration_anchor_indices],
        seed=derived_seed(master_seed, architecture, model_seed, "repair-benign"),
        repetitions=int(config["benign_repetitions"]),
        sigma=float(config["benign_sigma"]),
    )
    benign_hidden, benign_logits = _encode(model, benign_images, device=device)
    thresholds = _fixed_thresholds(
        projection=projection,
        standardized=standardized,
        anchor_hidden=design_anchor_hidden,
        anchor_logits=design_anchor_logits,
        benign_hidden=benign_hidden,
        benign_logits=benign_logits,
        benign_anchor_indices=benign_anchor_indices,
        epsilon_multipliers=tuple(float(value) for value in config["epsilon_multipliers"]),
        control_quantile=float(config["control_quantile"]),
        payload_quantile=float(config["payload_quantile"]),
        behavior_quantile=float(config["behavior_quantile"]),
    )
    design_profile, design_mask = _profile(
        projection=projection,
        standardized=standardized,
        anchor_hidden=design_anchor_hidden,
        anchor_logits=design_anchor_logits,
        anchor_labels=design_anchor_labels,
        candidate_hidden=design_candidate_hidden,
        candidate_logits=design_candidate_logits,
        candidate_labels=design_candidate_labels,
        thresholds=thresholds,
    )
    targeted_basis, dangerous_spectrum = _dangerous_basis(
        projection=projection,
        anchor_hidden=design_anchor_hidden,
        candidate_hidden=design_candidate_hidden,
        collision_mask=design_mask,
    )

    heldout_anchor_hidden = evaluation_hidden[evaluation_anchor_indices]
    heldout_anchor_logits = evaluation_logits[evaluation_anchor_indices]
    heldout_anchor_labels = data.evaluation_labels[evaluation_anchor_indices]
    heldout_candidate_hidden = evaluation_hidden[evaluation_candidate_indices]
    heldout_candidate_logits = evaluation_logits[evaluation_candidate_indices]
    heldout_candidate_labels = data.evaluation_labels[evaluation_candidate_indices]
    heldout_baseline, _ = _profile(
        projection=projection,
        standardized=standardized,
        anchor_hidden=heldout_anchor_hidden,
        anchor_logits=heldout_anchor_logits,
        anchor_labels=heldout_anchor_labels,
        candidate_hidden=heldout_candidate_hidden,
        candidate_logits=heldout_candidate_logits,
        candidate_labels=heldout_candidate_labels,
        thresholds=thresholds,
    )
    baseline_capacity = float(heldout_baseline["nominal"]["capacity_fraction"])
    if baseline_capacity < float(config["minimum_baseline_capacity_fraction"]):
        raise RuntimeError(
            "held-out baseline capacity is below the preregistered repair floor: "
            f"{baseline_capacity:.6f}"
        )
    if targeted_basis.shape[1] < max(closure_ranks):
        raise RuntimeError(
            "calibration collision spectrum has insufficient rank for the registered repair sweep"
        )

    variance_basis = _variance_null_basis(projection, support_hidden, center)
    rows: list[dict[str, object]] = []
    energy_matches: list[dict[str, object]] = []
    for closure_rank in closure_ranks:
        random_basis, match = _energy_matched_random_basis(
            projection=projection,
            hidden_support=support_hidden,
            center=center,
            target_basis=targeted_basis,
            rank=closure_rank,
            candidates=int(config["random_basis_candidates"]),
            seed=derived_seed(
                master_seed,
                architecture,
                model_seed,
                "repair-random-basis",
                closure_rank,
            ),
        )
        energy_matches.append({"closure_rank": closure_rank, **match})
        bases = {
            "targeted": targeted_basis,
            "random_energy_matched": random_basis,
            "variance_null": variance_basis,
        }
        for name, basis in bases.items():
            result = _intervention_result(
                name=name,
                basis=basis,
                closure_rank=closure_rank,
                center=center,
                model=model,
                projection=projection,
                standardized=standardized,
                thresholds=thresholds,
                anchor_hidden=heldout_anchor_hidden,
                anchor_labels=heldout_anchor_labels,
                candidate_hidden=heldout_candidate_hidden,
                candidate_labels=heldout_candidate_labels,
                full_evaluation_hidden=evaluation_hidden,
                full_evaluation_labels=data.evaluation_labels,
                baseline_profile=heldout_baseline,
                baseline_accuracy=baseline_accuracy,
            )
            result["removed_support_energy"] = _support_removed_energy(
                support_hidden,
                center,
                basis,
                closure_rank,
            )
            result["accuracy_within_budget"] = bool(
                float(result["accuracy_loss"]) <= float(config["maximum_accuracy_loss"])
            )
            rows.append(result)

    return {
        "artifact_type": "qcollide_cifar_prospective_repair_component",
        "schema_version": 1,
        "dataset": data.dataset_name,
        "architecture": architecture,
        "model_seed": model_seed,
        "visible_rank": visible_rank,
        "closure_ranks": list(closure_ranks),
        "training_seed": training_seed,
        "training": training,
        "selection": {
            "calibration_anchor_count": len(calibration_anchor_indices),
            "calibration_candidate_count": len(calibration_candidate_indices),
            "evaluation_anchor_count": len(evaluation_anchor_indices),
            "evaluation_candidate_count": len(evaluation_candidate_indices),
            "prospective_disjoint_design_and_evaluation": True,
        },
        "thresholds": {
            "nominal_epsilon": float(thresholds["nominal_epsilon"]),
            "control_thresholds": [float(value) for value in thresholds["control_thresholds"]],
            "payload_delta": float(thresholds["payload_delta"]),
            "behavior_gamma": float(thresholds["behavior_gamma"]),
            "post_intervention_recalibration": False,
        },
        "design_profile": design_profile,
        "dangerous_spectrum": dangerous_spectrum,
        "heldout_baseline_profile": heldout_baseline,
        "baseline_evaluation_accuracy": baseline_accuracy,
        "energy_matches": energy_matches,
        "rows": rows,
        "claim_boundary": {
            "targeted_basis_uses_evaluation_collisions": False,
            "thresholds_recalibrated_after_intervention": False,
            "interventions_modify_hidden_representation_not_model_weights": True,
            "causal_parameter_training_claimed": False,
            "heldout_representation_intervention_claimed": True,
        },
    }


__all__ = [
    "_apply_closure",
    "_correct_two_way_split",
    "_energy_matched_random_basis",
    "run_cifar_repair_component",
]
