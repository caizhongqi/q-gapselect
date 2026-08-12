"""Calibration-selected, exactly energy-matched CIFAR representation repair.

This v2 experiment fixes two problems exposed by the first prospective repair
run: the baseline collision operating point could be saturated, and a random
null-space projection could not match the energy removed by the targeted
projection. The operating point is selected from calibration data only. Repair
strength is soft-scaled so targeted, random-null, and variance-null controls
remove exactly the same support energy before held-out evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from typing import Any

import numpy as np

from .cifar_models import derived_seed, load_cifar_data, train_cifar_model
from .cifar_repair_utility import (
    _correct_two_way_split,
    _dangerous_basis,
    _fixed_thresholds,
    _logits_from_hidden,
    _profile,
    _support_removed_energy,
    _variance_null_basis,
)
from .cifar_topology import _device, _encode, _make_benign_images
from .vision_linear import (
    control_output,
    nullspace_basis,
    standardized_projection,
    visible_control_projection,
)


def soft_strength_for_energy(full_energy: float, budget: float) -> float:
    """Return alpha in h' = h - alpha P h for an exact perturbation budget."""

    full_energy = float(full_energy)
    budget = float(budget)
    if full_energy < 0.0 or budget < 0.0:
        raise ValueError("energies must be non-negative")
    if budget == 0.0:
        return 0.0
    if full_energy <= 0.0 or budget > full_energy * (1.0 + 1e-12):
        raise ValueError("requested budget is not attainable by this basis")
    return min(1.0, sqrt(budget / full_energy))


def apply_soft_closure(
    hidden: np.ndarray,
    *,
    center: np.ndarray,
    basis: np.ndarray,
    rank: int,
    strength: float,
) -> np.ndarray:
    """Shrink coordinates in a registered subspace without changing its span."""

    selected = np.asarray(basis, dtype=float)[:, : min(int(rank), basis.shape[1])]
    if selected.shape[1] == 0 or strength == 0.0:
        return np.asarray(hidden, dtype=float).copy()
    centered = np.asarray(hidden, dtype=float) - np.asarray(center, dtype=float)
    return np.asarray(hidden, dtype=float) - float(strength) * (
        (centered @ selected) @ selected.T
    )


def _actual_perturbation_energy(
    hidden: np.ndarray,
    modified: np.ndarray,
) -> float:
    difference = np.asarray(modified, dtype=float) - np.asarray(hidden, dtype=float)
    return float(np.mean(np.sum(difference * difference, axis=1)))


def _high_energy_random_null_basis(
    *,
    projection: np.ndarray,
    hidden_support: np.ndarray,
    center: np.ndarray,
    rank: int,
    candidates: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Choose a strong random null control using support energy only."""

    nullspace = nullspace_basis(projection)
    effective_rank = min(int(rank), nullspace.shape[1])
    if effective_rank <= 0:
        raise RuntimeError("control projection has no registered null space")
    rng = np.random.default_rng(seed)
    best_basis: np.ndarray | None = None
    best_energy = -1.0
    for _ in range(int(candidates)):
        coordinates = rng.normal(size=(nullspace.shape[1], effective_rank))
        q, _ = np.linalg.qr(coordinates)
        basis = nullspace @ q[:, :effective_rank]
        energy = _support_removed_energy(
            hidden_support,
            center,
            basis,
            effective_rank,
        )
        if energy > best_energy:
            best_basis = basis
            best_energy = energy
    if best_basis is None or best_energy <= 0.0:
        raise RuntimeError("failed to construct a positive-energy random null basis")
    return best_basis, float(best_energy)


def _select_operating_point(
    *,
    model,
    support_hidden: np.ndarray,
    design_anchor_hidden: np.ndarray,
    design_anchor_logits: np.ndarray,
    design_anchor_labels: np.ndarray,
    design_candidate_hidden: np.ndarray,
    design_candidate_logits: np.ndarray,
    design_candidate_labels: np.ndarray,
    benign_hidden: np.ndarray,
    benign_logits: np.ndarray,
    benign_anchor_indices: np.ndarray,
    visible_ranks: Sequence[int],
    epsilon_multipliers: Sequence[float],
    capacity_target: float,
    capacity_min: float,
    capacity_max: float,
    minimum_edges: int,
    minimum_basis_rank: int,
    control_quantile: float,
    payload_quantile: float,
    behavior_quantile: float,
) -> dict[str, Any]:
    """Select a non-saturated design operating point using calibration only."""

    candidates: list[dict[str, Any]] = []
    for visible_rank in tuple(int(value) for value in visible_ranks):
        projection = visible_control_projection(model, visible_rank)
        standardized = standardized_projection(projection, support_hidden)
        base_thresholds = _fixed_thresholds(
            projection=projection,
            standardized=standardized,
            anchor_hidden=design_anchor_hidden,
            anchor_logits=design_anchor_logits,
            benign_hidden=benign_hidden,
            benign_logits=benign_logits,
            benign_anchor_indices=benign_anchor_indices,
            epsilon_multipliers=tuple(float(value) for value in epsilon_multipliers),
            control_quantile=float(control_quantile),
            payload_quantile=float(payload_quantile),
            behavior_quantile=float(behavior_quantile),
        )
        base_epsilon = float(base_thresholds["nominal_epsilon"])
        for multiplier in tuple(float(value) for value in epsilon_multipliers):
            chosen_epsilon = max(1e-10, multiplier * base_epsilon)
            thresholds = {
                **base_thresholds,
                "nominal_epsilon": chosen_epsilon,
            }
            profile, collision_mask = _profile(
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
            capacity = float(profile["nominal"]["capacity_fraction"])
            edge_count = int(np.count_nonzero(collision_mask))
            if not capacity_min <= capacity <= capacity_max:
                continue
            if edge_count < int(minimum_edges):
                continue
            targeted_basis, spectrum = _dangerous_basis(
                projection=projection,
                anchor_hidden=design_anchor_hidden,
                candidate_hidden=design_candidate_hidden,
                collision_mask=collision_mask,
            )
            if targeted_basis.shape[1] < int(minimum_basis_rank):
                continue
            candidates.append(
                {
                    "visible_rank": visible_rank,
                    "epsilon_multiplier": multiplier,
                    "projection": projection,
                    "standardized": standardized,
                    "thresholds": thresholds,
                    "design_profile": profile,
                    "design_mask": collision_mask,
                    "targeted_basis": targeted_basis,
                    "dangerous_spectrum": spectrum,
                    "capacity_fraction": capacity,
                    "collision_edge_count": edge_count,
                    "objective": abs(capacity - float(capacity_target)),
                }
            )
    if not candidates:
        raise RuntimeError("no calibration operating point satisfies the registered repair gates")
    candidates.sort(
        key=lambda row: (
            float(row["objective"]),
            int(row["visible_rank"]),
            abs(float(row["epsilon_multiplier"]) - 1.0),
        )
    )
    selected = candidates[0]
    return {
        **selected,
        "candidate_count": len(candidates),
    }


def _soft_intervention_result(
    *,
    name: str,
    basis: np.ndarray,
    closure_rank: int,
    strength: float,
    energy_budget: float,
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
    support_hidden: np.ndarray,
    baseline_profile: Mapping[str, object],
    baseline_accuracy: float,
) -> dict[str, object]:
    modified_anchor = apply_soft_closure(
        anchor_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
        strength=strength,
    )
    modified_candidate = apply_soft_closure(
        candidate_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
        strength=strength,
    )
    modified_evaluation = apply_soft_closure(
        full_evaluation_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
        strength=strength,
    )
    modified_support = apply_soft_closure(
        support_hidden,
        center=center,
        basis=basis,
        rank=closure_rank,
        strength=strength,
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
    actual_energy = _actual_perturbation_energy(support_hidden, modified_support)
    relative_energy_mismatch = abs(actual_energy - energy_budget) / max(
        float(energy_budget),
        1e-12,
    )
    baseline_control = control_output(full_evaluation_hidden, standardized)
    modified_control = control_output(modified_evaluation, standardized)
    return {
        "intervention": name,
        "closure_rank": int(closure_rank),
        "strength": float(strength),
        "energy_budget": float(energy_budget),
        "actual_removed_support_energy": actual_energy,
        "relative_energy_mismatch": relative_energy_mismatch,
        "evaluation_accuracy": evaluation_accuracy,
        "accuracy_change": evaluation_accuracy - baseline_accuracy,
        "accuracy_loss": max(0.0, baseline_accuracy - evaluation_accuracy),
        "capacity_fraction": capacity,
        "capacity_reduction": baseline_capacity - capacity,
        "capacity_auc": capacity_auc,
        "capacity_auc_reduction": baseline_auc - capacity_auc,
        "control_drift_rms": float(
            np.sqrt(np.mean((modified_control - baseline_control) ** 2))
        ),
        "profile": profile,
    }


def run_cifar_calibrated_repair_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Run one calibration-selected, exactly energy-matched repair component."""

    if architecture not in tuple(str(value) for value in config["architectures"]):
        raise ValueError(f"architecture {architecture!r} is not configured")
    if model_seed not in tuple(int(value) for value in config["model_seeds"]):
        raise ValueError(f"model_seed={model_seed} is not configured")

    master_seed = int(config["master_seed"])
    closure_ranks = tuple(int(value) for value in config["closure_ranks"])
    energy_fractions = tuple(float(value) for value in config["energy_budget_fractions"])
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
        seed=derived_seed(master_seed, architecture, model_seed, "repair-v2-design-split"),
    )
    evaluation_anchor_indices, evaluation_candidate_indices = _correct_two_way_split(
        evaluation_logits,
        data.evaluation_labels,
        class_count=data.class_count,
        per_class=per_class,
        seed=derived_seed(master_seed, architecture, model_seed, "repair-v2-heldout-split"),
    )
    support_rng = np.random.default_rng(
        derived_seed(master_seed, architecture, model_seed, "repair-v2-support")
    )
    support_indices = np.sort(
        support_rng.choice(
            len(data.train_images),
            size=int(config["support_samples"]),
            replace=False,
        )
    )
    support_hidden, _ = _encode(
        model,
        data.train_images[support_indices],
        device=device,
    )
    center = support_hidden.mean(axis=0)

    design_anchor_hidden = calibration_hidden[calibration_anchor_indices]
    design_anchor_logits = calibration_logits[calibration_anchor_indices]
    design_anchor_labels = data.calibration_labels[calibration_anchor_indices]
    design_candidate_hidden = calibration_hidden[calibration_candidate_indices]
    design_candidate_logits = calibration_logits[calibration_candidate_indices]
    design_candidate_labels = data.calibration_labels[calibration_candidate_indices]
    benign_images, benign_anchor_indices = _make_benign_images(
        data.calibration_images[calibration_anchor_indices],
        seed=derived_seed(master_seed, architecture, model_seed, "repair-v2-benign"),
        repetitions=int(config["benign_repetitions"]),
        sigma=float(config["benign_sigma"]),
    )
    benign_hidden, benign_logits = _encode(model, benign_images, device=device)
    operating = _select_operating_point(
        model=model,
        support_hidden=support_hidden,
        design_anchor_hidden=design_anchor_hidden,
        design_anchor_logits=design_anchor_logits,
        design_anchor_labels=design_anchor_labels,
        design_candidate_hidden=design_candidate_hidden,
        design_candidate_logits=design_candidate_logits,
        design_candidate_labels=design_candidate_labels,
        benign_hidden=benign_hidden,
        benign_logits=benign_logits,
        benign_anchor_indices=benign_anchor_indices,
        visible_ranks=tuple(int(value) for value in config["visible_ranks"]),
        epsilon_multipliers=tuple(float(value) for value in config["epsilon_multipliers"]),
        capacity_target=float(config["operating_capacity_target"]),
        capacity_min=float(config["operating_capacity_min"]),
        capacity_max=float(config["operating_capacity_max"]),
        minimum_edges=int(config["minimum_design_collision_edges"]),
        minimum_basis_rank=max(closure_ranks),
        control_quantile=float(config["control_quantile"]),
        payload_quantile=float(config["payload_quantile"]),
        behavior_quantile=float(config["behavior_quantile"]),
    )
    projection = np.asarray(operating["projection"], dtype=float)
    standardized = np.asarray(operating["standardized"], dtype=float)
    thresholds = operating["thresholds"]
    targeted_basis = np.asarray(operating["targeted_basis"], dtype=float)

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
    heldout_capacity = float(heldout_baseline["nominal"]["capacity_fraction"])
    if not float(config["heldout_capacity_min"]) <= heldout_capacity <= float(
        config["heldout_capacity_max"]
    ):
        raise RuntimeError(
            "held-out baseline capacity is outside the preregistered nonsaturation gate: "
            f"{heldout_capacity:.6f}"
        )

    variance_basis = _variance_null_basis(projection, support_hidden, center)
    rows: list[dict[str, object]] = []
    energy_plans: list[dict[str, object]] = []
    for closure_rank in closure_ranks:
        random_basis, random_full_energy = _high_energy_random_null_basis(
            projection=projection,
            hidden_support=support_hidden,
            center=center,
            rank=closure_rank,
            candidates=int(config["random_basis_candidates"]),
            seed=derived_seed(
                master_seed,
                architecture,
                model_seed,
                "repair-v2-random-basis",
                closure_rank,
            ),
        )
        target_full_energy = _support_removed_energy(
            support_hidden,
            center,
            targeted_basis,
            closure_rank,
        )
        variance_full_energy = _support_removed_energy(
            support_hidden,
            center,
            variance_basis,
            closure_rank,
        )
        common_max = min(target_full_energy, random_full_energy, variance_full_energy)
        if common_max <= 0.0:
            raise RuntimeError("registered repair bases have zero common energy support")
        for energy_fraction in energy_fractions:
            energy_budget = float(energy_fraction) * common_max
            plan = {
                "closure_rank": closure_rank,
                "energy_budget_fraction": energy_fraction,
                "common_max_energy": common_max,
                "energy_budget": energy_budget,
                "target_full_energy": target_full_energy,
                "random_full_energy": random_full_energy,
                "variance_full_energy": variance_full_energy,
            }
            energy_plans.append(plan)
            for name, basis, full_energy in (
                ("targeted_soft", targeted_basis, target_full_energy),
                ("random_high_energy_soft", random_basis, random_full_energy),
                ("variance_soft", variance_basis, variance_full_energy),
            ):
                strength = soft_strength_for_energy(full_energy, energy_budget)
                result = _soft_intervention_result(
                    name=name,
                    basis=basis,
                    closure_rank=closure_rank,
                    strength=strength,
                    energy_budget=energy_budget,
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
                    support_hidden=support_hidden,
                    baseline_profile=heldout_baseline,
                    baseline_accuracy=baseline_accuracy,
                )
                result["energy_budget_fraction"] = energy_fraction
                result["accuracy_within_budget"] = bool(
                    float(result["accuracy_loss"]) <= float(config["maximum_accuracy_loss"])
                )
                result["energy_match_within_tolerance"] = bool(
                    float(result["relative_energy_mismatch"])
                    <= float(config["maximum_energy_mismatch"])
                )
                rows.append(result)

    return {
        "artifact_type": "qcollide_cifar_calibrated_repair_component",
        "schema_version": 2,
        "dataset": data.dataset_name,
        "architecture": architecture,
        "model_seed": model_seed,
        "training_seed": training_seed,
        "training": training,
        "baseline_evaluation_accuracy": baseline_accuracy,
        "operating_point": {
            "visible_rank": int(operating["visible_rank"]),
            "epsilon_multiplier": float(operating["epsilon_multiplier"]),
            "design_capacity_fraction": float(operating["capacity_fraction"]),
            "design_collision_edge_count": int(operating["collision_edge_count"]),
            "candidate_count": int(operating["candidate_count"]),
            "dangerous_spectrum": operating["dangerous_spectrum"],
            "thresholds": {
                "nominal_epsilon": float(thresholds["nominal_epsilon"]),
                "control_thresholds": [
                    float(value) for value in thresholds["control_thresholds"]
                ],
                "payload_delta": float(thresholds["payload_delta"]),
                "behavior_gamma": float(thresholds["behavior_gamma"]),
            },
        },
        "heldout_baseline_profile": heldout_baseline,
        "energy_plans": energy_plans,
        "rows": rows,
        "claim_boundary": {
            "operating_point_uses_evaluation_data": False,
            "targeted_basis_uses_evaluation_collisions": False,
            "energy_budget_uses_evaluation_data": False,
            "thresholds_recalibrated_after_intervention": False,
            "soft_intervention_changes_model_weights": False,
        },
    }


__all__ = [
    "apply_soft_closure",
    "run_cifar_calibrated_repair_component",
    "soft_strength_for_energy",
]
