"""Single-run CIFAR-100 topology with fixed broad-coverage correct pools.

The v4 requirement of one correct anchor from every fine class is too strict at
~30% accuracy. V5 keeps the same performance-matching target and checkpoint
selection, but replaces all-class stratification with fixed-size, deterministically
balanced pools and an explicit minimum unique-class coverage gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .cifar_models import derived_seed, load_cifar_data, train_cifar_model
from .cifar_single_run_selected import select_checkpoint_from_calibration_trajectory
from .cifar_topology import (
    _device,
    _encode,
    _float_tuple,
    _integer_tuple,
    _make_benign_images,
    _nonnegative_int,
    _number,
    _positive_int,
    _profile_row,
    _sequence,
)


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(logits).argmax(axis=1) == np.asarray(labels)))


def balanced_correct_pool(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    pool_size: int,
    minimum_unique_classes: int,
    per_class_cap: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select a fixed-size correct pool while maximizing fine-class coverage."""

    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits and labels must be aligned")
    if pool_size <= 0 or minimum_unique_classes <= 0 or per_class_cap <= 0:
        raise ValueError("pool parameters must be positive")
    predictions = logits.argmax(axis=1)
    correct = predictions == labels
    classes = sorted(int(value) for value in np.unique(labels[correct]))
    if len(classes) < minimum_unique_classes:
        raise RuntimeError(
            "correct-sample fine-class coverage is below the preregistered gate: "
            f"{len(classes)} < {minimum_unique_classes}"
        )
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[int]] = {}
    for label in classes:
        values = np.flatnonzero(correct & (labels == label)).copy()
        rng.shuffle(values)
        buckets[label] = [int(value) for value in values[:per_class_cap]]
    class_order = np.asarray(classes, dtype=np.int64)
    rng.shuffle(class_order)
    selected: list[int] = []
    for round_index in range(per_class_cap):
        for label_value in class_order:
            label = int(label_value)
            bucket = buckets[label]
            if round_index < len(bucket):
                selected.append(bucket[round_index])
                if len(selected) == pool_size:
                    break
        if len(selected) == pool_size:
            break
    if len(selected) < pool_size:
        raise RuntimeError(
            "insufficient correct samples under the preregistered per-class cap: "
            f"selected {len(selected)} < {pool_size}"
        )
    indices = np.asarray(selected, dtype=np.int64)
    selected_labels = labels[indices]
    counts = {
        int(label): int(np.sum(selected_labels == label))
        for label in np.unique(selected_labels)
    }
    unique_count = len(counts)
    if unique_count < minimum_unique_classes:
        raise RuntimeError("selected pool failed its fine-class coverage gate")
    return indices, {
        "pool_size": int(pool_size),
        "unique_fine_classes": int(unique_count),
        "minimum_unique_fine_classes": int(minimum_unique_classes),
        "per_class_pool_cap": int(per_class_cap),
        "class_counts": counts,
        "available_correct_unique_classes": int(len(classes)),
        "correct_sample_count": int(np.sum(correct)),
    }


def run_cifar_single_run_broad_pool_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train once, select a matched checkpoint, and measure fixed-pool topology."""

    schema_version = _positive_int(config.get("schema_version"), "schema_version")
    master_seed = _positive_int(config.get("master_seed"), "master_seed")
    architectures = tuple(
        str(item) for item in _sequence(config.get("architectures"), "architectures")
    )
    if architecture not in architectures:
        raise ValueError(f"architecture {architecture!r} is not configured")
    model_seeds = _integer_tuple(config.get("model_seeds"), "model_seeds", allow_zero=True)
    if model_seed not in model_seeds:
        raise ValueError(f"model_seed={model_seed} is not configured")

    epochs = _positive_int(config.get("epochs"), "epochs")
    checkpoint_epochs = _integer_tuple(
        config.get("checkpoint_epochs"),
        "checkpoint_epochs",
        allow_zero=True,
    )
    if checkpoint_epochs[0] != 0 or checkpoint_epochs[-1] != epochs:
        raise ValueError("checkpoint_epochs must start at zero and end at epochs")
    visible_ranks = _integer_tuple(config.get("main_visible_ranks"), "main_visible_ranks")

    data = load_cifar_data(
        dataset=str(config.get("dataset")),
        dataset_seed=_nonnegative_int(config.get("dataset_seed"), "dataset_seed"),
        data_root=str(config.get("data_root")),
        train_samples=_positive_int(config.get("train_samples"), "train_samples"),
        calibration_samples=_positive_int(
            config.get("calibration_samples"),
            "calibration_samples",
        ),
        evaluation_samples=_positive_int(
            config.get("evaluation_samples"),
            "evaluation_samples",
        ),
        control_grid_size=_positive_int(config.get("control_grid_size"), "control_grid_size"),
    )
    training_seed = derived_seed(master_seed, architecture, model_seed, "training")
    model, training, checkpoints = train_cifar_model(
        architecture,
        data,
        seed=training_seed,
        hidden_dimension=_positive_int(config.get("hidden_dimension"), "hidden_dimension"),
        epochs=epochs,
        batch_size=_positive_int(config.get("batch_size"), "batch_size"),
        learning_rate=_number(config.get("learning_rate"), "learning_rate"),
        control_loss_weight=_number(config.get("control_loss_weight"), "control_loss_weight"),
        control_grid_size=_positive_int(config.get("control_grid_size"), "control_grid_size"),
        checkpoint_epochs=checkpoint_epochs,
        device_name=str(config.get("device", "auto")),
    )
    device = _device(str(config.get("device", "auto")))

    calibration_trajectory: dict[int, float] = {}
    for epoch in checkpoint_epochs:
        model.load_state_dict(checkpoints[epoch])
        model.to(device)
        model.eval()
        _, logits = _encode(model, data.calibration_images, device=device)
        calibration_trajectory[epoch] = _accuracy(logits, data.calibration_labels)

    target = _number(config.get("performance_match_target"), "performance_match_target")
    tolerance = _number(
        config.get("performance_match_tolerance"),
        "performance_match_tolerance",
    )
    selection = select_checkpoint_from_calibration_trajectory(
        calibration_trajectory,
        target_accuracy=target,
        maximum_accuracy_mismatch=tolerance,
    )
    selected_epoch = int(selection["checkpoint_epoch"])
    model.load_state_dict(checkpoints[selected_epoch])
    model.to(device)
    model.eval()

    calibration_hidden_all, calibration_logits_all = _encode(
        model,
        data.calibration_images,
        device=device,
    )
    evaluation_hidden_all, evaluation_logits_all = _encode(
        model,
        data.evaluation_images,
        device=device,
    )
    calibration_accuracy = _accuracy(calibration_logits_all, data.calibration_labels)
    evaluation_accuracy = _accuracy(evaluation_logits_all, data.evaluation_labels)
    if abs(calibration_accuracy - float(selection["calibration_accuracy"])) > 1e-12:
        raise RuntimeError("selected checkpoint accuracy changed inside the same run")

    minimum_classes = _positive_int(
        config.get("minimum_unique_fine_classes"),
        "minimum_unique_fine_classes",
    )
    per_class_cap = _positive_int(config.get("per_class_pool_cap"), "per_class_pool_cap")
    anchor_indices, anchor_coverage = balanced_correct_pool(
        calibration_logits_all,
        data.calibration_labels,
        pool_size=_positive_int(config.get("anchor_pool_size"), "anchor_pool_size"),
        minimum_unique_classes=minimum_classes,
        per_class_cap=per_class_cap,
        seed=derived_seed(master_seed, architecture, model_seed, "broad-anchor-pool"),
    )
    candidate_indices, candidate_coverage = balanced_correct_pool(
        evaluation_logits_all,
        data.evaluation_labels,
        pool_size=_positive_int(config.get("candidate_pool_size"), "candidate_pool_size"),
        minimum_unique_classes=minimum_classes,
        per_class_cap=per_class_cap,
        seed=derived_seed(master_seed, architecture, model_seed, "broad-candidate-pool"),
    )

    anchor_hidden = calibration_hidden_all[anchor_indices]
    anchor_logits = calibration_logits_all[anchor_indices]
    candidate_hidden = evaluation_hidden_all[candidate_indices]
    candidate_logits = evaluation_logits_all[candidate_indices]

    support_count = _positive_int(config.get("support_samples"), "support_samples")
    support_rng = np.random.default_rng(
        derived_seed(master_seed, architecture, model_seed, "support")
    )
    support_indices = np.sort(
        support_rng.choice(len(data.train_images), size=support_count, replace=False)
    )
    support_hidden, _ = _encode(model, data.train_images[support_indices], device=device)

    benign_images, benign_anchor_indices = _make_benign_images(
        data.calibration_images[anchor_indices],
        seed=derived_seed(master_seed, architecture, model_seed, "benign"),
        repetitions=_positive_int(config.get("benign_repetitions"), "benign_repetitions"),
        sigma=_number(config.get("benign_sigma"), "benign_sigma"),
    )
    benign_hidden, benign_logits = _encode(model, benign_images, device=device)

    epsilon_multipliers = _float_tuple(
        config.get("epsilon_multipliers"),
        "epsilon_multipliers",
        minimum=0.0,
    )
    quantiles = {
        "control_quantile": _number(config.get("control_quantile"), "control_quantile"),
        "payload_quantile": _number(config.get("payload_quantile"), "payload_quantile"),
        "behavior_quantile": _number(config.get("behavior_quantile"), "behavior_quantile"),
    }
    rows: list[dict[str, Any]] = []
    for visible_rank in visible_ranks:
        rows.append(
            _profile_row(
                model,
                architecture=architecture,
                model_seed=model_seed,
                checkpoint_epoch=selected_epoch,
                visible_rank=visible_rank,
                support_hidden=support_hidden,
                anchor_hidden=anchor_hidden,
                anchor_logits=anchor_logits,
                anchor_labels=data.calibration_labels[anchor_indices],
                candidate_hidden=candidate_hidden,
                candidate_logits=candidate_logits,
                candidate_labels=data.evaluation_labels[candidate_indices],
                benign_hidden=benign_hidden,
                benign_logits=benign_logits,
                benign_anchor_indices=benign_anchor_indices,
                epsilon_multipliers=epsilon_multipliers,
                control_quantile=quantiles["control_quantile"],
                payload_quantile=quantiles["payload_quantile"],
                behavior_quantile=quantiles["behavior_quantile"],
            )
        )

    return {
        "artifact_type": "qcollide_cifar_broad_pool_selected_component",
        "schema_version": schema_version,
        "selection_schema_version": 3,
        "architecture": architecture,
        "model_seed": model_seed,
        "training_seed": training_seed,
        "selected_checkpoint_epoch": selected_epoch,
        "selected_performance": {
            "calibration_accuracy": calibration_accuracy,
            "evaluation_accuracy": evaluation_accuracy,
            "target_accuracy": target,
            "maximum_accuracy_mismatch": tolerance,
            "absolute_calibration_mismatch": float(selection["absolute_calibration_mismatch"]),
            "within_tolerance": True,
        },
        "selection": {
            "selection_mode": "single_run_fixed_broad_coverage_correct_pools",
            "calibration_trajectory": [
                {
                    "checkpoint_epoch": int(epoch),
                    "calibration_accuracy": float(calibration_trajectory[epoch]),
                }
                for epoch in checkpoint_epochs
            ],
            "anchor_coverage": anchor_coverage,
            "candidate_coverage": candidate_coverage,
            "support_count": support_count,
        },
        "training": training,
        "rows": rows,
        "claim_scope": {
            "dataset": data.dataset_name,
            "selected_checkpoint_only": True,
            "fixed_pool_size_across_architectures": True,
            "all_fine_classes_required": False,
            "minimum_unique_fine_class_gate": minimum_classes,
            "performance_selection_and_topology_share_training_run": True,
            "cross_runner_checkpoint_replay_assumed": False,
        },
    }


__all__ = ["balanced_correct_pool", "run_cifar_single_run_broad_pool_component"]
