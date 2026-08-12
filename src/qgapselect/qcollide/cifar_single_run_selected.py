"""Single-run performance matching and checkpoint-local CIFAR topology.

This module removes the cross-run replay assumption exposed by CIFAR-100
selected-v3. A model is trained once, every preregistered checkpoint is scored
inside that same process, and topology is measured immediately from the chosen
checkpoint state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .cifar_models import derived_seed, load_cifar_data, train_cifar_model
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
    _stratified_correct_indices,
)


def select_checkpoint_from_calibration_trajectory(
    trajectory: Mapping[int, float],
    *,
    target_accuracy: float,
    maximum_accuracy_mismatch: float,
) -> dict[str, object]:
    """Select the observed checkpoint closest to the fixed calibration target.

    Ties are broken toward the earlier epoch. The function fails closed if the
    best observed checkpoint is outside the preregistered tolerance.
    """

    if not trajectory:
        raise ValueError("trajectory must be non-empty")
    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must lie in (0,1)")
    if not 0.0 <= maximum_accuracy_mismatch < 1.0:
        raise ValueError("maximum_accuracy_mismatch must lie in [0,1)")
    rows = []
    for epoch, accuracy in trajectory.items():
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("trajectory epochs must be non-negative integers")
        value = float(accuracy)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("trajectory accuracies must lie in [0,1]")
        rows.append((epoch, value, abs(value - target_accuracy)))
    epoch, accuracy, mismatch = min(rows, key=lambda item: (item[2], item[0]))
    if mismatch > maximum_accuracy_mismatch + 1e-12:
        raise RuntimeError(
            "no checkpoint satisfies the fixed performance-matching gate: "
            f"best mismatch {mismatch:.6f} > {maximum_accuracy_mismatch:.6f}"
        )
    return {
        "checkpoint_epoch": int(epoch),
        "calibration_accuracy": float(accuracy),
        "absolute_calibration_mismatch": float(mismatch),
        "within_tolerance": True,
    }


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(logits).argmax(axis=1) == np.asarray(labels)))


def run_cifar_single_run_selected_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train once, select the matched checkpoint, then measure its topology."""

    schema_version = _positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("only schema_version=1 is supported")
    master_seed = _positive_int(config.get("master_seed"), "master_seed")
    architectures = tuple(
        str(item) for item in _sequence(config.get("architectures"), "architectures")
    )
    if architecture not in architectures:
        raise ValueError(f"architecture {architecture!r} is not configured")
    model_seeds = _integer_tuple(
        config.get("model_seeds"),
        "model_seeds",
        allow_zero=True,
    )
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
    if tuple(sorted(set(checkpoint_epochs))) != checkpoint_epochs:
        raise ValueError("checkpoint_epochs must be strictly increasing")
    visible_ranks = _integer_tuple(config.get("main_visible_ranks"), "main_visible_ranks")

    data = load_cifar_data(
        dataset=str(config.get("dataset")),
        dataset_seed=_nonnegative_int(config.get("dataset_seed"), "dataset_seed"),
        data_root=str(config.get("data_root", ".cache/qcollide-data")),
        train_samples=_positive_int(config.get("train_samples"), "train_samples"),
        calibration_samples=_positive_int(
            config.get("calibration_samples"),
            "calibration_samples",
        ),
        evaluation_samples=_positive_int(
            config.get("evaluation_samples"),
            "evaluation_samples",
        ),
        control_grid_size=_positive_int(
            config.get("control_grid_size"),
            "control_grid_size",
        ),
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
        control_loss_weight=_number(
            config.get("control_loss_weight"),
            "control_loss_weight",
        ),
        control_grid_size=_positive_int(
            config.get("control_grid_size"),
            "control_grid_size",
        ),
        checkpoint_epochs=checkpoint_epochs,
        device_name=str(config.get("device", "auto")),
    )
    device = _device(str(config.get("device", "auto")))

    calibration_trajectory: dict[int, float] = {}
    for epoch in checkpoint_epochs:
        model.load_state_dict(checkpoints[epoch])
        model.to(device)
        model.eval()
        _, calibration_logits = _encode(
            model,
            data.calibration_images,
            device=device,
        )
        calibration_trajectory[epoch] = _accuracy(
            calibration_logits,
            data.calibration_labels,
        )

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

    per_class = _positive_int(config.get("anchors_per_class"), "anchors_per_class")
    anchor_indices = _stratified_correct_indices(
        calibration_logits_all,
        data.calibration_labels,
        class_count=data.class_count,
        per_class=per_class,
    )
    candidate_indices = _stratified_correct_indices(
        evaluation_logits_all,
        data.evaluation_labels,
        class_count=data.class_count,
        per_class=per_class,
    )
    anchor_hidden = calibration_hidden_all[anchor_indices]
    anchor_logits = calibration_logits_all[anchor_indices]
    candidate_hidden = evaluation_hidden_all[candidate_indices]
    candidate_logits = evaluation_logits_all[candidate_indices]

    support_count = _positive_int(config.get("support_samples"), "support_samples")
    if support_count > len(data.train_images):
        raise ValueError("support_samples exceeds training partition")
    support_rng = np.random.default_rng(
        derived_seed(master_seed, architecture, model_seed, "support")
    )
    support_indices = np.sort(
        support_rng.choice(len(data.train_images), size=support_count, replace=False)
    )
    support_hidden, _ = _encode(
        model,
        data.train_images[support_indices],
        device=device,
    )

    benign_repetitions = _positive_int(
        config.get("benign_repetitions"),
        "benign_repetitions",
    )
    benign_sigma = _number(config.get("benign_sigma"), "benign_sigma")
    benign_images, benign_anchor_indices = _make_benign_images(
        data.calibration_images[anchor_indices],
        seed=derived_seed(master_seed, architecture, model_seed, "benign"),
        repetitions=benign_repetitions,
        sigma=benign_sigma,
    )
    benign_hidden, benign_logits = _encode(model, benign_images, device=device)

    epsilon_multipliers = _float_tuple(
        config.get("epsilon_multipliers"),
        "epsilon_multipliers",
        minimum=0.0,
    )
    if tuple(sorted(set(epsilon_multipliers))) != epsilon_multipliers:
        raise ValueError("epsilon_multipliers must be strictly increasing")
    quantiles = {
        "control_quantile": _number(config.get("control_quantile"), "control_quantile"),
        "payload_quantile": _number(config.get("payload_quantile"), "payload_quantile"),
        "behavior_quantile": _number(config.get("behavior_quantile"), "behavior_quantile"),
    }
    for name, value in quantiles.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0,1)")

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

    trajectory_rows = [
        {
            "checkpoint_epoch": int(epoch),
            "calibration_accuracy": float(calibration_trajectory[epoch]),
            "absolute_target_mismatch": abs(
                float(calibration_trajectory[epoch]) - target
            ),
        }
        for epoch in checkpoint_epochs
    ]
    return {
        "artifact_type": "qcollide_cifar_selected_checkpoint_topology_component",
        "schema_version": schema_version,
        "selection_schema_version": 2,
        "master_seed": master_seed,
        "architecture": architecture,
        "model_seed": model_seed,
        "selected_checkpoint_epoch": selected_epoch,
        "training_seed": training_seed,
        "selected_performance": {
            "calibration_accuracy": calibration_accuracy,
            "evaluation_accuracy": evaluation_accuracy,
            "target_accuracy": target,
            "maximum_accuracy_mismatch": tolerance,
            "absolute_calibration_mismatch": float(
                selection["absolute_calibration_mismatch"]
            ),
            "within_tolerance": True,
        },
        "selection": {
            "selection_mode": "single_training_run_dense_checkpoint_selection",
            "selection_rule": "minimum absolute calibration mismatch; earlier epoch tie-break",
            "selection_checkpoint_epoch": selected_epoch,
            "checkpoint_count": len(checkpoint_epochs),
            "calibration_trajectory": trajectory_rows,
            "anchors_per_class": per_class,
            "anchor_count": len(anchor_indices),
            "candidate_count": len(candidate_indices),
            "support_count": support_count,
        },
        "training": training,
        "rows": rows,
        "claim_scope": {
            "dataset": data.dataset_name,
            "selected_checkpoint_only": True,
            "checkpoint_local_anchor_selection": True,
            "performance_selection_and_topology_share_training_run": True,
            "cross_runner_checkpoint_replay_assumed": False,
            "persistent_control_threshold_filtration": True,
            "coherent_quantum_execution": False,
            "hardware_runtime_advantage": False,
        },
    }


__all__ = [
    "run_cifar_single_run_selected_component",
    "select_checkpoint_from_calibration_trajectory",
]
