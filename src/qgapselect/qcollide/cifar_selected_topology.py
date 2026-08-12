"""Selected-checkpoint CIFAR topology with checkpoint-local sample selection."""

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


def selected_checkpoint_epoch(
    config: Mapping[str, object],
    architecture: str,
    model_seed: int,
) -> int:
    """Resolve one preregistered architecture × seed checkpoint."""

    mapping = config.get("selected_checkpoint_epoch")
    if not isinstance(mapping, Mapping):
        raise ValueError("selected_checkpoint_epoch must be a mapping")
    architecture_mapping = mapping.get(architecture)
    if not isinstance(architecture_mapping, Mapping):
        raise ValueError(f"no selected checkpoint mapping for {architecture!r}")
    key = str(model_seed)
    if key not in architecture_mapping:
        raise ValueError(f"no selected checkpoint for {architecture!r}, seed={model_seed}")
    value = architecture_mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("selected checkpoint epoch must be a non-negative integer")
    return value


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(logits).argmax(axis=1) == np.asarray(labels)))


def run_cifar_selected_topology_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train deterministically, then measure topology only at the frozen checkpoint."""

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
    selected_epoch = selected_checkpoint_epoch(config, architecture, model_seed)
    if selected_epoch not in checkpoint_epochs:
        raise ValueError("selected checkpoint must be included in checkpoint_epochs")
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
    model.load_state_dict(checkpoints[selected_epoch])
    device = _device(str(config.get("device", "auto")))
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
    target = _number(config.get("performance_match_target"), "performance_match_target")
    tolerance = _number(
        config.get("performance_match_tolerance"),
        "performance_match_tolerance",
    )
    mismatch = abs(calibration_accuracy - target)
    if mismatch > tolerance + 1e-12:
        raise RuntimeError(
            f"selected checkpoint misses fixed performance gate: {mismatch:.6f} > {tolerance:.6f}"
        )

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

    return {
        "artifact_type": "qcollide_cifar_selected_checkpoint_topology_component",
        "schema_version": schema_version,
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
            "absolute_calibration_mismatch": mismatch,
            "within_tolerance": True,
        },
        "selection": {
            "anchors_per_class": per_class,
            "anchor_count": len(anchor_indices),
            "candidate_count": len(candidate_indices),
            "support_count": support_count,
            "selection_checkpoint_epoch": selected_epoch,
        },
        "training": training,
        "rows": rows,
        "claim_scope": {
            "dataset": data.dataset_name,
            "selected_checkpoint_only": True,
            "checkpoint_local_anchor_selection": True,
            "persistent_control_threshold_filtration": True,
            "coherent_quantum_execution": False,
            "hardware_runtime_advantage": False,
        },
    }


__all__ = ["run_cifar_selected_topology_component", "selected_checkpoint_epoch"]
