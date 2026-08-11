"""Functional-collision atlas for frozen ImageNet-pretrained visual encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from .text_atlas import (
    control_output,
    control_projection_from_head,
    pairwise_distances,
    residual_hidden,
    standardized_projection,
)
from .topology_persistence import collision_filtration_profile


def low_frequency_rgb_controls(images: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Label-free low-frequency RGB grid descriptors for CIFAR images."""

    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("images must have shape (batch,height,width,3)")
    if values.shape[1] % grid_size or values.shape[2] % grid_size:
        raise ValueError("image dimensions must be divisible by grid_size")
    height = values.shape[1] // grid_size
    width = values.shape[2] // grid_size
    pooled = values.reshape(
        values.shape[0],
        grid_size,
        height,
        grid_size,
        width,
        3,
    ).mean(axis=(2, 4))
    return pooled.reshape(values.shape[0], -1)


def _balanced_indices(labels: Sequence[int], per_class: int, seed: int) -> np.ndarray:
    targets = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(np.unique(targets).tolist()):
        candidates = np.flatnonzero(targets == label)
        if len(candidates) < per_class:
            raise RuntimeError(f"class {label} has only {len(candidates)} samples")
        chosen = rng.choice(candidates, size=per_class, replace=False)
        selected.extend(int(value) for value in chosen)
    return np.asarray(selected, dtype=np.int64)


def _select_correct_per_class(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
) -> np.ndarray:
    predictions = np.asarray(logits).argmax(axis=1)
    selected: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) < per_class:
            raise RuntimeError(
                f"class {label} has {len(candidates)} correctly predicted samples; "
                f"{per_class} required"
            )
        margins = logits[candidates, label] - np.max(
            np.where(
                np.arange(logits.shape[1])[None, :] == label,
                -np.inf,
                logits[candidates],
            ),
            axis=1,
        )
        order = candidates[np.argsort(-margins, kind="mergesort")]
        selected.extend(int(value) for value in order[:per_class])
    return np.asarray(selected, dtype=np.int64)


def _nearest_same_label_indices(controls: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(controls, controls)
    np.fill_diagonal(distances, np.inf)
    output = np.empty(len(labels), dtype=np.int64)
    for index, label in enumerate(labels):
        candidates = np.flatnonzero(labels == label)
        candidates = candidates[candidates != index]
        if len(candidates) == 0:
            raise RuntimeError("benign calibration requires at least two correct samples per class")
        output[index] = int(candidates[np.argmin(distances[index, candidates])])
    return output


def _dependencies():
    try:
        import torch
        import torchvision
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler
        from torchvision.models import (
            ConvNeXt_Tiny_Weights,
            ResNet50_Weights,
            Swin_T_Weights,
            ViT_B_16_Weights,
            convnext_tiny,
            resnet50,
            swin_t,
            vit_b_16,
        )
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError(
            "install pretrained vision dependencies: torch torchvision scikit-learn"
        ) from exc
    return (
        torch,
        torchvision,
        LogisticRegression,
        Ridge,
        StandardScaler,
        {
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
            "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
            "swin_t": (swin_t, Swin_T_Weights.DEFAULT),
            "vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT),
        },
    )


def _build_encoder(architecture: str, model_map: Mapping[str, object]):
    if architecture not in model_map:
        raise ValueError(f"unsupported pretrained vision architecture {architecture!r}")
    constructor, weights = model_map[architecture]
    model = constructor(weights=weights)
    if architecture == "resnet50":
        model.fc = __import__("torch").nn.Identity()
    elif architecture == "convnext_tiny":
        model.classifier[-1] = __import__("torch").nn.Identity()
    elif architecture == "swin_t":
        model.head = __import__("torch").nn.Identity()
    elif architecture == "vit_b_16":
        model.heads = __import__("torch").nn.Identity()
    return model.eval(), weights


def _extract_features(
    model,
    transform,
    images: Sequence[object],
    *,
    torch,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([transform(image) for image in images[start : start + batch_size]])
            values = model(batch)
            if isinstance(values, tuple):
                values = values[0]
            outputs.append(values.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _dataset_arrays(dataset, indices: np.ndarray) -> tuple[list[object], np.ndarray, np.ndarray]:
    images: list[object] = []
    raw: list[np.ndarray] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image)
        raw.append(np.asarray(image, dtype=np.float32) / 255.0)
        labels.append(int(label))
    return images, np.asarray(raw, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def run_pretrained_vision_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    fixture_seed: int,
) -> dict[str, object]:
    """Run one frozen pretrained visual encoder with common task/control probes."""

    torch, torchvision, LogisticRegression, Ridge, StandardScaler, model_map = _dependencies()
    if architecture not in tuple(str(value) for value in config["architectures"]):
        raise ValueError(f"architecture {architecture!r} is not configured")
    if fixture_seed not in tuple(int(value) for value in config["fixture_seeds"]):
        raise ValueError(f"fixture_seed={fixture_seed} is not configured")
    torch.manual_seed(fixture_seed)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    np.random.seed(fixture_seed % (2**32))

    root = Path(str(config.get("data_root", ".cache/qcollide-pretrained-vision")))
    train_dataset = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=str(root), train=False, download=True)
    train_targets = np.asarray(train_dataset.targets, dtype=np.int64)
    test_targets = np.asarray(test_dataset.targets, dtype=np.int64)
    train_count = int(config["train_per_class"])
    calibration_count = int(config["calibration_per_class"])
    evaluation_count = int(config["evaluation_per_class"])

    rng = np.random.default_rng(fixture_seed)
    train_indices: list[int] = []
    calibration_indices: list[int] = []
    for label in range(10):
        candidates = np.flatnonzero(train_targets == label)
        chosen = rng.choice(candidates, size=train_count + calibration_count, replace=False)
        train_indices.extend(int(value) for value in chosen[:train_count])
        calibration_indices.extend(int(value) for value in chosen[train_count:])
    evaluation_indices = _balanced_indices(
        test_targets,
        evaluation_count,
        fixture_seed + 11,
    )

    train_images, train_raw, train_labels = _dataset_arrays(
        train_dataset,
        np.asarray(train_indices, dtype=np.int64),
    )
    calibration_images, calibration_raw, calibration_labels = _dataset_arrays(
        train_dataset,
        np.asarray(calibration_indices, dtype=np.int64),
    )
    evaluation_images, evaluation_raw, evaluation_labels = _dataset_arrays(
        test_dataset,
        evaluation_indices,
    )

    model, weights = _build_encoder(architecture, model_map)
    transform = weights.transforms()
    batch_size = int(config.get("feature_batch_size", 16))
    train_hidden_raw = _extract_features(
        model,
        transform,
        train_images,
        torch=torch,
        batch_size=batch_size,
    )
    calibration_hidden_raw = _extract_features(
        model,
        transform,
        calibration_images,
        torch=torch,
        batch_size=batch_size,
    )
    evaluation_hidden_raw = _extract_features(
        model,
        transform,
        evaluation_images,
        torch=torch,
        batch_size=batch_size,
    )
    hidden_scaler = StandardScaler().fit(train_hidden_raw)
    train_hidden = hidden_scaler.transform(train_hidden_raw)
    calibration_hidden = hidden_scaler.transform(calibration_hidden_raw)
    evaluation_hidden = hidden_scaler.transform(evaluation_hidden_raw)

    classifier = LogisticRegression(
        C=float(config.get("classifier_c", 1.0)),
        max_iter=int(config.get("classifier_max_iter", 1000)),
        solver="lbfgs",
    ).fit(train_hidden, train_labels)
    calibration_logits = classifier.decision_function(calibration_hidden)
    evaluation_logits = classifier.decision_function(evaluation_hidden)
    calibration_predictions = calibration_logits.argmax(axis=1)
    evaluation_predictions = evaluation_logits.argmax(axis=1)
    calibration_accuracy = float(np.mean(calibration_predictions == calibration_labels))
    evaluation_accuracy = float(np.mean(evaluation_predictions == evaluation_labels))

    grid_size = int(config.get("control_grid_size", 4))
    descriptor_scaler = StandardScaler().fit(low_frequency_rgb_controls(train_raw, grid_size))
    train_controls = descriptor_scaler.transform(low_frequency_rgb_controls(train_raw, grid_size))
    calibration_controls = descriptor_scaler.transform(
        low_frequency_rgb_controls(calibration_raw, grid_size)
    )
    control_head = Ridge(alpha=float(config.get("control_ridge_alpha", 1.0))).fit(
        train_hidden,
        train_controls,
    )
    control_r2 = float(control_head.score(calibration_hidden, calibration_controls))
    coefficients = np.asarray(control_head.coef_, dtype=float)

    correct_calibration = np.flatnonzero(calibration_predictions == calibration_labels)
    benign_hidden = calibration_hidden[correct_calibration]
    benign_logits = calibration_logits[correct_calibration]
    benign_labels = calibration_labels[correct_calibration]
    anchors_per_class = int(config["anchors_per_class"])
    anchor_indices = _select_correct_per_class(
        calibration_logits,
        calibration_labels,
        per_class=anchors_per_class,
    )
    candidate_indices = _select_correct_per_class(
        evaluation_logits,
        evaluation_labels,
        per_class=anchors_per_class,
    )
    anchor_hidden = calibration_hidden[anchor_indices]
    candidate_hidden = evaluation_hidden[candidate_indices]
    anchor_logits = calibration_logits[anchor_indices]
    candidate_logits = evaluation_logits[candidate_indices]
    anchor_labels = calibration_labels[anchor_indices]
    candidate_labels = evaluation_labels[candidate_indices]

    rows: list[dict[str, Any]] = []
    for visible_rank in (int(value) for value in config["visible_ranks"]):
        projection = control_projection_from_head(coefficients, visible_rank)
        standardized = standardized_projection(projection, train_hidden)
        benign_control = control_output(benign_hidden, standardized)
        benign_residual = residual_hidden(benign_hidden, projection)
        nearest = _nearest_same_label_indices(benign_control, benign_labels)
        nominal_epsilon = max(
            float(
                np.quantile(
                    np.linalg.norm(benign_control - benign_control[nearest], axis=1),
                    float(config["control_quantile"]),
                )
            ),
            1e-8,
        )
        payload_delta = float(
            np.quantile(
                np.linalg.norm(benign_residual - benign_residual[nearest], axis=1),
                float(config["payload_quantile"]),
            )
        )
        behavior_gamma = float(
            np.quantile(
                np.linalg.norm(benign_logits - benign_logits[nearest], axis=1)
                / sqrt(benign_logits.shape[1]),
                float(config["behavior_quantile"]),
            )
        )
        thresholds = tuple(
            sorted(
                {
                    max(1e-10, nominal_epsilon * float(multiplier))
                    for multiplier in config["epsilon_multipliers"]
                }
            )
        )
        anchor_control = control_output(anchor_hidden, standardized)
        candidate_control = control_output(candidate_hidden, standardized)
        anchor_residual = residual_hidden(anchor_hidden, projection)
        candidate_residual = residual_hidden(candidate_hidden, projection)
        control_distances = pairwise_distances(candidate_control, anchor_control)
        payload_distances = pairwise_distances(candidate_residual, anchor_residual)
        behavior_distances = pairwise_distances(candidate_logits, anchor_logits) / sqrt(
            anchor_logits.shape[1]
        )
        valid_pairs = candidate_labels[:, None] != anchor_labels[None, :]
        displacements = candidate_residual[:, None, :] - anchor_residual[None, :, :]
        profile = collision_filtration_profile(
            control_distances,
            payload_distances,
            behavior_distances,
            thresholds,
            nominal_epsilon=nominal_epsilon,
            payload_delta=payload_delta,
            behavior_gamma=behavior_gamma,
            valid_pairs=valid_pairs,
            edge_displacements=displacements,
        )
        rows.append(
            {
                "visible_rank": visible_rank,
                "nominal_epsilon": nominal_epsilon,
                "payload_delta": payload_delta,
                "behavior_gamma": behavior_gamma,
                "filtration_summary": asdict(profile.summary),
                "basin_persistence": asdict(profile.basin_persistence),
                "filtration_points": [
                    {"control_epsilon": point.control_epsilon, **asdict(point.metrics)}
                    for point in profile.points
                ],
            }
        )

    minimum_control_r2 = float(config["minimum_control_r2"])
    minimum_accuracy = float(config["minimum_evaluation_accuracy"])
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "artifact_type": "qcollide_pretrained_vision_functional_collision_component",
        "schema_version": 1,
        "domain": "pretrained_vision",
        "dataset": "cifar10",
        "architecture": architecture,
        "fixture_seed": fixture_seed,
        "pretraining": "imagenet1k_default_torchvision_weights",
        "model_diagnostics": {
            "hidden_dimension": int(train_hidden.shape[1]),
            "parameter_count": parameter_count,
            "calibration_accuracy": calibration_accuracy,
            "evaluation_accuracy": evaluation_accuracy,
            "control_head_calibration_r2": control_r2,
        },
        "gates": {
            "minimum_control_r2": minimum_control_r2,
            "control_r2_pass": control_r2 >= minimum_control_r2,
            "minimum_evaluation_accuracy": minimum_accuracy,
            "evaluation_accuracy_pass": evaluation_accuracy >= minimum_accuracy,
        },
        "selection": {
            "train_count": int(len(train_labels)),
            "calibration_count": int(len(calibration_labels)),
            "evaluation_count": int(len(evaluation_labels)),
            "anchors_per_class": anchors_per_class,
        },
        "rows": rows,
        "claim_boundary": {
            "encoder_weights_are_frozen": True,
            "common_linear_task_probe": True,
            "control_descriptors_are_label_agnostic": True,
            "pretraining_dataset_is_not_the_evaluation_dataset": True,
            "architecture_causality_claimed": False,
            "imagenet_accuracy_claimed": False,
            "quantum_execution": False,
        },
    }


__all__ = ["low_frequency_rgb_controls", "run_pretrained_vision_component"]
