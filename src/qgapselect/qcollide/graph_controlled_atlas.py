"""Control-validated Cora collision atlas with an explicit auxiliary control objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from math import sqrt
from typing import Any

import numpy as np

from .graph_atlas import (
    _build_model,
    _max_overlap_degree,
    _nearest_same_label_indices,
    _optional_dependencies,
    _select_correct_per_class,
    graph_control_descriptors,
)
from .text_atlas import (
    control_output,
    control_projection_from_head,
    pairwise_distances,
    residual_hidden,
    standardized_projection,
)
from .topology_persistence import collision_filtration_profile


def multioutput_r2(expected: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray]:
    """Uniform-average held-out R² with per-descriptor diagnostics."""

    y = np.asarray(expected, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    if y.shape != yhat.shape or y.ndim != 2:
        raise ValueError("expected and predicted controls must be aligned matrices")
    residual = np.sum((y - yhat) ** 2, axis=0)
    centered = y - y.mean(axis=0, keepdims=True)
    total = np.sum(centered**2, axis=0)
    scores = 1.0 - residual / np.maximum(total, 1e-12)
    return float(scores.mean()), scores


def run_controlled_graph_atlas_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train one GNN with a label-free control head and measure collision topology."""

    (
        torch,
        _,
        StandardScaler,
        Planetoid,
        NormalizeFeatures,
        GCNConv,
        SAGEConv,
        GATConv,
    ) = _optional_dependencies()
    allowed = tuple(str(value) for value in config["architectures"])
    seeds = tuple(int(value) for value in config["model_seeds"])
    if architecture not in allowed:
        raise ValueError(f"architecture {architecture!r} is not configured")
    if model_seed not in seeds:
        raise ValueError(f"model_seed={model_seed} is not configured")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed % (2**32))
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    dataset = Planetoid(
        root=str(config.get("data_root", ".cache/qcollide-graph")),
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0]
    hidden_dimension = int(config["hidden_dimension"])
    model = _build_model(
        torch,
        architecture=architecture,
        in_channels=int(dataset.num_features),
        hidden=hidden_dimension,
        classes=int(dataset.num_classes),
        convs=(GCNConv, SAGEConv, GATConv),
    )

    features = data.x.cpu().float().numpy()
    edge_index = data.edge_index.cpu().numpy().astype(np.int64)
    descriptors_raw = graph_control_descriptors(features, edge_index)
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    train_indices = np.flatnonzero(train_mask)
    calibration_indices = np.flatnonzero(val_mask)
    evaluation_indices = np.flatnonzero(test_mask)
    descriptor_scaler = StandardScaler().fit(descriptors_raw[train_indices])
    descriptors = descriptor_scaler.transform(descriptors_raw).astype(np.float32)
    descriptor_tensor = torch.from_numpy(descriptors)

    control_head = torch.nn.Linear(hidden_dimension, descriptors.shape[1])
    optimizer = torch.optim.Adam(
        [*model.parameters(), *control_head.parameters()],
        lr=float(config.get("learning_rate", 0.01)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )
    control_loss_weight = float(config["control_loss_weight"])
    epochs = int(config.get("epochs", 250))
    for _ in range(epochs):
        model.train()
        control_head.train()
        optimizer.zero_grad(set_to_none=True)
        logits, hidden_tensor = model(data.x, data.edge_index)
        task_loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )
        control_prediction = control_head(hidden_tensor)
        control_loss = torch.nn.functional.mse_loss(
            control_prediction[data.train_mask],
            descriptor_tensor[data.train_mask],
        )
        loss = task_loss + control_loss_weight * control_loss
        loss.backward()
        optimizer.step()

    model.eval()
    control_head.eval()
    with torch.no_grad():
        logits_tensor, hidden_tensor = model(data.x, data.edge_index)
        control_prediction_tensor = control_head(hidden_tensor)
    logits = logits_tensor.cpu().float().numpy()
    hidden_raw = hidden_tensor.cpu().float().numpy()
    control_prediction = control_prediction_tensor.cpu().float().numpy()
    labels = data.y.cpu().numpy().astype(np.int64)
    predictions = logits.argmax(axis=1)

    control_r2, descriptor_r2 = multioutput_r2(
        descriptors[calibration_indices],
        control_prediction[calibration_indices],
    )
    accuracies = {
        "train_accuracy": float(np.mean(predictions[train_indices] == labels[train_indices])),
        "calibration_accuracy": float(
            np.mean(predictions[calibration_indices] == labels[calibration_indices])
        ),
        "evaluation_accuracy": float(
            np.mean(predictions[evaluation_indices] == labels[evaluation_indices])
        ),
    }

    hidden_scaler = StandardScaler().fit(hidden_raw[train_indices])
    hidden = hidden_scaler.transform(hidden_raw)
    weight = control_head.weight.detach().cpu().float().numpy()
    scale = np.asarray(hidden_scaler.scale_, dtype=float)
    coefficients = weight * scale[None, :]

    anchors_per_class = int(config.get("anchors_per_class", 6))
    anchor_indices = _select_correct_per_class(
        logits,
        labels,
        calibration_indices,
        anchors_per_class,
    )
    candidate_indices = _select_correct_per_class(
        logits,
        labels,
        evaluation_indices,
        anchors_per_class,
    )
    correct_calibration = calibration_indices[
        predictions[calibration_indices] == labels[calibration_indices]
    ]
    benign_hidden = hidden[correct_calibration]
    benign_logits = logits[correct_calibration]
    benign_labels = labels[correct_calibration]

    epsilon_multipliers = tuple(float(value) for value in config["epsilon_multipliers"])
    rows: list[dict[str, Any]] = []
    for visible_rank in (int(value) for value in config["visible_ranks"]):
        projection = control_projection_from_head(coefficients, visible_rank)
        standardized = standardized_projection(projection, hidden[train_indices])
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
            sorted({max(1e-10, nominal_epsilon * value) for value in epsilon_multipliers})
        )

        anchor_hidden = hidden[anchor_indices]
        candidate_hidden = hidden[candidate_indices]
        anchor_control = control_output(anchor_hidden, standardized)
        candidate_control = control_output(candidate_hidden, standardized)
        anchor_residual = residual_hidden(anchor_hidden, projection)
        candidate_residual = residual_hidden(candidate_hidden, projection)
        control_distances = pairwise_distances(candidate_control, anchor_control)
        payload_distances = pairwise_distances(candidate_residual, anchor_residual)
        behavior_distances = pairwise_distances(
            logits[candidate_indices],
            logits[anchor_indices],
        ) / sqrt(logits.shape[1])
        valid_pairs = labels[candidate_indices, None] != labels[anchor_indices][None, :]
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
        points: list[dict[str, Any]] = []
        for point in profile.points:
            epsilon = float(point.control_epsilon)
            mask = (
                (control_distances <= epsilon)
                & (payload_distances >= payload_delta)
                & (behavior_distances >= behavior_gamma)
                & valid_pairs
            )
            points.append(
                {
                    "control_epsilon": epsilon,
                    **asdict(point.metrics),
                    "max_overlap_degree": _max_overlap_degree(mask),
                }
            )
        rows.append(
            {
                "visible_rank": visible_rank,
                "nominal_epsilon": nominal_epsilon,
                "payload_delta": payload_delta,
                "behavior_gamma": behavior_gamma,
                "filtration_summary": asdict(profile.summary),
                "basin_persistence": asdict(profile.basin_persistence),
                "filtration_points": points,
            }
        )

    min_control_r2 = float(config["minimum_control_r2"])
    min_evaluation_accuracy = float(config["minimum_evaluation_accuracy"])
    return {
        "artifact_type": "qcollide_controlled_graph_functional_collision_component",
        "schema_version": 1,
        "domain": "graph_learning",
        "dataset": "cora",
        "architecture": architecture,
        "model_seed": model_seed,
        "model_diagnostics": {
            **accuracies,
            "hidden_dimension": int(hidden.shape[1]),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
                + sum(parameter.numel() for parameter in control_head.parameters())
            ),
            "control_head_calibration_r2": control_r2,
            "control_descriptor_r2": [float(value) for value in descriptor_r2],
            "control_loss_weight": control_loss_weight,
        },
        "gates": {
            "minimum_control_r2": min_control_r2,
            "control_r2_pass": control_r2 >= min_control_r2,
            "minimum_evaluation_accuracy": min_evaluation_accuracy,
            "evaluation_accuracy_pass": accuracies["evaluation_accuracy"]
            >= min_evaluation_accuracy,
        },
        "selection": {
            "anchors_per_class": anchors_per_class,
            "anchor_count": int(len(anchor_indices)),
            "candidate_count": int(len(candidate_indices)),
        },
        "rows": rows,
        "claim_boundary": {
            "auxiliary_control_is_label_agnostic": True,
            "control_gate_frozen_before_v2_results": True,
            "thresholds_calibrated_on_validation_nodes_only": True,
            "evaluation_nodes_are_test_mask_only": True,
            "architecture_causality_claimed": False,
            "quantum_execution": False
        },
    }


__all__ = ["multioutput_r2", "run_controlled_graph_atlas_component"]
