"""Cora functional-collision atlas across GCN, GraphSAGE, and GAT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
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


def graph_control_descriptors(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Return six label-agnostic local structural/feature descriptors per node."""

    values = np.asarray(features, dtype=float)
    edges = np.asarray(edge_index, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("features must be a matrix")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edge-count)")
    n = values.shape[0]
    src, dst = edges
    if src.size and (src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n):
        raise ValueError("edge_index contains an out-of-range node")

    degree = np.bincount(src, minlength=n).astype(float)
    neighbor_degree_sum = np.bincount(src, weights=degree[dst], minlength=n)
    mean_neighbor_degree = neighbor_degree_sum / np.maximum(degree, 1.0)
    two_hop_volume = np.bincount(src, weights=np.maximum(degree[dst], 1.0), minlength=n)

    neighbor_sum = np.zeros_like(values, dtype=float)
    np.add.at(neighbor_sum, src, values[dst])
    neighbor_mean = neighbor_sum / np.maximum(degree[:, None], 1.0)
    feature_norm = np.linalg.norm(values, axis=1)
    neighbor_norm = np.linalg.norm(neighbor_mean, axis=1)
    cosine = np.sum(values * neighbor_mean, axis=1) / np.maximum(
        feature_norm * neighbor_norm,
        1e-12,
    )
    local_feature_gap = np.linalg.norm(values - neighbor_mean, axis=1)

    return np.column_stack(
        [
            np.log1p(degree),
            np.log1p(mean_neighbor_degree),
            np.log1p(two_hop_volume),
            feature_norm,
            cosine,
            local_feature_gap,
        ]
    )


def _optional_dependencies():
    try:
        import torch
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from torch_geometric.datasets import Planetoid
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv
        from torch_geometric.transforms import NormalizeFeatures
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError(
            "install graph atlas extras with: pip install -e '.[atlas_graph]'"
        ) from exc
    return torch, Ridge, StandardScaler, Planetoid, NormalizeFeatures, GCNConv, SAGEConv, GATConv


def _build_model(torch, *, architecture: str, in_channels: int, hidden: int, classes: int, convs):
    GCNConv, SAGEConv, GATConv = convs
    nn = torch.nn

    class GraphEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if architecture == "gcn":
                self.conv1 = GCNConv(in_channels, hidden)
                self.conv2 = GCNConv(hidden, hidden)
            elif architecture == "graphsage":
                self.conv1 = SAGEConv(in_channels, hidden)
                self.conv2 = SAGEConv(hidden, hidden)
            elif architecture == "gat":
                heads = 4
                if hidden % heads:
                    raise ValueError("GAT hidden dimension must be divisible by 4")
                self.conv1 = GATConv(in_channels, hidden // heads, heads=heads, dropout=0.2)
                self.conv2 = GATConv(hidden, hidden, heads=1, concat=False, dropout=0.2)
            else:
                raise ValueError(f"unknown graph architecture {architecture!r}")
            self.head = nn.Linear(hidden, classes)

        def encode(self, x, edge_index):
            hidden_values = self.conv1(x, edge_index)
            hidden_values = torch.relu(hidden_values)
            hidden_values = nn.functional.dropout(hidden_values, p=0.25, training=self.training)
            hidden_values = self.conv2(hidden_values, edge_index)
            return torch.relu(hidden_values)

        def forward(self, x, edge_index):
            hidden_values = self.encode(x, edge_index)
            return self.head(hidden_values), hidden_values

    return GraphEncoder()


def _select_correct_per_class(
    logits: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    count: int,
) -> np.ndarray:
    predictions = np.asarray(logits).argmax(axis=1)
    selected: list[int] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        candidates = indices[(labels[indices] == label) & (predictions[indices] == label)]
        if len(candidates) < count:
            raise RuntimeError(
                f"class {label} has {len(candidates)} correct split nodes; {count} required"
            )
        selected.extend(int(value) for value in candidates[:count])
    return np.asarray(selected, dtype=np.int64)


def _nearest_same_label_indices(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(values, values)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    distances[~same] = np.inf
    if np.any(~np.isfinite(np.min(distances, axis=1))):
        raise RuntimeError("each calibration class needs at least two correct nodes")
    return np.argmin(distances, axis=1)


def _max_overlap_degree(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    left = int(mask.sum(axis=1).max(initial=0))
    right = int(mask.sum(axis=0).max(initial=0))
    return max(left, right)


def run_graph_atlas_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train one Cora model and measure its functional-collision topology."""

    (
        torch,
        Ridge,
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
    model = _build_model(
        torch,
        architecture=architecture,
        in_channels=int(dataset.num_features),
        hidden=int(config["hidden_dimension"]),
        classes=int(dataset.num_classes),
        convs=(GCNConv, SAGEConv, GATConv),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.01)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )
    epochs = int(config.get("epochs", 200))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_tensor, hidden_tensor = model(data.x, data.edge_index)
    logits = logits_tensor.cpu().float().numpy()
    hidden_raw = hidden_tensor.cpu().float().numpy()
    labels = data.y.cpu().numpy().astype(np.int64)
    edge_index = data.edge_index.cpu().numpy().astype(np.int64)
    features = data.x.cpu().float().numpy()
    predictions = logits.argmax(axis=1)

    train_indices = np.flatnonzero(data.train_mask.cpu().numpy())
    calibration_indices = np.flatnonzero(data.val_mask.cpu().numpy())
    evaluation_indices = np.flatnonzero(data.test_mask.cpu().numpy())
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
    descriptors = graph_control_descriptors(features, edge_index)
    descriptor_scaler = StandardScaler().fit(descriptors[train_indices])
    controls = descriptor_scaler.transform(descriptors)
    control_head = Ridge(alpha=float(config.get("control_ridge_alpha", 1.0))).fit(
        hidden[train_indices],
        controls[train_indices],
    )
    control_r2 = float(
        control_head.score(hidden[calibration_indices], controls[calibration_indices])
    )
    coefficients = np.asarray(control_head.coef_, dtype=float)

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
            logits[candidate_indices], logits[anchor_indices]
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
            eps = float(point.control_epsilon)
            mask = (
                (control_distances <= eps)
                & (payload_distances >= payload_delta)
                & (behavior_distances >= behavior_gamma)
                & valid_pairs
            )
            points.append(
                {
                    "control_epsilon": eps,
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

    return {
        "artifact_type": "qcollide_graph_functional_collision_atlas_component",
        "schema_version": 1,
        "domain": "graph_learning",
        "dataset": "cora",
        "architecture": architecture,
        "model_seed": model_seed,
        "model_diagnostics": {
            **accuracies,
            "hidden_dimension": int(hidden.shape[1]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "control_head_calibration_r2": control_r2,
        },
        "selection": {
            "anchors_per_class": anchors_per_class,
            "anchor_count": int(len(anchor_indices)),
            "candidate_count": int(len(candidate_indices)),
        },
        "rows": rows,
        "claim_boundary": {
            "control_descriptors_are_label_agnostic": True,
            "thresholds_calibrated_on_validation_nodes_only": True,
            "evaluation_nodes_are_test_mask_only": True,
            "generated_adversarial_graphs": False,
            "quantum_execution": False,
        },
    }


__all__ = ["graph_control_descriptors", "run_graph_atlas_component"]
