"""Anchor preparation and benign-threshold calibration."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jacrev, vmap

from .vision_linear import control_output, residual_hidden, row_basis, standardized_projection
from .vision_types import AnchorPanel, BenignPanel, ProjectionCalibration


def stratified_correct_indices(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
) -> np.ndarray:
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(images[:, None].astype(np.float32)))
    predictions = logits.argmax(dim=1).cpu().numpy()
    selected: list[int] = []
    for label in range(10):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) < per_class:
            raise RuntimeError(
                f"class {label} has only {len(candidates)} correctly classified examples; "
                f"{per_class} are required"
            )
        selected.extend(int(value) for value in candidates[:per_class])
    return np.asarray(selected, dtype=np.int64)


def prepare_anchor_panel(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> AnchorPanel:
    selected_images = images[indices].astype(np.float32)
    tensor = torch.from_numpy(selected_images[:, None])

    def encode_single(image: torch.Tensor) -> torch.Tensor:
        return model.encode(image.unsqueeze(0)).squeeze(0)

    hidden_jacobians = (
        vmap(jacrev(encode_single))(tensor)
        .detach()
        .cpu()
        .numpy()
        .reshape(len(indices), -1, 64)
    )
    with torch.no_grad():
        logits, _, hidden = model(tensor)
    logits_array = logits.cpu().numpy()
    selected_labels = labels[indices].astype(np.int64)
    targets: list[int] = []
    for row, true_label in zip(logits_array, selected_labels, strict=True):
        order = np.argsort(row)[::-1]
        targets.append(next(int(label) for label in order if label != int(true_label)))
    return AnchorPanel(
        images=selected_images,
        labels=selected_labels,
        hidden=hidden.cpu().numpy(),
        logits=logits_array,
        hidden_jacobians=hidden_jacobians,
        targets=np.asarray(targets, dtype=np.int64),
    )


def make_benign_panel(
    model: torch.nn.Module,
    anchors: AnchorPanel,
    *,
    seed: int,
    repetitions: int,
    radius: float,
) -> BenignPanel:
    rng = np.random.default_rng(seed)
    perturbed: list[np.ndarray] = []
    repeated_anchor_hidden: list[np.ndarray] = []
    for image, hidden in zip(anchors.images, anchors.hidden, strict=True):
        flattened = image.reshape(-1)
        for _ in range(repetitions):
            direction = rng.normal(size=64)
            direction /= np.linalg.norm(direction) + 1e-12
            candidate = np.clip(flattened + radius * direction, 0.0, 1.0)
            perturbed.append(candidate.reshape(8, 8).astype(np.float32))
            repeated_anchor_hidden.append(hidden)
    with torch.no_grad():
        _, _, perturbed_hidden = model(torch.from_numpy(np.asarray(perturbed)[:, None]))
    return BenignPanel(
        anchor_hidden=np.asarray(repeated_anchor_hidden, dtype=float),
        perturbed_hidden=perturbed_hidden.cpu().numpy(),
    )


def calibrate_projection(
    projection: np.ndarray,
    hidden_support: np.ndarray,
    benign: BenignPanel,
    *,
    control_quantile: float,
    payload_quantile: float,
) -> ProjectionCalibration:
    standardized = standardized_projection(projection, hidden_support)
    control_distances = np.linalg.norm(
        control_output(benign.perturbed_hidden, standardized)
        - control_output(benign.anchor_hidden, standardized),
        axis=1,
    )
    payload_distances = np.linalg.norm(
        residual_hidden(benign.perturbed_hidden, projection)
        - residual_hidden(benign.anchor_hidden, projection),
        axis=1,
    )
    control_epsilon = float(np.quantile(control_distances, control_quantile))
    payload_delta = float(np.quantile(payload_distances, payload_quantile))
    return ProjectionCalibration(
        projection=row_basis(projection),
        standardized_projection=standardized,
        control_epsilon=control_epsilon,
        payload_delta=payload_delta,
        control_acceptance=float(np.mean(control_distances <= control_epsilon)),
        payload_exceedance=float(np.mean(payload_distances >= payload_delta)),
    )


__all__ = [
    "calibrate_projection",
    "make_benign_panel",
    "prepare_anchor_panel",
    "stratified_correct_indices",
]
