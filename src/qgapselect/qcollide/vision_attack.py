"""Adaptive local attacks and exact collision packing for real images."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import sqrt

import numpy as np
import torch

from .topology import collision_topology_metrics
from .vision_linear import control_output, nullspace_basis, residual_hidden
from .vision_types import AnchorPanel, AttackCandidate, PackingResult, ProjectionCalibration


def local_geometry(
    model: torch.nn.Module,
    panel: AnchorPanel,
    index: int,
    calibration: ProjectionCalibration,
) -> tuple[np.ndarray, float, int]:
    main_weight = model.main_head.weight.detach().cpu().numpy()
    true_label = int(panel.labels[index])
    target_label = int(panel.targets[index])
    hidden_jacobian = panel.hidden_jacobians[index]
    behavior_gradient = (main_weight[target_label] - main_weight[true_label]) @ hidden_jacobian
    control_jacobian = (
        calibration.standardized_projection @ hidden_jacobian
        / sqrt(calibration.standardized_projection.shape[0])
    )
    nullspace = nullspace_basis(control_jacobian)
    projected = (
        nullspace @ (nullspace.T @ behavior_gradient)
        if nullspace.shape[1]
        else np.zeros_like(behavior_gradient)
    )
    projected_norm = float(np.linalg.norm(projected))
    total_norm = float(np.linalg.norm(behavior_gradient))
    openness = projected_norm / total_norm if total_norm > 0.0 else 0.0
    direction = projected / projected_norm if projected_norm > 1e-10 else np.zeros_like(projected)
    return direction, float(openness), int(nullspace.shape[1])


def generate_attacks(
    model: torch.nn.Module,
    anchors: AnchorPanel,
    calibration: ProjectionCalibration,
    *,
    maximum_radius: float,
    line_search_steps: int,
    target_margin: float,
) -> tuple[AttackCandidate | None, ...]:
    directions: list[np.ndarray] = []
    openness_values: list[float] = []
    tunnel_dimensions: list[int] = []
    for index in range(anchors.size):
        direction, openness, tunnel_dimension = local_geometry(
            model,
            anchors,
            index,
            calibration,
        )
        directions.append(direction)
        openness_values.append(openness)
        tunnel_dimensions.append(tunnel_dimension)

    image_shape = anchors.images.shape[1:]
    input_dimension = int(np.prod(image_shape))
    scales = np.linspace(0.025, maximum_radius, line_search_steps, dtype=np.float32)
    flattened = anchors.images.reshape(anchors.size, input_dimension)
    direction_matrix = np.asarray(directions, dtype=np.float32)
    candidate_images = np.clip(
        flattened[:, None, :] + scales[None, :, None] * direction_matrix[:, None, :],
        0.0,
        1.0,
    ).reshape(-1, 1, *image_shape)
    logits_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, len(candidate_images), 2048):
            tensor = torch.from_numpy(candidate_images[start_index : start_index + 2048])
            logits, _, hidden = model(tensor)
            logits_parts.append(logits.cpu().numpy())
            hidden_parts.append(hidden.cpu().numpy())
    all_logits = np.concatenate(logits_parts, axis=0).reshape(
        anchors.size,
        line_search_steps,
        -1,
    )
    all_hidden = np.concatenate(hidden_parts, axis=0).reshape(
        anchors.size,
        line_search_steps,
        -1,
    )
    candidate_images_view = candidate_images.reshape(
        anchors.size,
        line_search_steps,
        *image_shape,
    )

    source_control = control_output(anchors.hidden, calibration.standardized_projection)
    source_payload = residual_hidden(anchors.hidden, calibration.projection)
    attacks: list[AttackCandidate | None] = []
    for index in range(anchors.size):
        direction = direction_matrix[index]
        if np.linalg.norm(direction) <= 1e-10:
            attacks.append(None)
            continue
        true_label = int(anchors.labels[index])
        target_label = int(anchors.targets[index])
        logits = all_logits[index]
        hidden = all_hidden[index]
        target_gaps = logits[:, target_label] - logits[:, true_label]
        controls = control_output(hidden, calibration.standardized_projection)
        payloads = residual_hidden(hidden, calibration.projection)
        control_distances = np.linalg.norm(
            controls - source_control[index][None, :],
            axis=1,
        )
        payload_distances = np.linalg.norm(
            payloads - source_payload[index][None, :],
            axis=1,
        )
        accepted = np.flatnonzero(
            (target_gaps >= target_margin)
            & (control_distances <= calibration.control_epsilon)
            & (payload_distances >= calibration.payload_delta)
        )
        if len(accepted) == 0:
            attacks.append(None)
            continue
        selected = int(accepted[0])
        candidate_image = candidate_images_view[index, selected]
        attacks.append(
            AttackCandidate(
                source_index=index,
                source_label=true_label,
                target_label=target_label,
                image=candidate_image.astype(np.float32),
                hidden=hidden[selected],
                logits=logits[selected],
                input_l2=float(
                    np.linalg.norm(candidate_image.reshape(-1) - flattened[index])
                ),
                source_control_distance=float(control_distances[selected]),
                source_payload_distance=float(payload_distances[selected]),
                openness=openness_values[index],
                tunnel_dimension=tunnel_dimensions[index],
            )
        )
    return tuple(attacks)


def maximum_bipartite_matching(adjacency: tuple[tuple[int, ...], ...], n_right: int) -> int:
    n_left = len(adjacency)
    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    distance = [0] * n_left
    infinity = n_left + n_right + 1

    def breadth_first() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in range(n_left):
            if pair_left[left] == -1:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = infinity
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                mate = pair_right[right]
                if mate == -1:
                    found = True
                elif distance[mate] == infinity:
                    distance[mate] = distance[left] + 1
                    queue.append(mate)
        return found

    def depth_first(left: int) -> bool:
        for right in adjacency[left]:
            mate = pair_right[right]
            if mate == -1 or (
                distance[mate] == distance[left] + 1 and depth_first(mate)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        distance[left] = infinity
        return False

    matching = 0
    while breadth_first():
        for left in range(n_left):
            if pair_left[left] == -1 and depth_first(left):
                matching += 1
    return matching


def evaluate_packing(
    attacks: Sequence[AttackCandidate | None],
    anchors: AnchorPanel,
    calibration: ProjectionCalibration,
) -> PackingResult:
    anchor_control = control_output(anchors.hidden, calibration.standardized_projection)
    anchor_payload = residual_hidden(anchors.hidden, calibration.projection)
    adjacency: list[tuple[int, ...]] = []
    edge_vectors: list[np.ndarray] = []
    input_distances: list[float] = []
    valid_count = 0
    for attack in attacks:
        edges: list[int] = []
        if attack is not None:
            valid_count += 1
            input_distances.append(attack.input_l2)
            attack_control = control_output(attack.hidden, calibration.standardized_projection)
            attack_payload = residual_hidden(attack.hidden, calibration.projection)
            prediction = int(np.argmax(attack.logits))
            for anchor_index in range(anchors.size):
                if int(anchors.labels[anchor_index]) != attack.source_label:
                    continue
                if prediction == int(anchors.labels[anchor_index]):
                    continue
                control_distance = float(
                    np.linalg.norm(attack_control - anchor_control[anchor_index])
                )
                payload_vector = attack_payload - anchor_payload[anchor_index]
                payload_distance = float(np.linalg.norm(payload_vector))
                if (
                    control_distance <= calibration.control_epsilon
                    and payload_distance >= calibration.payload_delta
                ):
                    edges.append(anchor_index)
                    edge_vectors.append(payload_vector)
        adjacency.append(tuple(edges))
    canonical_adjacency = tuple(adjacency)
    matching = maximum_bipartite_matching(canonical_adjacency, anchors.size)
    vector_matrix = (
        np.stack(edge_vectors, axis=0)
        if edge_vectors
        else np.empty((0, anchor_payload.shape[1]), dtype=float)
    )
    topology = collision_topology_metrics(
        canonical_adjacency,
        anchors.size,
        edge_vectors=vector_matrix,
        matching_size=matching,
    )
    return PackingResult(
        edge_count=topology.edge_count,
        matching_size=matching,
        packing_fraction=matching / anchors.size,
        candidate_fraction=valid_count / len(attacks),
        mean_input_l2=float(np.mean(input_distances)) if input_distances else None,
        topology=topology,
    )


__all__ = [
    "evaluate_packing",
    "generate_attacks",
    "local_geometry",
    "maximum_bipartite_matching",
]
