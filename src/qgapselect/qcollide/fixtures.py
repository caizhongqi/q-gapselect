"""Deterministic F0--F4 fixture generators for Q-COLLIDE."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
from math import sqrt

import numpy as np

from .contracts import CollisionCriteria, CollisionInstance, EndpointRecord, Side
from .geometry import tunnel_certificate


def _fingerprint(payload: object) -> str:
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def _record(
    index: int,
    side: Side,
    signature: tuple[int, ...],
    *,
    control: Iterable[float],
    payload: Iterable[float],
    behavior: float,
    prefixes: tuple[tuple[int, ...], ...] = (),
    stage_costs: tuple[float, ...] = (1.0,),
) -> EndpointRecord:
    return EndpointRecord(
        index=index,
        side=side,
        signature=signature,
        prefixes=prefixes,
        control=tuple(float(x) for x in control),
        payload=tuple(float(x) for x in payload),
        behavior=float(behavior),
        stage_costs=stage_costs,
    )


def pair_oracle_negative(n: int, marked_pairs: int, seed: int) -> CollisionInstance:
    """F0: all endpoint records are identical; only the pair oracle carries marks."""

    if n <= 0 or not 1 <= marked_pairs <= n * n:
        raise ValueError("invalid n or marked_pairs")
    rng = np.random.default_rng(seed)
    flat = rng.choice(n * n, size=marked_pairs, replace=False)
    marks = frozenset((int(value // n), int(value % n)) for value in flat)
    left = tuple(
        _record(i, "A", (0,), control=(0.0,), payload=(1.0,), behavior=1.0)
        for i in range(n)
    )
    right = tuple(
        _record(j, "B", (0,), control=(0.0,), payload=(0.0,), behavior=0.0)
        for j in range(n)
    )
    metadata = {"fixture": "F0", "seed": seed, "marked_pairs": marked_pairs}
    metadata["fingerprint"] = _fingerprint((metadata, sorted(marks)))
    return CollisionInstance(
        name=f"f0_pair_oracle_n{n}_k{marked_pairs}_s{seed}",
        left=left,
        right=right,
        criteria=CollisionCriteria(0.0, 0.5, 0.5),
        endpoint_local=False,
        explicit_pair_marks=marks,
        metadata=metadata,
    )


def packed_claw(n: int, matching_size: int, seed: int) -> CollisionInstance:
    """F1: exactly ``matching_size`` vertex-disjoint endpoint-local claws."""

    if n <= 0 or not 1 <= matching_size <= n:
        raise ValueError("invalid n or matching_size")
    rng = np.random.default_rng(seed)
    chosen_left = rng.choice(n, size=matching_size, replace=False)
    chosen_right = rng.choice(n, size=matching_size, replace=False)
    left_labels = [(1, i) for i in range(n)]
    right_labels = [(2, j) for j in range(n)]
    planted: list[tuple[int, int]] = []
    for label, (i, j) in enumerate(zip(chosen_left, chosen_right, strict=True)):
        signature = (0, label)
        left_labels[int(i)] = signature
        right_labels[int(j)] = signature
        planted.append((int(i), int(j)))

    left = tuple(
        _record(i, "A", left_labels[i], control=left_labels[i], payload=(1.0,), behavior=1.0)
        for i in range(n)
    )
    right = tuple(
        _record(j, "B", right_labels[j], control=right_labels[j], payload=(0.0,), behavior=0.0)
        for j in range(n)
    )
    metadata = {
        "fixture": "F1",
        "seed": seed,
        "planted_matching": sorted(planted),
        "matching_size": matching_size,
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    return CollisionInstance(
        name=f"f1_packed_claw_n{n}_nu{matching_size}_s{seed}",
        left=left,
        right=right,
        criteria=CollisionCriteria(0.0, 0.5, 0.5),
        endpoint_local=True,
        metadata=metadata,
    )


def random_range(n: int, range_size: int, seed: int) -> CollisionInstance:
    """F2: two independent random functions into a common discrete range."""

    if n <= 0 or range_size <= 0:
        raise ValueError("n and range_size must be positive")
    rng = np.random.default_rng(seed)
    left_labels = rng.integers(0, range_size, size=n)
    right_labels = rng.integers(0, range_size, size=n)
    left = tuple(
        _record(
            i,
            "A",
            (int(left_labels[i]),),
            control=(float(left_labels[i]),),
            payload=(1.0,),
            behavior=1.0,
        )
        for i in range(n)
    )
    right = tuple(
        _record(
            j,
            "B",
            (int(right_labels[j]),),
            control=(float(right_labels[j]),),
            payload=(0.0,),
            behavior=0.0,
        )
        for j in range(n)
    )
    metadata = {"fixture": "F2", "seed": seed, "range_size": range_size}
    metadata["fingerprint"] = _fingerprint((metadata, left_labels.tolist(), right_labels.tolist()))
    return CollisionInstance(
        name=f"f2_random_range_n{n}_r{range_size}_s{seed}",
        left=left,
        right=right,
        criteria=CollisionCriteria(0.0, 0.5, 0.5),
        endpoint_local=True,
        metadata=metadata,
    )


def _prefixes(
    label: int,
    cumulative_bits: tuple[int, ...],
    total_bits: int,
) -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    for bits in cumulative_bits:
        shift = total_bits - bits
        values.append((int(label >> shift),))
    return tuple(values)


def weighted_prefix_claw(
    n: int,
    matching_size: int,
    cumulative_bits: tuple[int, ...],
    stage_costs: tuple[float, ...],
    seed: int,
) -> CollisionInstance:
    """F3: exact packed claws with progressively revealed signatures."""

    if len(cumulative_bits) != len(stage_costs) or not cumulative_bits:
        raise ValueError("prefix levels and stage costs must be non-empty and aligned")
    if any(a >= b for a, b in zip(cumulative_bits, cumulative_bits[1:], strict=False)):
        raise ValueError("cumulative_bits must be strictly increasing")
    total_bits = cumulative_bits[-1]
    if total_bits > 62:
        raise ValueError("total_bits must not exceed 62")
    if (1 << total_bits) < 4 * n:
        raise ValueError("final signature range is too small to guarantee unique labels")
    base = packed_claw(n, matching_size, seed)
    rng = np.random.default_rng(seed ^ 0xC0111DE)
    labels = rng.choice(1 << total_bits, size=2 * n - matching_size, replace=False)
    planted = base.metadata["planted_matching"]
    left_label = [-1] * n
    right_label = [-1] * n
    cursor = 0
    for i, j in planted:
        value = int(labels[cursor])
        cursor += 1
        left_label[i] = value
        right_label[j] = value
    for i in range(n):
        if left_label[i] == -1:
            left_label[i] = int(labels[cursor])
            cursor += 1
    for j in range(n):
        if right_label[j] == -1:
            right_label[j] = int(labels[cursor])
            cursor += 1

    left = tuple(
        _record(
            i,
            "A",
            (left_label[i],),
            control=(float(left_label[i]),),
            payload=(1.0,),
            behavior=1.0,
            prefixes=_prefixes(left_label[i], cumulative_bits, total_bits),
            stage_costs=stage_costs,
        )
        for i in range(n)
    )
    right = tuple(
        _record(
            j,
            "B",
            (right_label[j],),
            control=(float(right_label[j]),),
            payload=(0.0,),
            behavior=0.0,
            prefixes=_prefixes(right_label[j], cumulative_bits, total_bits),
            stage_costs=stage_costs,
        )
        for j in range(n)
    )
    metadata = {
        "fixture": "F3",
        "seed": seed,
        "matching_size": matching_size,
        "cumulative_bits": cumulative_bits,
        "stage_costs": stage_costs,
        "planted_matching": planted,
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    return CollisionInstance(
        name=f"f3_weighted_prefix_n{n}_nu{matching_size}_s{seed}",
        left=left,
        right=right,
        criteria=CollisionCriteria(0.0, 0.5, 0.5),
        endpoint_local=True,
        metadata=metadata,
    )


def _rank_r_matrix(rng: np.random.Generator, rows: int, columns: int, rank: int) -> np.ndarray:
    if not 0 <= rank <= min(rows, columns):
        raise ValueError("invalid requested rank")
    if rank == 0:
        return np.zeros((rows, columns), dtype=float)
    left, _ = np.linalg.qr(rng.normal(size=(rows, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(columns, rank)))
    singular = np.linspace(1.0, 0.5, rank)
    return left @ np.diag(singular) @ right.T


def geometry_to_packing(
    *,
    manifold_dimension: int,
    control_rank: int,
    openness: float,
    anchors: int,
    control_epsilon: float,
    payload_delta: float,
    behavior_gamma: float,
    control_curvature: float,
    behavior_curvature: float,
    payload_curvature: float,
    local_radius: float,
    seed: int,
) -> CollisionInstance:
    """F4: synthetic local manifolds with explicit geometry certificates."""

    if manifold_dimension <= 0 or not 0 <= control_rank <= manifold_dimension:
        raise ValueError("invalid manifold_dimension or control_rank")
    if not 0.0 <= openness <= 1.0:
        raise ValueError("openness must lie in [0,1]")
    if anchors <= 0:
        raise ValueError("anchors must be positive")

    rng = np.random.default_rng(seed)
    control_dimension = max(1, control_rank)
    left: list[EndpointRecord] = []
    right: list[EndpointRecord] = []
    certificates: list[dict[str, object]] = []

    for anchor in range(anchors):
        a_matrix = _rank_r_matrix(rng, control_dimension, manifold_dimension, control_rank)
        _, _, vh = np.linalg.svd(a_matrix, full_matrices=True)
        row_basis = vh[:control_rank].T
        null_basis = vh[control_rank:].T

        if control_rank > 0:
            row_vector = row_basis @ rng.normal(size=control_rank)
            row_vector /= np.linalg.norm(row_vector)
        else:
            row_vector = np.zeros(manifold_dimension)
        if null_basis.shape[1] > 0:
            null_vector = null_basis @ rng.normal(size=null_basis.shape[1])
            null_vector /= np.linalg.norm(null_vector)
        else:
            null_vector = np.zeros(manifold_dimension)

        if null_basis.shape[1] == 0:
            gradient = row_vector
        elif control_rank == 0:
            gradient = null_vector
        else:
            gradient = sqrt(1.0 - openness * openness) * row_vector + openness * null_vector
        if np.linalg.norm(gradient) == 0.0:
            gradient = np.ones(manifold_dimension) / sqrt(manifold_dimension)

        payload_jacobian = np.eye(manifold_dimension)
        certificate = tunnel_certificate(
            a_matrix,
            gradient,
            payload_jacobian,
            control_epsilon=control_epsilon,
            payload_delta=payload_delta,
            behavior_gamma=behavior_gamma,
            control_curvature=control_curvature,
            behavior_curvature=behavior_curvature,
            payload_curvature=payload_curvature,
            local_radius=local_radius,
        )
        direction = np.asarray(certificate.geometry.direction, dtype=float)
        if certificate.certified:
            step = 0.5 * (certificate.required_step + certificate.safe_step)
        elif np.isfinite(certificate.required_step):
            step = min(certificate.required_step, local_radius)
        else:
            step = local_radius

        z = step * direction
        mu = np.zeros(control_dimension)
        mu[0] = anchor * max(8.0 * control_epsilon, 1.0)
        curve_direction = np.zeros(control_dimension)
        curve_direction[0] = 1.0
        control_attack = (
            mu
            + a_matrix @ z
            + 0.5 * control_curvature * float(z @ z) * curve_direction
        )
        payload_attack = payload_jacobian @ z
        if np.linalg.norm(z) > 0.0 and payload_curvature > 0.0:
            payload_attack -= 0.5 * payload_curvature * float(z @ z) * direction
        behavior_attack = float(gradient @ z - 0.5 * behavior_curvature * float(z @ z))

        signature = (anchor,)
        left.append(
            _record(
                anchor,
                "A",
                signature,
                control=control_attack,
                payload=payload_attack,
                behavior=behavior_attack,
            )
        )
        right.append(
            _record(
                anchor,
                "B",
                signature,
                control=mu,
                payload=np.zeros(manifold_dimension),
                behavior=0.0,
            )
        )
        certificate_payload = asdict(certificate)
        certificate_payload["realized_control_distance"] = float(
            np.linalg.norm(control_attack - mu)
        )
        certificate_payload["realized_payload_distance"] = float(np.linalg.norm(payload_attack))
        certificate_payload["realized_behavior_distance"] = abs(behavior_attack)
        certificates.append(certificate_payload)

    metadata = {
        "fixture": "F4",
        "seed": seed,
        "manifold_dimension": manifold_dimension,
        "control_rank": control_rank,
        "requested_openness": openness,
        "anchors": anchors,
        "certificates": certificates,
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    return CollisionInstance(
        name=(
            f"f4_geometry_m{manifold_dimension}_r{control_rank}_"
            f"eta{openness:.3f}_s{anchors}_seed{seed}"
        ),
        left=tuple(left),
        right=tuple(right),
        criteria=CollisionCriteria(control_epsilon, payload_delta, behavior_gamma),
        endpoint_local=True,
        metadata=metadata,
    )
