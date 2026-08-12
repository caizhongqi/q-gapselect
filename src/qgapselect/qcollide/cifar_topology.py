"""CIFAR functional-collision topology and persistent-filtration campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The CIFAR topology experiment requires the optional 'vision' dependencies: "
        "pip install -e '.[vision]'"
    ) from exc

from .cifar_models import (
    derived_seed,
    load_cifar_data,
    train_cifar_model,
)
from .topology_persistence import collision_filtration_profile
from .vision_linear import (
    control_output,
    nullspace_basis,
    residual_hidden,
    standardized_projection,
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _integer_tuple(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    parser = _nonnegative_int if allow_zero else _positive_int
    output = tuple(
        parser(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    )
    if not output:
        raise ValueError(f"{name} must be non-empty")
    return output


def _float_tuple(value: object, name: str, *, minimum: float = 0.0) -> tuple[float, ...]:
    output = tuple(
        _number(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, name))
    )
    if not output:
        raise ValueError(f"{name} must be non-empty")
    return output


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _encode(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(images)
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            logits, _, hidden = model(batch)
            logits_parts.append(logits.cpu())
            hidden_parts.append(hidden.cpu())
    return (
        torch.cat(hidden_parts, dim=0).numpy(),
        torch.cat(logits_parts, dim=0).numpy(),
    )


def _stratified_correct_indices(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    per_class: int,
) -> np.ndarray:
    predictions = np.asarray(logits).argmax(axis=1)
    selected: list[int] = []
    for label in range(class_count):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) < per_class:
            raise RuntimeError(
                f"class {label} has only {len(candidates)} correctly classified examples; "
                f"{per_class} are required"
            )
        selected.extend(int(value) for value in candidates[:per_class])
    return np.asarray(selected, dtype=np.int64)


def _visible_projection(model: torch.nn.Module, visible_rank: int) -> np.ndarray:
    control_weight = model.control_head.weight.detach().cpu().numpy()
    _, singular_values, vh = np.linalg.svd(control_weight, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    numerical_rank = int(np.count_nonzero(singular_values > 1e-9 * scale))
    if not 1 <= visible_rank <= numerical_rank:
        raise ValueError(
            f"visible_rank={visible_rank} exceeds numerical control rank {numerical_rank}"
        )
    return vh[:visible_rank].copy()


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    return np.linalg.norm(difference, axis=2)


def _hidden_openness(
    model: torch.nn.Module,
    projection: np.ndarray,
    anchor_logits: np.ndarray,
    anchor_labels: np.ndarray,
) -> tuple[float, int]:
    basis = nullspace_basis(projection)
    gradients: list[np.ndarray] = []
    main_weight = model.main_head.weight.detach().cpu().numpy()
    for logits, label in zip(anchor_logits, anchor_labels, strict=True):
        order = np.argsort(logits)[::-1]
        target = next(int(value) for value in order if int(value) != int(label))
        gradients.append(main_weight[target] - main_weight[int(label)])
    ratios: list[float] = []
    for gradient in gradients:
        norm = float(np.linalg.norm(gradient))
        projected = basis @ (basis.T @ gradient) if basis.size else np.zeros_like(gradient)
        ratios.append(float(np.linalg.norm(projected)) / norm if norm > 0.0 else 0.0)
    return float(np.mean(ratios)), int(basis.shape[1])


def _make_benign_images(
    anchor_images: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perturbed: list[np.ndarray] = []
    anchor_indices: list[int] = []
    for index, image in enumerate(anchor_images):
        for _ in range(repetitions):
            noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
            perturbed.append(np.clip(image + noise, 0.0, 1.0))
            anchor_indices.append(index)
    return np.asarray(perturbed, dtype=np.float32), np.asarray(anchor_indices, dtype=np.int64)


def _profile_row(
    model: torch.nn.Module,
    *,
    architecture: str,
    model_seed: int,
    checkpoint_epoch: int,
    visible_rank: int,
    support_hidden: np.ndarray,
    anchor_hidden: np.ndarray,
    anchor_logits: np.ndarray,
    anchor_labels: np.ndarray,
    candidate_hidden: np.ndarray,
    candidate_logits: np.ndarray,
    candidate_labels: np.ndarray,
    benign_hidden: np.ndarray,
    benign_logits: np.ndarray,
    benign_anchor_indices: np.ndarray,
    epsilon_multipliers: tuple[float, ...],
    control_quantile: float,
    payload_quantile: float,
    behavior_quantile: float,
) -> dict[str, Any]:
    projection = _visible_projection(model, visible_rank)
    standardized = standardized_projection(projection, support_hidden)
    anchor_control = control_output(anchor_hidden, standardized)
    candidate_control = control_output(candidate_hidden, standardized)
    benign_control = control_output(benign_hidden, standardized)
    benign_anchor_control = anchor_control[benign_anchor_indices]
    benign_control_distances = np.linalg.norm(
        benign_control - benign_anchor_control,
        axis=1,
    )
    nominal_epsilon = max(
        float(np.quantile(benign_control_distances, control_quantile)),
        1e-8,
    )
    thresholds = tuple(
        sorted({max(1e-10, multiplier * nominal_epsilon) for multiplier in epsilon_multipliers})
    )

    anchor_residual = residual_hidden(anchor_hidden, projection)
    candidate_residual = residual_hidden(candidate_hidden, projection)
    benign_residual = residual_hidden(benign_hidden, projection)
    benign_anchor_residual = anchor_residual[benign_anchor_indices]
    benign_payload = np.linalg.norm(benign_residual - benign_anchor_residual, axis=1)
    payload_delta = float(np.quantile(benign_payload, payload_quantile))

    benign_behavior = np.linalg.norm(
        benign_logits - anchor_logits[benign_anchor_indices],
        axis=1,
    ) / sqrt(anchor_logits.shape[1])
    behavior_gamma = float(np.quantile(benign_behavior, behavior_quantile))

    control_distances = _pairwise_distances(candidate_control, anchor_control)
    payload_distances = _pairwise_distances(candidate_residual, anchor_residual)
    behavior_distances = _pairwise_distances(candidate_logits, anchor_logits) / sqrt(
        anchor_logits.shape[1]
    )
    candidate_predictions = candidate_logits.argmax(axis=1)
    anchor_predictions = anchor_logits.argmax(axis=1)
    valid_pairs = (
        (candidate_labels[:, None] != anchor_labels[None, :])
        & (candidate_predictions[:, None] != anchor_predictions[None, :])
    )
    edge_displacements = (
        candidate_residual[:, None, :] - anchor_residual[None, :, :]
    )
    profile = collision_filtration_profile(
        control_distances,
        payload_distances,
        behavior_distances,
        thresholds,
        nominal_epsilon=nominal_epsilon,
        payload_delta=payload_delta,
        behavior_gamma=behavior_gamma,
        valid_pairs=valid_pairs,
        edge_displacements=edge_displacements,
    )
    openness, tunnel_dimension = _hidden_openness(
        model,
        projection,
        anchor_logits,
        anchor_labels,
    )
    point_payload = [
        {
            "control_epsilon": point.control_epsilon,
            **asdict(point.metrics),
        }
        for point in profile.points
    ]
    return {
        "architecture": architecture,
        "model_seed": model_seed,
        "checkpoint_epoch": checkpoint_epoch,
        "visible_rank": visible_rank,
        "mean_hidden_openness": openness,
        "hidden_tunnel_dimension": tunnel_dimension,
        "nominal_epsilon": nominal_epsilon,
        "payload_delta": payload_delta,
        "behavior_gamma": behavior_gamma,
        "control_epsilon_multipliers": list(epsilon_multipliers),
        "filtration_summary": asdict(profile.summary),
        "basin_persistence": asdict(profile.basin_persistence),
        "filtration_points": point_payload,
        "valid_pair_fraction": float(np.mean(valid_pairs)),
    }


def run_cifar_topology_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train one CIFAR model and emit final plus checkpoint filtration profiles."""

    schema_version = _positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("only schema_version=1 is supported")
    master_seed = _positive_int(config.get("master_seed"), "master_seed")
    allowed_architectures = tuple(
        str(item) for item in _sequence(config.get("architectures"), "architectures")
    )
    if architecture not in allowed_architectures:
        raise ValueError(f"architecture {architecture!r} is not configured")
    configured_seeds = _integer_tuple(config.get("model_seeds"), "model_seeds", allow_zero=True)
    if model_seed not in configured_seeds:
        raise ValueError(f"model_seed={model_seed} is not configured")

    epochs = _positive_int(config.get("epochs"), "epochs")
    checkpoint_epochs = _integer_tuple(
        config.get("checkpoint_epochs"),
        "checkpoint_epochs",
        allow_zero=True,
    )
    if checkpoint_epochs[0] != 0 or checkpoint_epochs[-1] != epochs:
        raise ValueError("checkpoint_epochs must start at 0 and end at epochs")
    if tuple(sorted(set(checkpoint_epochs))) != checkpoint_epochs:
        raise ValueError("checkpoint_epochs must be strictly increasing")
    visible_ranks = _integer_tuple(config.get("visible_ranks"), "visible_ranks")
    dynamics_ranks = _integer_tuple(
        config.get("dynamics_visible_ranks"),
        "dynamics_visible_ranks",
    )
    if any(rank not in visible_ranks for rank in dynamics_ranks):
        raise ValueError("dynamics_visible_ranks must be a subset of visible_ranks")

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
    final_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    _, final_calibration_logits = _encode(
        model,
        data.calibration_images,
        device=device,
    )
    _, final_evaluation_logits = _encode(
        model,
        data.evaluation_images,
        device=device,
    )
    per_class = _positive_int(config.get("anchors_per_class"), "anchors_per_class")
    anchor_indices = _stratified_correct_indices(
        final_calibration_logits,
        data.calibration_labels,
        class_count=data.class_count,
        per_class=per_class,
    )
    candidate_indices = _stratified_correct_indices(
        final_evaluation_logits,
        data.evaluation_labels,
        class_count=data.class_count,
        per_class=per_class,
    )
    support_count = _positive_int(config.get("support_samples"), "support_samples")
    if support_count > len(data.train_images):
        raise ValueError("support_samples exceeds training partition")
    support_rng = np.random.default_rng(
        derived_seed(master_seed, architecture, model_seed, "support")
    )
    support_indices = np.sort(
        support_rng.choice(len(data.train_images), size=support_count, replace=False)
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

    epsilon_multipliers = _float_tuple(
        config.get("epsilon_multipliers"),
        "epsilon_multipliers",
        minimum=0.0,
    )
    if tuple(sorted(set(epsilon_multipliers))) != epsilon_multipliers:
        raise ValueError("epsilon_multipliers must be strictly increasing")
    control_quantile = _number(config.get("control_quantile"), "control_quantile")
    payload_quantile = _number(config.get("payload_quantile"), "payload_quantile")
    behavior_quantile = _number(config.get("behavior_quantile"), "behavior_quantile")
    for name, value in (
        ("control_quantile", control_quantile),
        ("payload_quantile", payload_quantile),
        ("behavior_quantile", behavior_quantile),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0,1)")

    rows: list[dict[str, Any]] = []
    for checkpoint_epoch in checkpoint_epochs:
        model.load_state_dict(checkpoints[checkpoint_epoch])
        model.to(device)
        model.eval()
        support_hidden, _ = _encode(
            model,
            data.train_images[support_indices],
            device=device,
        )
        anchor_hidden, anchor_logits = _encode(
            model,
            data.calibration_images[anchor_indices],
            device=device,
        )
        candidate_hidden, candidate_logits = _encode(
            model,
            data.evaluation_images[candidate_indices],
            device=device,
        )
        benign_hidden, benign_logits = _encode(model, benign_images, device=device)
        ranks = visible_ranks if checkpoint_epoch == epochs else dynamics_ranks
        for visible_rank in ranks:
            rows.append(
                _profile_row(
                    model,
                    architecture=architecture,
                    model_seed=model_seed,
                    checkpoint_epoch=checkpoint_epoch,
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
                    control_quantile=control_quantile,
                    payload_quantile=payload_quantile,
                    behavior_quantile=behavior_quantile,
                )
            )
    model.load_state_dict(final_state)
    return {
        "artifact_type": "qcollide_cifar_functional_collision_topology_component",
        "schema_version": schema_version,
        "master_seed": master_seed,
        "architecture": architecture,
        "model_seed": model_seed,
        "claim_scope": {
            "dataset": data.dataset_name,
            "natural_cross_class_collision_graph": True,
            "post_intervention_attack_regeneration": False,
            "persistent_control_threshold_filtration": True,
            "training_checkpoint_dynamics": True,
            "quantum_values_are": "analytic structural inputs only",
            "coherent_quantum_execution": False,
            "hardware_runtime_advantage": False,
        },
        "training": training,
        "selection": {
            "anchors_per_class": per_class,
            "anchor_count": len(anchor_indices),
            "candidate_count": len(candidate_indices),
            "support_count": support_count,
            "anchor_indices": anchor_indices.tolist(),
            "candidate_indices": candidate_indices.tolist(),
        },
        "rows": rows,
    }


__all__ = ["run_cifar_topology_component"]
