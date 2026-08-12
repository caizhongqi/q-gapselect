"""Accuracy-track replay for CIFAR collision-topology checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The CIFAR performance track requires the optional 'vision' dependencies: "
        "pip install -e '.[vision]'"
    ) from exc

from .cifar_models import derived_seed, load_cifar_data, train_cifar_model


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _classification_accuracy(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    correct = 0
    model.eval()
    tensor = torch.from_numpy(images)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            logits, _, _ = model(batch)
            predictions = logits.argmax(dim=1).cpu().numpy()
            expected = labels[start : start + len(batch)]
            correct += int(np.count_nonzero(predictions == expected))
    return correct / len(images)


def run_cifar_performance_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Replay one deterministic training run and evaluate every frozen checkpoint."""

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
    rows: list[dict[str, Any]] = []
    for epoch in checkpoint_epochs:
        model.load_state_dict(checkpoints[epoch])
        model.to(device)
        calibration_accuracy = _classification_accuracy(
            model,
            data.calibration_images,
            data.calibration_labels,
            device=device,
        )
        evaluation_accuracy = _classification_accuracy(
            model,
            data.evaluation_images,
            data.evaluation_labels,
            device=device,
        )
        rows.append(
            {
                "architecture": architecture,
                "model_seed": model_seed,
                "checkpoint_epoch": epoch,
                "calibration_accuracy": calibration_accuracy,
                "evaluation_accuracy": evaluation_accuracy,
            }
        )
    final_row = rows[-1]
    consistency_error = abs(
        float(final_row["evaluation_accuracy"])
        - float(training["evaluation_accuracy"])
    )
    if consistency_error > 1e-12:
        raise RuntimeError(
            "deterministic replay final accuracy does not match training diagnostics"
        )
    return {
        "artifact_type": "qcollide_cifar_checkpoint_performance_component",
        "schema_version": schema_version,
        "master_seed": master_seed,
        "architecture": architecture,
        "model_seed": model_seed,
        "training_seed": training_seed,
        "claim_scope": {
            "dataset": data.dataset_name,
            "deterministic_checkpoint_replay": True,
            "topology_measured_in_this_artifact": False,
            "performance_matching_support": True,
            "architecture_causality_claimed": False,
        },
        "training": {
            **training,
            "model_seed": model_seed,
        },
        "rows": rows,
        "final_accuracy_consistency_error": consistency_error,
    }


__all__ = ["run_cifar_performance_component"]
