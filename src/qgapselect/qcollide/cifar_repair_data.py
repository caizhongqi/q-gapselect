"""Prepared CIFAR-10 data loading for the prospective repair experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .cifar_models import CIFARData, _stratified_take, cifar_control_targets


def load_prepared_cifar10_data(
    *,
    prepared_directory: str | Path,
    dataset_seed: int,
    train_samples: int | None,
    calibration_samples: int,
    evaluation_samples: int | None,
    control_grid_size: int = 4,
) -> CIFARData:
    """Reproduce the standard CIFAR partition logic from a frozen raw snapshot."""

    root = Path(prepared_directory)
    required = {
        "train_images": root / "train_images.npy",
        "train_labels": root / "train_labels.npy",
        "evaluation_images": root / "evaluation_images.npy",
        "evaluation_labels": root / "evaluation_labels.npy",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared CIFAR-10 snapshot is missing: {missing}")

    train_uint8 = np.load(required["train_images"], mmap_mode="r")
    labels_all = np.asarray(np.load(required["train_labels"]), dtype=np.int64)
    evaluation_uint8 = np.load(required["evaluation_images"], mmap_mode="r")
    evaluation_labels_all = np.asarray(
        np.load(required["evaluation_labels"]),
        dtype=np.int64,
    )
    if train_uint8.shape != (50000, 32, 32, 3):
        raise ValueError(f"unexpected CIFAR-10 train shape {train_uint8.shape}")
    if evaluation_uint8.shape != (10000, 32, 32, 3):
        raise ValueError(
            f"unexpected CIFAR-10 evaluation shape {evaluation_uint8.shape}"
        )
    if labels_all.shape != (50000,) or evaluation_labels_all.shape != (10000,):
        raise ValueError("unexpected CIFAR-10 label shape")
    if not 10 <= calibration_samples < len(train_uint8):
        raise ValueError("calibration_samples must lie in [10, training-size)")

    train_pool, calibration_indices = train_test_split(
        np.arange(len(train_uint8)),
        test_size=calibration_samples,
        random_state=dataset_seed,
        stratify=labels_all,
    )
    train_local = _stratified_take(
        labels_all[train_pool],
        train_samples,
        dataset_seed + 1,
    )
    evaluation_indices = _stratified_take(
        evaluation_labels_all,
        evaluation_samples,
        dataset_seed + 2,
    )
    train_indices = train_pool[train_local]

    def images(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values[indices], dtype=np.float32)
        return selected.transpose(0, 3, 1, 2) / 255.0

    train_images = images(train_indices, train_uint8)
    calibration_images = images(calibration_indices, train_uint8)
    evaluation_images = images(evaluation_indices, evaluation_uint8)
    raw_controls = cifar_control_targets(train_images, control_grid_size)
    mean = raw_controls.mean(axis=0, keepdims=True).astype(np.float32)
    scale = (raw_controls.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return CIFARData(
        dataset_name="CIFAR-10",
        class_count=10,
        train_images=train_images,
        train_labels=labels_all[train_indices],
        calibration_images=calibration_images,
        calibration_labels=labels_all[calibration_indices],
        evaluation_images=evaluation_images,
        evaluation_labels=evaluation_labels_all[evaluation_indices],
        control_mean=mean,
        control_scale=scale,
    )


__all__ = ["load_prepared_cifar10_data"]
