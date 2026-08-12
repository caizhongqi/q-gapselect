"""Prepared CIFAR-100 raw-array adapter for deterministic Q-COLLIDE runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PreparedCIFAR100:
    """Minimal ``torchvision.datasets.CIFAR100``-compatible raw-array adapter."""

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool = True,
        download: bool = False,
        **_: object,
    ) -> None:
        del download
        directory = Path(root)
        if train:
            image_path = directory / "train_images.npy"
            label_path = directory / "train_labels.npy"
            expected_images = (50000, 32, 32, 3)
            expected_labels = (50000,)
        else:
            image_path = directory / "evaluation_images.npy"
            label_path = directory / "evaluation_labels.npy"
            expected_images = (10000, 32, 32, 3)
            expected_labels = (10000,)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"prepared CIFAR-100 snapshot not found under {directory}"
            )
        self.data = np.load(image_path, mmap_mode="r")
        labels = np.asarray(np.load(label_path), dtype=np.int64)
        if self.data.shape != expected_images or labels.shape != expected_labels:
            raise ValueError("prepared CIFAR-100 snapshot has an unexpected shape")
        if labels.min(initial=0) < 0 or labels.max(initial=0) >= 100:
            raise ValueError("prepared CIFAR-100 labels are outside [0,100)")
        self.targets = [int(value) for value in labels]


def install_prepared_cifar100_adapter() -> None:
    """Route the standard Q-COLLIDE CIFAR loader to the prepared snapshot."""

    from . import cifar_models

    cifar_models.CIFAR100 = PreparedCIFAR100


__all__ = ["PreparedCIFAR100", "install_prepared_cifar100_adapter"]
