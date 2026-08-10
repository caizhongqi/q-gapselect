"""PyTorch models and the real-image Digits dataset for Q-COLLIDE."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The vision experiment requires the optional 'vision' dependencies: "
        "pip install -e '.[vision]'"
    ) from exc


@dataclass(frozen=True)
class VisionData:
    train_images: np.ndarray
    train_labels: np.ndarray
    calibration_images: np.ndarray
    calibration_labels: np.ndarray
    evaluation_images: np.ndarray
    evaluation_labels: np.ndarray
    train_control_targets: np.ndarray
    calibration_control_targets: np.ndarray
    evaluation_control_targets: np.ndarray
    control_mean: np.ndarray
    control_scale: np.ndarray


class ConvControlNet(nn.Module):
    def __init__(self, hidden_dimension: int, control_dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.embedding = nn.Linear(32 * 4 * 4, hidden_dimension)
        self.main_head = nn.Linear(hidden_dimension, 10)
        self.control_head = nn.Linear(hidden_dimension, control_dimension)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images).flatten(1)
        return torch.tanh(self.embedding(hidden))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(images)
        return self.main_head(hidden), self.control_head(hidden), hidden


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_dimension: int,
        control_dimension: int,
        *,
        patch_size: int = 2,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if 8 % patch_size:
            raise ValueError("patch_size must divide 8")
        self.patch_size = patch_size
        patch_count = (8 // patch_size) ** 2
        self.patch_embedding = nn.Linear(patch_size * patch_size, hidden_dimension)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dimension))
        self.position = nn.Parameter(torch.randn(1, patch_count + 1, hidden_dimension) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=heads,
            dim_feedforward=2 * hidden_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.normalization = nn.LayerNorm(hidden_dimension)
        self.main_head = nn.Linear(hidden_dimension, 10)
        self.control_head = nn.Linear(hidden_dimension, control_dimension)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        patch_size = self.patch_size
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(images.shape[0], -1, patch_size * patch_size)
        tokens = self.patch_embedding(patches)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1) + self.position
        tokens = self.transformer(tokens)
        return self.normalization(tokens[:, 0])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(images)
        return self.main_head(hidden), self.control_head(hidden), hidden


def derived_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(("qcollide-real-vision-v1", str(master_seed), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def image_control_targets(images: np.ndarray) -> np.ndarray:
    """Eight low-frequency, label-agnostic image attributes."""

    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 8):
        raise ValueError("images must have shape (n, 8, 8)")
    output = np.zeros((len(values), 8), dtype=np.float32)
    output[:, 0] = values[:, :4, :4].mean(axis=(1, 2))
    output[:, 1] = values[:, :4, 4:].mean(axis=(1, 2))
    output[:, 2] = values[:, 4:, :4].mean(axis=(1, 2))
    output[:, 3] = values[:, 4:, 4:].mean(axis=(1, 2))
    coordinates = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    mass = values.sum(axis=(1, 2)) + 1e-6
    output[:, 4] = (values.sum(axis=1) * coordinates[None, :]).sum(axis=1) / mass
    output[:, 5] = (values.sum(axis=2) * coordinates[None, :]).sum(axis=1) / mass
    output[:, 6] = np.diagonal(values, axis1=1, axis2=2).mean(axis=1) - np.diagonal(
        values[:, :, ::-1], axis1=1, axis2=2
    ).mean(axis=1)
    output[:, 7] = values.mean(axis=(1, 2))
    return output


def load_real_digits(*, dataset_seed: int) -> VisionData:
    dataset = load_digits()
    images = dataset.images.astype(np.float32) / 16.0
    labels = dataset.target.astype(np.int64)
    train_images, held_images, train_labels, held_labels = train_test_split(
        images,
        labels,
        test_size=0.4,
        random_state=dataset_seed,
        stratify=labels,
    )
    calibration_images, evaluation_images, calibration_labels, evaluation_labels = (
        train_test_split(
            held_images,
            held_labels,
            test_size=0.5,
            random_state=dataset_seed + 1,
            stratify=held_labels,
        )
    )
    train_targets_raw = image_control_targets(train_images)
    mean = train_targets_raw.mean(axis=0, keepdims=True)
    scale = train_targets_raw.std(axis=0, keepdims=True) + 1e-6

    def normalize(partition: np.ndarray) -> np.ndarray:
        return (image_control_targets(partition) - mean) / scale

    return VisionData(
        train_images=train_images,
        train_labels=train_labels,
        calibration_images=calibration_images,
        calibration_labels=calibration_labels,
        evaluation_images=evaluation_images,
        evaluation_labels=evaluation_labels,
        train_control_targets=normalize(train_images),
        calibration_control_targets=normalize(calibration_images),
        evaluation_control_targets=normalize(evaluation_images),
        control_mean=mean,
        control_scale=scale,
    )


def make_model(
    architecture: str,
    *,
    hidden_dimension: int,
    control_dimension: int,
) -> nn.Module:
    if architecture == "cnn":
        return ConvControlNet(hidden_dimension, control_dimension)
    if architecture == "tiny_vit":
        return TinyVisionTransformer(hidden_dimension, control_dimension)
    raise ValueError(f"unknown architecture {architecture!r}")


def train_vision_model(
    architecture: str,
    data: VisionData,
    *,
    seed: int,
    hidden_dimension: int,
    control_dimension: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    control_loss_weight: float,
) -> tuple[nn.Module, dict[str, Any]]:
    set_deterministic_seed(seed)
    model = make_model(
        architecture,
        hidden_dimension=hidden_dimension,
        control_dimension=control_dimension,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    images = torch.from_numpy(data.train_images[:, None])
    labels = torch.from_numpy(data.train_labels)
    controls = torch.from_numpy(data.train_control_targets.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(images), generator=generator)
        model.train()
        for start in range(0, len(images), batch_size):
            indices = permutation[start : start + batch_size]
            logits, predicted_control, _ = model(images[indices])
            loss = F.cross_entropy(logits, labels[indices]) + control_loss_weight * F.mse_loss(
                predicted_control,
                controls[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()

    def diagnostics(
        partition_images: np.ndarray,
        partition_labels: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, float]:
        with torch.no_grad():
            logits, predicted_control, _ = model(torch.from_numpy(partition_images[:, None]))
        accuracy = float(np.mean(logits.argmax(dim=1).numpy() == partition_labels))
        control_mse = float(np.mean((predicted_control.numpy() - target) ** 2))
        return accuracy, control_mse

    calibration_accuracy, calibration_control_mse = diagnostics(
        data.calibration_images,
        data.calibration_labels,
        data.calibration_control_targets,
    )
    evaluation_accuracy, evaluation_control_mse = diagnostics(
        data.evaluation_images,
        data.evaluation_labels,
        data.evaluation_control_targets,
    )
    return model, {
        "architecture": architecture,
        "calibration_accuracy": calibration_accuracy,
        "evaluation_accuracy": evaluation_accuracy,
        "calibration_control_mse": calibration_control_mse,
        "evaluation_control_mse": evaluation_control_mse,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
    }


__all__ = [
    "VisionData",
    "derived_seed",
    "image_control_targets",
    "load_real_digits",
    "make_model",
    "set_deterministic_seed",
    "train_vision_model",
]
