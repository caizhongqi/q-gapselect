"""PyTorch models and scalable real-image datasets for Q-COLLIDE."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
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
    dataset_name: str
    image_size: int
    class_count: int
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
    def __init__(
        self,
        hidden_dimension: int,
        control_dimension: int,
        *,
        class_count: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.embedding = nn.Linear(32 * 4 * 4, hidden_dimension)
        self.main_head = nn.Linear(hidden_dimension, class_count)
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
        image_size: int,
        class_count: int,
        patch_size: int | None = None,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        chosen_patch = patch_size or (2 if image_size <= 8 else 4)
        if image_size % chosen_patch:
            raise ValueError("patch_size must divide image_size")
        if hidden_dimension % heads:
            raise ValueError("hidden_dimension must be divisible by heads")
        self.image_size = image_size
        self.patch_size = chosen_patch
        patch_count = (image_size // chosen_patch) ** 2
        self.patch_embedding = nn.Linear(chosen_patch * chosen_patch, hidden_dimension)
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
        self.main_head = nn.Linear(hidden_dimension, class_count)
        self.control_head = nn.Linear(hidden_dimension, control_dimension)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("input image size does not match the model")
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
    """Eight low-frequency, label-agnostic attributes for square grayscale images."""

    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("images must have shape (n, side, side)")
    side = values.shape[1]
    midpoint = side // 2
    if midpoint == 0:
        raise ValueError("image side must be at least two")
    output = np.zeros((len(values), 8), dtype=np.float32)
    output[:, 0] = values[:, :midpoint, :midpoint].mean(axis=(1, 2))
    output[:, 1] = values[:, :midpoint, midpoint:].mean(axis=(1, 2))
    output[:, 2] = values[:, midpoint:, :midpoint].mean(axis=(1, 2))
    output[:, 3] = values[:, midpoint:, midpoint:].mean(axis=(1, 2))
    coordinates = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    mass = values.sum(axis=(1, 2)) + 1e-6
    output[:, 4] = (values.sum(axis=1) * coordinates[None, :]).sum(axis=1) / mass
    output[:, 5] = (values.sum(axis=2) * coordinates[None, :]).sum(axis=1) / mass
    indices = np.arange(side)
    output[:, 6] = values[:, indices, indices].mean(axis=1) - values[
        :, indices, side - 1 - indices
    ].mean(axis=1)
    output[:, 7] = values.mean(axis=(1, 2))
    return output


def _stratified_take(labels: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    if count is None or count >= len(indices):
        return indices
    class_count = len(np.unique(labels))
    if count < class_count:
        raise ValueError("sample limit must include at least one item per class")
    selected, _ = train_test_split(
        indices,
        train_size=count,
        random_state=seed,
        stratify=labels,
    )
    return np.sort(selected)


def _assemble_data(
    *,
    dataset_name: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    calibration_images: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_images: np.ndarray,
    evaluation_labels: np.ndarray,
) -> VisionData:
    train_targets_raw = image_control_targets(train_images)
    mean = train_targets_raw.mean(axis=0, keepdims=True)
    scale = train_targets_raw.std(axis=0, keepdims=True) + 1e-6

    def normalize(partition: np.ndarray) -> np.ndarray:
        return (image_control_targets(partition) - mean) / scale

    return VisionData(
        dataset_name=dataset_name,
        image_size=int(train_images.shape[1]),
        class_count=int(len(np.unique(train_labels))),
        train_images=train_images.astype(np.float32),
        train_labels=train_labels.astype(np.int64),
        calibration_images=calibration_images.astype(np.float32),
        calibration_labels=calibration_labels.astype(np.int64),
        evaluation_images=evaluation_images.astype(np.float32),
        evaluation_labels=evaluation_labels.astype(np.int64),
        train_control_targets=normalize(train_images).astype(np.float32),
        calibration_control_targets=normalize(calibration_images).astype(np.float32),
        evaluation_control_targets=normalize(evaluation_images).astype(np.float32),
        control_mean=mean,
        control_scale=scale,
    )


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
    return _assemble_data(
        dataset_name="scikit-learn digits",
        train_images=train_images,
        train_labels=train_labels,
        calibration_images=calibration_images,
        calibration_labels=calibration_labels,
        evaluation_images=evaluation_images,
        evaluation_labels=evaluation_labels,
    )


def load_fashion_mnist(
    *,
    dataset_seed: int,
    data_root: str | Path,
    train_samples: int | None,
    calibration_samples: int | None,
    evaluation_samples: int | None,
) -> VisionData:
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "Fashion-MNIST requires torchvision: pip install -e '.[vision]'"
        ) from exc

    root = Path(data_root)
    training = FashionMNIST(root=root, train=True, download=True)
    evaluation = FashionMNIST(root=root, train=False, download=True)
    full_train_images = training.data.numpy().astype(np.float32) / 255.0
    full_train_labels = np.asarray(training.targets, dtype=np.int64)
    evaluation_images_all = evaluation.data.numpy().astype(np.float32) / 255.0
    evaluation_labels_all = np.asarray(evaluation.targets, dtype=np.int64)

    calibration_count = calibration_samples or 10000
    if not 10 <= calibration_count < len(full_train_images):
        raise ValueError("calibration_samples must lie in [10, training-size)")
    train_pool_indices, calibration_indices = train_test_split(
        np.arange(len(full_train_images)),
        test_size=calibration_count,
        random_state=dataset_seed,
        stratify=full_train_labels,
    )
    train_local = _stratified_take(
        full_train_labels[train_pool_indices],
        train_samples,
        dataset_seed + 1,
    )
    evaluation_indices = _stratified_take(
        evaluation_labels_all,
        evaluation_samples,
        dataset_seed + 2,
    )
    train_indices = train_pool_indices[train_local]
    return _assemble_data(
        dataset_name="Fashion-MNIST",
        train_images=full_train_images[train_indices],
        train_labels=full_train_labels[train_indices],
        calibration_images=full_train_images[calibration_indices],
        calibration_labels=full_train_labels[calibration_indices],
        evaluation_images=evaluation_images_all[evaluation_indices],
        evaluation_labels=evaluation_labels_all[evaluation_indices],
    )


def load_real_vision_dataset(
    *,
    dataset: str,
    dataset_seed: int,
    data_root: str | Path = ".cache/qcollide-data",
    train_samples: int | None = None,
    calibration_samples: int | None = None,
    evaluation_samples: int | None = None,
) -> VisionData:
    normalized = dataset.strip().lower().replace("-", "_")
    if normalized in {"digits", "sklearn_digits", "scikit_learn_digits"}:
        return load_real_digits(dataset_seed=dataset_seed)
    if normalized in {"fashion_mnist", "fashionmnist"}:
        return load_fashion_mnist(
            dataset_seed=dataset_seed,
            data_root=data_root,
            train_samples=train_samples,
            calibration_samples=calibration_samples,
            evaluation_samples=evaluation_samples,
        )
    raise ValueError(f"unknown real-vision dataset {dataset!r}")


def make_model(
    architecture: str,
    *,
    hidden_dimension: int,
    control_dimension: int,
    image_size: int,
    class_count: int,
) -> nn.Module:
    if architecture == "cnn":
        return ConvControlNet(
            hidden_dimension,
            control_dimension,
            class_count=class_count,
        )
    if architecture == "tiny_vit":
        return TinyVisionTransformer(
            hidden_dimension,
            control_dimension,
            image_size=image_size,
            class_count=class_count,
        )
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
        image_size=data.image_size,
        class_count=data.class_count,
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
        logits_parts: list[torch.Tensor] = []
        control_parts: list[torch.Tensor] = []
        with torch.no_grad():
            tensor = torch.from_numpy(partition_images[:, None])
            for start in range(0, len(tensor), 2048):
                logits, predicted_control, _ = model(tensor[start : start + 2048])
                logits_parts.append(logits.cpu())
                control_parts.append(predicted_control.cpu())
        logits_all = torch.cat(logits_parts, dim=0)
        control_all = torch.cat(control_parts, dim=0).numpy()
        accuracy = float(np.mean(logits_all.argmax(dim=1).numpy() == partition_labels))
        control_mse = float(np.mean((control_all - target) ** 2))
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
        "dataset": data.dataset_name,
        "image_size": data.image_size,
        "class_count": data.class_count,
        "train_samples": len(data.train_images),
        "calibration_samples": len(data.calibration_images),
        "evaluation_samples": len(data.evaluation_images),
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
    "load_fashion_mnist",
    "load_real_digits",
    "load_real_vision_dataset",
    "make_model",
    "set_deterministic_seed",
    "train_vision_model",
]
