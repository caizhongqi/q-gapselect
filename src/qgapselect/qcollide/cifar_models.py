"""CIFAR datasets and parameter-matched control-head architectures for Q-COLLIDE."""

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
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.datasets import CIFAR10, CIFAR100
    from torchvision.transforms import v2
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The CIFAR topology experiment requires the optional 'vision' dependencies: "
        "pip install -e '.[vision]'"
    ) from exc


@dataclass(frozen=True)
class CIFARData:
    """Frozen train/calibration/evaluation partitions and control normalization."""

    dataset_name: str
    class_count: int
    train_images: np.ndarray
    train_labels: np.ndarray
    calibration_images: np.ndarray
    calibration_labels: np.ndarray
    evaluation_images: np.ndarray
    evaluation_labels: np.ndarray
    control_mean: np.ndarray
    control_scale: np.ndarray

    @property
    def image_size(self) -> int:
        return int(self.train_images.shape[-1])

    @property
    def input_channels(self) -> int:
        return int(self.train_images.shape[1])

    @property
    def control_dimension(self) -> int:
        return int(self.control_mean.size)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.normalization1 = nn.BatchNorm2d(out_channels)
        self.convolution2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.normalization1(self.convolution1(inputs)))
        output = self.normalization2(self.convolution2(output))
        return F.relu(output + self.shortcut(inputs))


class CIFARResNet18Control(nn.Module):
    """CIFAR-adapted ResNet-18 with a common hidden/control interface."""

    def __init__(
        self,
        hidden_dimension: int,
        control_dimension: int,
        *,
        class_count: int,
        base_width: int = 16,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(),
        )
        widths = (base_width, 2 * base_width, 4 * base_width, 8 * base_width)
        self.stage1 = self._stage(widths[0], widths[0], blocks=2, stride=1)
        self.stage2 = self._stage(widths[0], widths[1], blocks=2, stride=2)
        self.stage3 = self._stage(widths[1], widths[2], blocks=2, stride=2)
        self.stage4 = self._stage(widths[2], widths[3], blocks=2, stride=2)
        self.embedding = nn.Linear(widths[-1], hidden_dimension)
        self.main_head = nn.Linear(hidden_dimension, class_count)
        self.control_head = nn.Linear(hidden_dimension, control_dimension)

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [ResidualBlock(in_channels, out_channels, stride)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        hidden = self.stage1(hidden)
        hidden = self.stage2(hidden)
        hidden = self.stage3(hidden)
        hidden = self.stage4(hidden)
        hidden = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        return torch.tanh(self.embedding(hidden))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(images)
        return self.main_head(hidden), self.control_head(hidden), hidden


class CIFARTinyVisionTransformerControl(nn.Module):
    """Small ViT configured for 32x32 images and the shared control interface."""

    def __init__(
        self,
        hidden_dimension: int,
        control_dimension: int,
        *,
        class_count: int,
        patch_size: int = 4,
        depth: int = 6,
        heads: int = 4,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        if 32 % patch_size:
            raise ValueError("patch_size must divide 32")
        if hidden_dimension % heads:
            raise ValueError("hidden_dimension must be divisible by heads")
        token_count = (32 // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3,
            hidden_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dimension))
        self.position = nn.Parameter(torch.randn(1, token_count + 1, hidden_dimension) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=heads,
            dim_feedforward=mlp_ratio * hidden_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.normalization = nn.LayerNorm(hidden_dimension)
        self.main_head = nn.Linear(hidden_dimension, class_count)
        self.control_head = nn.Linear(hidden_dimension, control_dimension)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([class_token, patches], dim=1) + self.position
        encoded = self.transformer(tokens)
        return self.normalization(encoded[:, 0])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(images)
        return self.main_head(hidden), self.control_head(hidden), hidden


class MixerBlock(nn.Module):
    def __init__(
        self,
        token_count: int,
        hidden_dimension: int,
        *,
        token_mlp_dimension: int,
        channel_mlp_dimension: int,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dimension)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_count, token_mlp_dimension),
            nn.GELU(),
            nn.Linear(token_mlp_dimension, token_count),
        )
        self.channel_norm = nn.LayerNorm(hidden_dimension)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dimension, channel_mlp_dimension),
            nn.GELU(),
            nn.Linear(channel_mlp_dimension, hidden_dimension),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.token_norm(tokens).transpose(1, 2)
        tokens = tokens + self.token_mlp(normalized).transpose(1, 2)
        return tokens + self.channel_mlp(self.channel_norm(tokens))


class CIFARMLPMixerControl(nn.Module):
    """MLP-Mixer control model matched to the CIFAR ViT tokenization."""

    def __init__(
        self,
        hidden_dimension: int,
        control_dimension: int,
        *,
        class_count: int,
        patch_size: int = 4,
        depth: int = 10,
    ) -> None:
        super().__init__()
        if 32 % patch_size:
            raise ValueError("patch_size must divide 32")
        token_count = (32 // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3,
            hidden_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    token_count,
                    hidden_dimension,
                    token_mlp_dimension=max(32, token_count // 2),
                    channel_mlp_dimension=2 * hidden_dimension,
                )
                for _ in range(depth)
            ]
        )
        self.normalization = nn.LayerNorm(hidden_dimension)
        self.main_head = nn.Linear(hidden_dimension, class_count)
        self.control_head = nn.Linear(hidden_dimension, control_dimension)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(images).flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens)
        return self.normalization(tokens).mean(dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode(images)
        return self.main_head(hidden), self.control_head(hidden), hidden


def derived_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(("qcollide-cifar-topology-v1", str(master_seed), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def cifar_control_targets(images: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Return label-agnostic RGB low-frequency descriptors."""

    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 3 or values.shape[2] != values.shape[3]:
        raise ValueError("images must have shape (n,3,side,side)")
    side = values.shape[-1]
    if side % grid_size:
        raise ValueError("grid_size must divide image side")
    block = side // grid_size
    pooled = values.reshape(
        len(values),
        3,
        grid_size,
        block,
        grid_size,
        block,
    ).mean(axis=(3, 5))
    return pooled.reshape(len(values), -1)


def torch_control_targets(
    images: torch.Tensor,
    control_mean: torch.Tensor,
    control_scale: torch.Tensor,
    *,
    grid_size: int = 4,
) -> torch.Tensor:
    descriptor = F.adaptive_avg_pool2d(images, (grid_size, grid_size)).flatten(1)
    return (descriptor - control_mean) / control_scale


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


def _raw_cifar(dataset: str, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = dataset.strip().lower().replace("-", "_")
    dataset_class: type[CIFAR10] | type[CIFAR100]
    if normalized in {"cifar10", "cifar_10"}:
        dataset_class = CIFAR10
    elif normalized in {"cifar100", "cifar_100"}:
        dataset_class = CIFAR100
    else:
        raise ValueError(f"unknown CIFAR dataset {dataset!r}")
    training = dataset_class(root=root, train=True, download=True)
    evaluation = dataset_class(root=root, train=False, download=True)
    train_images = np.asarray(training.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    train_labels = np.asarray(training.targets, dtype=np.int64)
    evaluation_images = (
        np.asarray(evaluation.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    )
    evaluation_labels = np.asarray(evaluation.targets, dtype=np.int64)
    return train_images, train_labels, evaluation_images, evaluation_labels


def load_cifar_data(
    *,
    dataset: str,
    dataset_seed: int,
    data_root: str | Path,
    train_samples: int | None,
    calibration_samples: int,
    evaluation_samples: int | None,
    control_grid_size: int = 4,
) -> CIFARData:
    train_all, labels_all, evaluation_all, evaluation_labels_all = _raw_cifar(
        dataset,
        Path(data_root),
    )
    if not 10 <= calibration_samples < len(train_all):
        raise ValueError("calibration_samples must lie in [10, training-size)")
    train_pool, calibration_indices = train_test_split(
        np.arange(len(train_all)),
        test_size=calibration_samples,
        random_state=dataset_seed,
        stratify=labels_all,
    )
    train_local = _stratified_take(labels_all[train_pool], train_samples, dataset_seed + 1)
    evaluation_indices = _stratified_take(
        evaluation_labels_all,
        evaluation_samples,
        dataset_seed + 2,
    )
    train_indices = train_pool[train_local]
    train_images = train_all[train_indices]
    raw_controls = cifar_control_targets(train_images, control_grid_size)
    mean = raw_controls.mean(axis=0, keepdims=True).astype(np.float32)
    scale = (raw_controls.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    normalized = dataset.strip().lower().replace("-", "_")
    return CIFARData(
        dataset_name="CIFAR-10" if normalized in {"cifar10", "cifar_10"} else "CIFAR-100",
        class_count=10 if normalized in {"cifar10", "cifar_10"} else 100,
        train_images=train_images,
        train_labels=labels_all[train_indices],
        calibration_images=train_all[calibration_indices],
        calibration_labels=labels_all[calibration_indices],
        evaluation_images=evaluation_all[evaluation_indices],
        evaluation_labels=evaluation_labels_all[evaluation_indices],
        control_mean=mean,
        control_scale=scale,
    )


def make_cifar_model(
    architecture: str,
    *,
    hidden_dimension: int,
    control_dimension: int,
    class_count: int,
) -> nn.Module:
    if architecture == "resnet18":
        return CIFARResNet18Control(
            hidden_dimension,
            control_dimension,
            class_count=class_count,
        )
    if architecture == "vit_tiny":
        return CIFARTinyVisionTransformerControl(
            hidden_dimension,
            control_dimension,
            class_count=class_count,
        )
    if architecture == "mlp_mixer":
        return CIFARMLPMixerControl(
            hidden_dimension,
            control_dimension,
            class_count=class_count,
        )
    raise ValueError(f"unknown CIFAR architecture {architecture!r}")


def _evaluate_model(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    control_mean: torch.Tensor,
    control_scale: torch.Tensor,
    grid_size: int,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[float, float]:
    correct = 0
    squared_error = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(images)
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            logits, predicted_control, _ = model(batch)
            target = torch_control_targets(
                batch,
                control_mean,
                control_scale,
                grid_size=grid_size,
            )
            prediction = logits.argmax(dim=1).cpu().numpy()
            expected = labels[start : start + len(batch)]
            correct += int(np.count_nonzero(prediction == expected))
            squared_error += float(F.mse_loss(predicted_control, target, reduction="sum"))
            count += int(target.numel())
    return correct / len(images), squared_error / count


def train_cifar_model(
    architecture: str,
    data: CIFARData,
    *,
    seed: int,
    hidden_dimension: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    control_loss_weight: float,
    control_grid_size: int,
    checkpoint_epochs: tuple[int, ...] = (),
    device_name: str = "auto",
) -> tuple[nn.Module, dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    """Train one architecture and retain requested CPU checkpoints."""

    set_deterministic_seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = make_cifar_model(
        architecture,
        hidden_dimension=hidden_dimension,
        control_dimension=data.control_dimension,
        class_count=data.class_count,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    dataset = TensorDataset(
        torch.from_numpy(data.train_images),
        torch.from_numpy(data.train_labels),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    augmentation = v2.Compose(
        [
            v2.RandomCrop((32, 32), padding=4),
            v2.RandomHorizontalFlip(),
        ]
    )
    control_mean = torch.from_numpy(data.control_mean).to(device)
    control_scale = torch.from_numpy(data.control_scale).to(device)
    requested = set(checkpoint_epochs)
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}

    def snapshot(epoch: int) -> None:
        checkpoints[epoch] = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    if 0 in requested:
        snapshot(0)
    epoch_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        example_count = 0
        for images, labels in loader:
            images = augmentation(images).to(device)
            labels = labels.to(device)
            control_target = torch_control_targets(
                images,
                control_mean,
                control_scale,
                grid_size=control_grid_size,
            )
            logits, predicted_control, _ = model(images)
            loss = F.cross_entropy(logits, labels) + control_loss_weight * F.mse_loss(
                predicted_control,
                control_target,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(images)
            example_count += len(images)
        scheduler.step()
        epoch_losses.append(loss_total / max(1, example_count))
        if epoch in requested:
            snapshot(epoch)
    model.eval()
    calibration_accuracy, calibration_control_mse = _evaluate_model(
        model,
        data.calibration_images,
        data.calibration_labels,
        control_mean=control_mean,
        control_scale=control_scale,
        grid_size=control_grid_size,
        device=device,
    )
    evaluation_accuracy, evaluation_control_mse = _evaluate_model(
        model,
        data.evaluation_images,
        data.evaluation_labels,
        control_mean=control_mean,
        control_scale=control_scale,
        grid_size=control_grid_size,
        device=device,
    )
    diagnostics: dict[str, Any] = {
        "architecture": architecture,
        "dataset": data.dataset_name,
        "class_count": data.class_count,
        "train_samples": len(data.train_images),
        "calibration_samples": len(data.calibration_images),
        "evaluation_samples": len(data.evaluation_images),
        "hidden_dimension": hidden_dimension,
        "control_dimension": data.control_dimension,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "epoch_losses": epoch_losses,
        "calibration_accuracy": calibration_accuracy,
        "evaluation_accuracy": evaluation_accuracy,
        "calibration_control_mse": calibration_control_mse,
        "evaluation_control_mse": evaluation_control_mse,
        "device": str(device),
    }
    return model, diagnostics, checkpoints


__all__ = [
    "CIFARData",
    "CIFARMLPMixerControl",
    "CIFARResNet18Control",
    "CIFARTinyVisionTransformerControl",
    "cifar_control_targets",
    "derived_seed",
    "load_cifar_data",
    "make_cifar_model",
    "set_deterministic_seed",
    "torch_control_targets",
    "train_cifar_model",
]
