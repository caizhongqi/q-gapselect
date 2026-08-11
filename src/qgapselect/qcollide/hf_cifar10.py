"""Torchvision-compatible CIFAR-10 adapter backed by the Hugging Face mirror."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HFCIFAR10:
    """Expose ``uoft-cs/cifar10`` through the subset of the torchvision API we use.

    The adapter intentionally keeps the pretrained-vision experiment independent
    of the University of Toronto download endpoint. ``download`` is accepted for
    API compatibility; Hugging Face ``datasets`` handles local caching itself.
    """

    dataset_id = "uoft-cs/cifar10"

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool = True,
        download: bool = False,
        **_: Any,
    ) -> None:
        del download
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - executable dependency guard
            raise RuntimeError(
                "Hugging Face CIFAR-10 loading requires the 'datasets' package"
            ) from exc

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = "train" if train else "test"
        self._dataset = load_dataset(
            self.dataset_id,
            split=self.split,
            cache_dir=str(self.root),
        )
        self.targets = [int(value) for value in self._dataset["label"]]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        sample = self._dataset[int(index)]
        image = sample["img"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        return image, int(sample["label"])


def install_hf_cifar10_torchvision_adapter() -> None:
    """Replace ``torchvision.datasets.CIFAR10`` for the current process only."""

    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError("torchvision is required for the CIFAR-10 adapter") from exc
    torchvision.datasets.CIFAR10 = HFCIFAR10


__all__ = ["HFCIFAR10", "install_hf_cifar10_torchvision_adapter"]
