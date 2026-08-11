#!/usr/bin/env python3
"""Freeze CIFAR-10 train/test arrays from the Hugging Face mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _write_split(dataset, *, image_path: Path, label_path: Path) -> dict[str, object]:
    labels = np.asarray(dataset["label"], dtype=np.int64)
    images = np.empty((len(dataset), 32, 32, 3), dtype=np.uint8)
    for index, sample in enumerate(dataset):
        image = sample["img"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        values = np.asarray(image, dtype=np.uint8)
        if values.shape != (32, 32, 3):
            raise ValueError(f"unexpected CIFAR-10 image shape {values.shape}")
        images[index] = values
    np.save(image_path, images, allow_pickle=False)
    np.save(label_path, labels, allow_pickle=False)
    digest = hashlib.sha256()
    for path in (image_path, label_path):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "count": len(dataset),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "combined_sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'datasets' package") from exc

    root = args.output_directory
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "hf-cache"
    train = load_dataset("uoft-cs/cifar10", split="train", cache_dir=str(cache))
    test = load_dataset("uoft-cs/cifar10", split="test", cache_dir=str(cache))
    if len(train) != 50000 or len(test) != 10000:
        raise RuntimeError("unexpected CIFAR-10 split size")
    metadata = {
        "artifact_type": "qcollide_cifar10_hf_raw_snapshot",
        "schema_version": 1,
        "dataset_id": "uoft-cs/cifar10",
        "train": _write_split(
            train,
            image_path=root / "train_images.npy",
            label_path=root / "train_labels.npy",
        ),
        "evaluation": _write_split(
            test,
            image_path=root / "evaluation_images.npy",
            label_path=root / "evaluation_labels.npy",
        ),
    }
    (root / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
