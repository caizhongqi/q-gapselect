"""Support-aware sampling of correctly classified CIFAR examples.

This module avoids requiring every CIFAR-100 class to have a correctly
classified anchor at a performance-matched checkpoint. It selects a fixed-size
set from the model's correct-support distribution using deterministic
coverage-first round-robin sampling, while reporting class-coverage metadata.
"""

from __future__ import annotations

import numpy as np


def support_aware_correct_indices(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    target_count: int,
    minimum_supported_classes: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select a fixed-size, coverage-first subset of correctly classified rows."""

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("logits and labels must be aligned")
    if class_count <= 1 or target_count <= 0:
        raise ValueError("invalid class_count or target_count")
    if not 1 <= minimum_supported_classes <= class_count:
        raise ValueError("invalid minimum_supported_classes")

    predictions = logits.argmax(axis=1)
    per_class: list[np.ndarray] = []
    for label in range(class_count):
        per_class.append(np.flatnonzero((labels == label) & (predictions == label)))
    supported = [label for label, indices in enumerate(per_class) if len(indices) > 0]
    if len(supported) < minimum_supported_classes:
        raise RuntimeError(
            "correct-support class coverage is below the preregistered gate: "
            f"{len(supported)} < {minimum_supported_classes}"
        )
    total_correct = int(sum(len(indices) for indices in per_class))
    if total_correct < target_count:
        raise RuntimeError(
            f"only {total_correct} correctly classified examples are available; "
            f"{target_count} are required"
        )

    selected: list[int] = []
    depth = 0
    while len(selected) < target_count:
        progressed = False
        for label in supported:
            indices = per_class[label]
            if depth < len(indices):
                selected.append(int(indices[depth]))
                progressed = True
                if len(selected) == target_count:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != target_count:
        raise RuntimeError("coverage-first sampling failed to fill the fixed sample size")

    selected_labels = labels[np.asarray(selected, dtype=np.int64)]
    counts = np.bincount(selected_labels, minlength=class_count).astype(float)
    probabilities = counts[counts > 0.0] / max(float(counts.sum()), 1.0)
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if len(probabilities) else 0.0
    normalized_entropy = (
        entropy / np.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    )
    metadata = {
        "sampling_mode": "coverage_first_correct_support",
        "target_count": target_count,
        "total_correct_count": total_correct,
        "supported_class_count": len(supported),
        "supported_class_fraction": len(supported) / class_count,
        "selected_class_count": int(np.count_nonzero(counts)),
        "selected_class_entropy": entropy,
        "selected_class_entropy_normalized": float(normalized_entropy),
        "minimum_supported_classes": minimum_supported_classes,
    }
    return np.asarray(selected, dtype=np.int64), metadata


__all__ = ["support_aware_correct_indices"]
