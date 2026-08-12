import numpy as np
import pytest

from qgapselect.qcollide.cifar_support_sampling import support_aware_correct_indices


def test_support_aware_sampling_balances_available_classes():
    labels = np.repeat(np.arange(5), 4)
    logits = np.full((20, 5), -1.0)
    for index, label in enumerate(labels):
        logits[index, label] = 2.0
    indices, metadata = support_aware_correct_indices(
        logits,
        labels,
        class_count=5,
        target_count=10,
        minimum_supported_classes=5,
    )
    selected = labels[indices]
    counts = np.bincount(selected, minlength=5)
    assert counts.tolist() == [2, 2, 2, 2, 2]
    assert metadata["supported_class_count"] == 5
    assert metadata["selected_class_count"] == 5


def test_support_aware_sampling_fails_closed_on_low_class_coverage():
    labels = np.repeat(np.arange(5), 2)
    logits = np.zeros((10, 5))
    logits[:, 0] = 1.0
    with pytest.raises(RuntimeError, match="class coverage"):
        support_aware_correct_indices(
            logits,
            labels,
            class_count=5,
            target_count=4,
            minimum_supported_classes=3,
        )
