import numpy as np

from qgapselect.qcollide.cifar_single_run_broad_pool import balanced_correct_pool


def test_balanced_correct_pool_maximizes_coverage_under_fixed_size():
    labels = np.repeat(np.arange(60), 5)
    logits = np.full((len(labels), 60), -10.0)
    logits[np.arange(len(labels)), labels] = 10.0
    indices, metadata = balanced_correct_pool(
        logits,
        labels,
        pool_size=120,
        minimum_unique_classes=50,
        per_class_cap=4,
        seed=20260811,
    )
    assert len(indices) == 120
    assert metadata["unique_fine_classes"] == 60
    assert max(metadata["class_counts"].values()) <= 2


def test_balanced_correct_pool_fails_closed_on_insufficient_class_coverage():
    labels = np.repeat(np.arange(40), 4)
    logits = np.full((len(labels), 40), -10.0)
    logits[np.arange(len(labels)), labels] = 10.0
    try:
        balanced_correct_pool(
            logits,
            labels,
            pool_size=80,
            minimum_unique_classes=50,
            per_class_cap=4,
            seed=20260811,
        )
    except RuntimeError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("coverage gate should fail")
