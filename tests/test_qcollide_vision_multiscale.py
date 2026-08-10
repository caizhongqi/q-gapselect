import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import torch

from qgapselect.qcollide.vision_common import adaptive_rank_summary
from qgapselect.qcollide.vision_merge import merge_real_vision_artifacts
from qgapselect.qcollide.vision_models import image_control_targets, make_model


def test_low_frequency_controls_support_28_by_28_images() -> None:
    rng = np.random.default_rng(7)
    images = rng.random((11, 28, 28), dtype=np.float32)
    controls = image_control_targets(images)
    assert controls.shape == (11, 8)
    assert np.all(np.isfinite(controls))


@pytest.mark.parametrize("architecture", ["cnn", "tiny_vit"])
def test_models_support_28_by_28_images(architecture: str) -> None:
    model = make_model(
        architecture,
        hidden_dimension=48,
        control_dimension=8,
        image_size=28,
        class_count=10,
    )
    logits, controls, hidden = model(torch.zeros(3, 1, 28, 28))
    assert logits.shape == (3, 10)
    assert controls.shape == (3, 8)
    assert hidden.shape == (3, 48)


def test_merge_rejects_cross_dataset_components() -> None:
    common = {
        "schema_version": 1,
        "master_seed": 5,
        "artifact_type": "same-type",
        "training": [],
        "spectra": [],
        "rows": [],
    }
    digits = {
        **common,
        "claim_scope": {
            "dataset": "scikit-learn digits",
            "image_size": 8,
            "class_count": 10,
            "architectures": ["cnn"],
        },
    }
    fashion = {
        **common,
        "claim_scope": {
            "dataset": "Fashion-MNIST",
            "image_size": 28,
            "class_count": 10,
            "architectures": ["cnn"],
        },
    }
    with pytest.raises(ValueError, match="share dataset"):
        merge_real_vision_artifacts([digits, fashion], packing_target=0.05)


def test_adaptive_rank_uses_tolerance_at_exact_fraction_boundary() -> None:
    summary = [
        {
            "architecture": "cnn",
            "visible_rank": 1,
            "intervention": "baseline",
            "closure_rank": 0,
            "mean_packing_fraction": 0.5,
        },
        {
            "architecture": "cnn",
            "visible_rank": 1,
            "intervention": "targeted",
            "closure_rank": 6,
            "mean_packing_fraction": 0.05000000000000001,
        },
    ]
    spectra = [
        {
            "architecture": "cnn",
            "visible_rank": 1,
            "rank_95": 7,
            "displacement_rank_95": 8,
        }
    ]
    result = adaptive_rank_summary(summary, spectra, packing_target=0.05)
    assert result[0]["adaptive_closure_rank"] == 6
