import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from qgapselect.qcollide.vision import run_real_vision_campaign
from qgapselect.qcollide.vision_linear import row_basis


def _micro_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "master_seed": 23,
        "dataset_seed": 17,
        "architectures": ["cnn"],
        "model_seeds": [0],
        "hidden_dimension": 16,
        "control_dimension": 8,
        "visible_ranks": [1, 4],
        "closure_ranks": [1],
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 0.003,
        "control_loss_weight": 0.25,
        "anchors_per_class": 1,
        "support_samples": 256,
        "benign_radius": 0.35,
        "benign_repetitions": 2,
        "control_quantile": 0.9,
        "payload_quantile": 0.9,
        "maximum_attack_radius": 2.0,
        "line_search_steps": 24,
        "target_margin": 0.0,
        "packing_target": 0.1,
    }


def test_row_basis_is_invariant_to_redundant_rows() -> None:
    matrix = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    basis = row_basis(matrix)
    assert basis.shape == (1, 3)
    assert np.isclose(abs(basis[0, 0]), 1.0)


def test_real_image_campaign_preserves_sham_and_claim_boundaries() -> None:
    artifact = run_real_vision_campaign(_micro_config())
    assert artifact["claim_scope"]["real_image_dataset"] is True
    assert artifact["claim_scope"]["production_scale_claimed"] is False
    assert artifact["claim_scope"]["coherent_quantum_execution"] is False
    assert artifact["training"][0]["evaluation_accuracy"] > 0.75
    rows = artifact["rows"]
    for visible_rank in (1, 4):
        baseline = [
            row
            for row in rows
            if row["visible_rank"] == visible_rank
            and row["intervention"] == "baseline"
        ][0]
        sham = [
            row
            for row in rows
            if row["visible_rank"] == visible_rank
            and row["intervention"] == "row_sham"
        ][0]
        assert np.isclose(sham["packing_fraction"], baseline["packing_fraction"])
        assert np.isclose(sham["mean_openness"], baseline["mean_openness"])
