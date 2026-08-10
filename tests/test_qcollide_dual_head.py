import numpy as np

from qgapselect.qcollide.dual_head import (
    run_dual_head_causal_campaign,
    train_dual_head_model,
)


def _micro_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "master_seed": 19,
        "model_seeds": [0, 1],
        "latent_dimension": 4,
        "hidden_dimension": 12,
        "visible_ranks": [1, 2, 4],
        "training_samples": 1024,
        "training_steps": 300,
        "batch_size": 256,
        "learning_rate": 0.01,
        "minimum_latent_scale": 0.2,
        "calibration_anchors": 12,
        "evaluation_anchors": 12,
        "support_samples": 256,
        "anchor_bound": 0.6,
        "latent_bound": 1.2,
        "benign_radius": 0.4,
        "benign_repetitions": 4,
        "benign_quantile": 0.9,
        "payload_delta": 0.15,
        "behavior_gamma": 0.1,
        "local_radius": 1.2,
        "attack_steps": 24,
        "closure_rank": 1,
    }


def test_hidden_jacobian_matches_finite_difference() -> None:
    model, _ = train_dual_head_model(
        seed=7,
        latent_dimension=4,
        hidden_dimension=8,
        training_samples=512,
        training_steps=50,
        batch_size=128,
        learning_rate=0.01,
        minimum_latent_scale=0.2,
    )
    point = np.array([0.2, -0.1, 0.3, -0.4])
    analytic = model.hidden_jacobian(point)
    numerical = np.empty_like(analytic)
    step = 1e-6
    for coordinate in range(point.size):
        offset = np.zeros_like(point)
        offset[coordinate] = step
        numerical[:, coordinate] = (
            model.hidden(point + offset) - model.hidden(point - offset)
        ) / (2.0 * step)
    assert np.allclose(analytic, numerical, atol=1e-6, rtol=1e-5)


def test_campaign_is_deterministic_and_preserves_claim_boundary() -> None:
    first = run_dual_head_causal_campaign(_micro_config())
    second = run_dual_head_causal_campaign(_micro_config())
    assert first == second
    assert first["claim_scope"]["synthetic_task_only"] is True
    assert first["claim_scope"]["coherent_quantum_execution"] is False
    assert first["claim_scope"]["pretrained_model_claimed"] is False


def test_full_rank_closes_baseline_and_targeted_never_increases_packing() -> None:
    artifact = run_dual_head_causal_campaign(_micro_config())
    summary = artifact["summary"]
    full_rank_baseline = [
        row
        for row in summary
        if row["visible_rank"] == 4 and row["intervention"] == "baseline"
    ][0]
    assert full_rank_baseline["mean_packing_fraction"] == 0.0
    for rank in (1, 2):
        baseline = [
            row
            for row in summary
            if row["visible_rank"] == rank and row["intervention"] == "baseline"
        ][0]
        targeted = [
            row
            for row in summary
            if row["visible_rank"] == rank and row["intervention"] == "targeted"
        ][0]
        assert targeted["mean_packing_fraction"] <= baseline["mean_packing_fraction"]
