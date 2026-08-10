import numpy as np

from qgapselect.qcollide.closure_spectrum import (
    dangerous_direction_spectrum,
    residual_dangerous_energy,
    run_closure_spectrum_campaign,
)


def _micro_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "master_seed": 23,
        "model_seeds": [0, 1],
        "latent_dimension": 4,
        "hidden_dimension": 12,
        "visible_ranks": [1, 2, 4],
        "closure_ranks": [0, 1, 2, 3],
        "training_samples": 1024,
        "training_steps": 250,
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
        "payload_delta": 0.08,
        "behavior_gamma": 0.1,
        "local_radius": 1.2,
        "attack_steps": 24,
        "packing_thresholds": [0.05, 0.1],
    }


def test_dangerous_spectrum_matches_known_energy() -> None:
    spectrum = dangerous_direction_spectrum(np.diag([3.0, 1.0]))
    assert np.allclose(spectrum["normalized_energy"], [0.9, 0.1])
    assert np.isclose(spectrum["stable_rank"], 10.0 / 9.0)
    assert spectrum["rank_90"] == 1
    assert spectrum["rank_99"] == 2


def test_residual_energy_uses_projection_row_space() -> None:
    displacements = np.diag([3.0, 1.0])
    projection = np.array([[1.0, 0.0]])
    assert np.isclose(residual_dangerous_energy(displacements, projection), 0.1)


def test_micro_campaign_is_deterministic_and_adaptive() -> None:
    first = run_closure_spectrum_campaign(_micro_config())
    second = run_closure_spectrum_campaign(_micro_config())
    assert first == second
    assert first["claim_scope"]["adaptive_reoptimization_included"] is True
    assert first["claim_scope"]["rank_dose_response_included"] is True
    assert first["claim_scope"]["new_lower_bound_claimed"] is False
    assert first["diagnostics"]["targeted_energy_optimal_fraction"] == 1.0


def test_targeted_residual_energy_is_monotone_in_closure_rank() -> None:
    artifact = run_closure_spectrum_campaign(_micro_config())
    for visible_rank in (1, 2):
        for model_seed in (0, 1):
            rows = sorted(
                (
                    row
                    for row in artifact["rows"]
                    if row["visible_rank"] == visible_rank
                    and row["model_seed"] == model_seed
                    and row["intervention"] == "targeted"
                ),
                key=lambda row: row["closure_rank"],
            )
            energies = [row["residual_dangerous_energy"] for row in rows]
            assert all(
                first + 1e-12 >= second
                for first, second in zip(energies, energies[1:], strict=False)
            )
