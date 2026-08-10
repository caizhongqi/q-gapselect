import numpy as np

from qgapselect.qcollide.scaling import fit_log_linear, run_scaling_campaign


def _micro_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "master_seed": 17,
        "f0": {
            "n_values": [16, 32, 64],
            "marked_exponents": [0.0, 1.0],
            "repetitions": 1,
        },
        "f1": {
            "n_values": [16, 32, 64],
            "matching_exponents": [0.0, 0.5, 1.0],
            "repetitions": 1,
        },
        "f2": {
            "n_values": [16, 32],
            "target_collision_exponents": [0.0, 1.0],
            "repetitions": 2,
        },
        "f3": {
            "n_values": [16, 32],
            "matching_exponents": [0.0, 1.0],
            "repetitions": 1,
            "schedules": [
                {
                    "name": "early",
                    "cumulative_bits": [3, 6, 12],
                    "stage_costs": [1.0, 2.0, 8.0],
                },
                {
                    "name": "late",
                    "cumulative_bits": [1, 2, 12],
                    "stage_costs": [1.0, 2.0, 8.0],
                },
            ],
        },
        "f4": {
            "manifold_dimensions": [8],
            "control_rank_fractions": [0.5, 1.0],
            "openness_values": [0.0, 1.0],
            "curvature_values": [0.05],
            "anchors": 4,
            "repetitions": 1,
            "control_epsilon": 0.2,
            "payload_delta": 0.05,
            "behavior_gamma": 0.05,
            "local_radius": 1.0,
        },
    }


def test_fit_log_linear_recovers_known_coefficients() -> None:
    rows = []
    for n in (8.0, 16.0, 32.0):
        for matching_size in (1.0, 2.0, 4.0):
            rows.append(
                {
                    "n": n,
                    "matching_size": matching_size,
                    "cost": 3.0 * n ** (2.0 / 3.0) * matching_size ** (-1.0 / 3.0),
                }
            )
    fit = fit_log_linear(
        rows,
        response="cost",
        predictors=("n", "matching_size"),
    )
    assert np.isclose(fit["coefficients"]["n"], 2.0 / 3.0)
    assert np.isclose(fit["coefficients"]["matching_size"], -1.0 / 3.0)
    assert np.isclose(fit["r_squared"], 1.0)


def test_micro_campaign_is_deterministic_and_preserves_claim_boundary() -> None:
    first = run_scaling_campaign(_micro_config())
    second = run_scaling_campaign(_micro_config())
    assert first == second
    assert first["claim_scope"]["statevector_execution"] is False
    assert first["claim_scope"]["new_lower_bound_claimed"] is False
    assert first["record_counts"] == {
        "F0": 6,
        "F1": 9,
        "F2": 8,
        "F3": 8,
        "F4": 4,
    }


def test_f0_negative_control_and_f1_coefficients_are_calibrated() -> None:
    artifact = run_scaling_campaign(_micro_config())
    f0 = artifact["panels"]["F0"]
    assert all(row["structured_claw_admissible"] is False for row in f0["rows"])
    for fit in f0["fits_by_marked_exponent"]:
        observed = fit["fit"]["coefficients"]["n"]
        assert np.isclose(observed, fit["expected_n_coefficient"], atol=0.02)

    f1 = artifact["panels"]["F1"]
    classical = f1["joint_fits"]["classical"]["coefficients"]
    quantum = f1["joint_fits"]["quantum_proxy"]["coefficients"]
    assert np.isclose(classical["n"], 1.0)
    assert np.isclose(classical["matching_size"], -0.5)
    assert np.isclose(quantum["n"], 2.0 / 3.0)
    assert np.isclose(quantum["matching_size"], -1.0 / 3.0)
