from qgapselect.qcollide.overlap_scaling import (
    analytic_packed_query_scales,
    merge_overlap_scaling_components,
)


def test_positive_capacity_query_scales_have_expected_exponents():
    scales = analytic_packed_query_scales(100, 100, 25)
    assert scales["defined_for_positive_capacity"] is True
    assert scales["effective_pair_scale"] == 400.0
    assert scales["classical_query_scale"] == 20.0
    assert abs(scales["quantum_query_scale"] - 400.0 ** (1.0 / 3.0)) < 1e-12
    assert scales["classical_to_quantum_ratio"] > 1.0


def test_zero_capacity_query_proxy_fails_closed():
    scales = analytic_packed_query_scales(100, 100, 0)
    assert scales["defined_for_positive_capacity"] is False
    assert scales["classical_to_quantum_ratio"] is None


def _record(family, degree_aware, truth, packed_error, scaled_error):
    record = {
        "family": family,
        "degree_aware": degree_aware,
        "true_matching_size": truth,
        "edge_count": truth if family == "matching" else 2 * truth,
        "maximum_degree": 1 if family == "matching" else 2,
        "target_q": 0.25,
        "survival_probability": 0.5,
        "packed_inversion_estimate": truth + packed_error,
        "mean_sampled_matching": 1.0,
        "scaled_matching_estimate": truth + scaled_error,
        "absolute_error_packed_inversion": abs(packed_error),
        "absolute_error_scaled_matching": abs(scaled_error),
        "deterministic_capacity_envelope": {"lower": truth, "upper": truth + 2},
        "uniform_survival_population_envelope": {"lower": 0.1, "upper": 0.9},
    }
    if not degree_aware:
        record["uniform_survival_capacity_certificate"] = {
            "observed_survival": 0.5,
            "trials": 1000,
            "failure_probability": 0.001,
            "survival_lower": 0.4,
            "survival_upper": 0.6,
            "capacity_lower": truth - 1,
            "capacity_upper": truth + 1,
        }
    return record


def test_merge_scaling_keeps_analytic_and_empirical_claims_separate():
    artifact = {
        "artifact_type": "qcollide_overlap_capacity_benchmark",
        "schema_version": 2,
        "n": 64,
        "requested_matching_size": 8,
        "records": [
            _record("matching", False, 8, 0.2, 0.3),
            _record("matching", True, 8, 0.1, 0.2),
            _record("sparse_overlap", False, 12, 4.0, 3.0),
            _record("sparse_overlap", True, 12, 1.0, 1.0),
        ],
    }
    merged = merge_overlap_scaling_components([artifact])
    assert merged["gates"]["all_deterministic_envelopes_cover_truth"] is True
    assert merged["gates"]["positive_capacity_cell_count"] == 2
    assert merged["summary"]["mean_sparse_degree_aware_packed_error"] < merged["summary"][
        "mean_sparse_uniform_packed_error"
    ]
    assert merged["claim_boundary"]["hardware_quantum_advantage_claimed"] is False
