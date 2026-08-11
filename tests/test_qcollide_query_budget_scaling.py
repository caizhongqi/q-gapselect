from __future__ import annotations

import numpy as np

from qgapselect.qcollide.query_budget_results import merge_query_budget_components
from qgapselect.qcollide.query_budget_scaling import (
    evaluate_query_budget_component,
    make_controlled_fixture,
    qccs_packed_query_proxy,
    sampled_edge_count,
    sampled_matching_capacity,
)


def test_controlled_fixtures_recover_exact_capacity_at_full_query() -> None:
    n = 64
    k = 8
    for index, family in enumerate(
        ("matching", "bounded_overlap", "multi_hub", "biclique")
    ):
        fixture = make_controlled_fixture(
            family,
            n=n,
            matching_size=k,
            seed=100 + index,
        )
        sampled = sampled_matching_capacity(
            fixture,
            queried_per_side=n,
            rng=np.random.default_rng(200 + index),
        )
        assert sampled == k
        edges = sampled_edge_count(
            fixture,
            queried_per_side=n,
            rng=np.random.default_rng(300 + index),
        )
        if family == "matching":
            assert edges == k
        else:
            assert edges > k


def test_query_budget_component_and_merge_keep_claim_boundaries() -> None:
    common = {
        "n": 64,
        "matching_densities": (0.125,),
        "budget_fractions": (0.5, 1.0),
        "graph_seeds": (1, 2),
        "sampling_trials": 8,
        "additive_epsilon": 0.2,
        "relative_epsilon": 0.25,
        "success_probability": 0.75,
        "seed": 1234,
    }
    matching = evaluate_query_budget_component(panel="matching", **common)
    overlap = evaluate_query_budget_component(panel="overlap", **common)
    merged = merge_query_budget_components(
        [matching, overlap],
        matching_n_values=[64],
        overlap_n_values=[64],
        matching_densities=[0.125],
    )
    assert merged["gates"]["complete_component_grid"] is True
    assert merged["gates"]["matching_full_reconstruction_relative_success"] is True
    assert merged["gates"]["overlap_full_reconstruction_relative_success"] is True
    assert merged["gates"]["overlap_edge_count_negative_control_failure_fraction"] == 1.0
    assert merged["claim_boundary"]["hardware_quantum_advantage_claimed"] is False
    assert merged["claim_boundary"]["overlap_quantum_advantage_claimed"] is False


def test_qccs_packed_proxy_has_one_third_structural_exponent() -> None:
    small = qccs_packed_query_proxy(256, 16, relative_epsilon=0.25)
    large = qccs_packed_query_proxy(512, 32, relative_epsilon=0.25)
    assert small["effective_pair_scale"] == large["effective_pair_scale"] * 0.5
    ratio = large["structural_query_scale"] / small["structural_query_scale"]
    expected = (large["effective_pair_scale"] / small["effective_pair_scale"]) ** (1.0 / 3.0)
    assert np.isclose(ratio, expected)
