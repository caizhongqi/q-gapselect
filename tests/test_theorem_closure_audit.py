from __future__ import annotations

import json

import pytest

from qgapselect.theorem_closure_audit import (
    CLOSURE_BLOCKED,
    HIDDEN_PARTITION_OPEN,
    PUBLIC_PARTITION_MATCH,
    build_layered_block_family,
    build_unified_theorem_closure_audit,
    default_asymptotic_witness,
    dependency_graph_is_valid,
    local_component_max_factor,
    machine_readable_status_map,
    stopping_time_dichotomy,
)
from qgapselect.variable_time_charged_history import (
    variable_time_charged_history_record,
)


def test_family_and_interface_fingerprints_are_stable_and_separated() -> None:
    charged = variable_time_charged_history_record(8)
    public_a = build_layered_block_family(
        charged,
        partition_visibility="public_static",
    )
    public_b = build_layered_block_family(
        charged,
        partition_visibility="public_static",
    )
    hidden = build_layered_block_family(
        charged,
        partition_visibility="hidden_instance_dependent",
    )

    assert public_a.base_family_id == public_b.base_family_id
    assert public_a.family_id == public_b.family_id
    assert public_a.interface.interface_id == public_b.interface.interface_id
    assert public_a.base_family_id == hidden.base_family_id
    assert public_a.family_id != hidden.family_id
    assert public_a.interface.interface_id != hidden.interface.interface_id
    assert public_a.interface.oracle_id == hidden.interface.oracle_id
    assert public_a.interface.output_relation_id == hidden.interface.output_relation_id


@pytest.mark.parametrize("m", [4, 8, 16, 32, 64, 128, 256])
def test_public_partition_composition_matches_current_candidate(m: int) -> None:
    audit = build_unified_theorem_closure_audit(m)
    public = audit.public_instantiation

    assert public.family_id == audit.public_family.family_id
    assert public.interface_id == audit.public_family.interface.interface_id
    assert public.partitioned_composition_same_interface
    assert public.partitioned_matches_candidate
    assert public.partitioned_over_candidate <= 1.2
    assert public.closure_status == PUBLIC_PARTITION_MATCH
    assert not public.strict_composition_advantage_established
    assert not public.candidate_upper_proved


def test_hidden_partition_does_not_reuse_public_history_baseline() -> None:
    audit = build_unified_theorem_closure_audit(32)
    hidden = audit.hidden_instantiation

    assert not hidden.partitioned_composition_same_interface
    assert not hidden.partitioned_matches_candidate
    assert hidden.closure_status == HIDDEN_PARTITION_OPEN
    assert "not_constructed" in hidden.candidate_upper_status
    assert not hidden.candidate_upper_proved
    assert not hidden.matching_lower_bound_established


def test_stopping_time_dichotomy_blocks_both_easy_relabellings() -> None:
    dichotomy = stopping_time_dichotomy()

    assert dichotomy.known_time_rms_covered_by_prior_work
    assert dichotomy.unknown_time_universal_rms_falsified_by_prior_lower_bound
    assert "sqrt(sum_i t_i^2)" in dichotomy.published_known_time_scale
    assert "sqrt(T log T)" in dichotomy.published_unknown_time_lower_scale
    assert len(dichotomy.source_urls) == 3


def test_default_public_partition_asymptotically_beats_candidate_proxy() -> None:
    witness = default_asymptotic_witness()

    assert witness.proof_status == "proved_local_proxy_asymptotic_lemma"
    assert witness.candidate_proxy_exponent == 3.5
    assert witness.partitioned_composition_upper_exponent == 3.25
    assert witness.composition_over_candidate_exponent == -0.25
    assert witness.partitioned_is_little_o_of_candidate_proxy


def test_dependency_ledger_is_acyclic_and_claim_remains_blocked() -> None:
    audit = build_unified_theorem_closure_audit(32)
    by_id = {item.obligation_id: item for item in audit.obligations}

    assert audit.dependency_graph_valid
    assert dependency_graph_is_valid(audit.obligations)
    assert by_id["CF-PUBLIC"].status == "falsified"
    assert by_id["UB-HIDDEN"].status == "proof_obligation"
    assert by_id["LB-WEIGHTED"].status == "proof_obligation"
    assert by_id["CLOSE"].status == "blocked"
    assert audit.closure_status == CLOSURE_BLOCKED
    assert not audit.theorem_claimable
    assert not audit.ccf_a_quantum_advantage_claimable


def test_machine_readable_status_map_exposes_all_three_pillars() -> None:
    audit = build_unified_theorem_closure_audit(32)
    status = machine_readable_status_map(audit)

    assert status["closure_status"] == CLOSURE_BLOCKED
    assert status["upper"]["hidden_proved"] is False  # type: ignore[index]
    assert status["composition"]["public_matches_candidate"] is True  # type: ignore[index]
    assert status["lower_bound"]["weighted_target_proved"] is False  # type: ignore[index]
    assert status["obligation_statuses"]["CLOSE"] == "blocked"  # type: ignore[index]
    assert status["ccf_a_quantum_advantage_claimable"] is False
    assert isinstance(json.dumps(status, sort_keys=True), str)


def test_machine_readable_status_map_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        machine_readable_status_map(object())  # type: ignore[arg-type]


def test_direct_product_floor_does_not_smuggle_checker_cost() -> None:
    audit = build_unified_theorem_closure_audit(16)
    public = audit.public_instantiation

    assert public.published_direct_product_floor > 0.0
    assert public.published_direct_product_floor < public.weighted_direct_product_target
    assert not public.weighted_direct_product_proved


def test_local_component_max_lemma_is_constant_factor_only() -> None:
    assert local_component_max_factor((3.0, 4.0, 5.0)) == pytest.approx(12.0 / 5.0)
    assert local_component_max_factor((1.0, 1.0, 1.0)) == 3.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: build_unified_theorem_closure_audit(True),
        lambda: build_unified_theorem_closure_audit(1),
        lambda: build_unified_theorem_closure_audit(8, failure_probability=0.0),
        lambda: build_unified_theorem_closure_audit(8, failure_probability=0.5),
        lambda: build_unified_theorem_closure_audit(8, match_tolerance=0.9),
        lambda: local_component_max_factor(()),
        lambda: local_component_max_factor((1.0, 0.0)),
    ],
)
def test_closure_audit_rejects_invalid_inputs(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]
