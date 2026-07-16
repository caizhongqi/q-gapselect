from __future__ import annotations

import math

import pytest

from qgapselect.frozen_quantum_instance_design import (
    FAMILIES,
    generate_frozen_quantum_instance_design,
)
from qgapselect.frozen_quantum_reference_benchmarking import (
    FrozenQuantumReferenceInstance,
)


def _generate(
    *,
    family: str = "clustered_boundary",
    case_seed: int = 7,
    permutation_seed: int | None = None,
    table_size: int = 2048,
):
    return generate_frozen_quantum_instance_design(
        case_id="n16-k4-gap",
        family=family,
        n_arms=16,
        k=4,
        threshold=0.5,
        public_gap_floor=math.pi / 128.0,
        table_size=table_size,
        case_seed=case_seed,
        permutation_seed=permutation_seed,
    )


def test_case_seed_generation_is_fully_deterministic() -> None:
    first = _generate(case_seed=91)
    second = _generate(case_seed=91)

    assert first.difficulty_fingerprint == second.difficulty_fingerprint
    assert first.signed_latent_angular_gaps == second.signed_latent_angular_gaps
    assert first.signed_empirical_angular_gaps == second.signed_empirical_angular_gaps
    assert first.structure == second.structure
    assert first.fixture.manifest_hash == second.fixture.manifest_hash
    assert first.manifest_document() == second.manifest_document()


@pytest.mark.parametrize("family", FAMILIES)
def test_different_instance_seeds_produce_nonisomorphic_difficulty_fingerprints(
    family: str,
) -> None:
    designs = [_generate(family=family, case_seed=seed) for seed in range(20)]
    fingerprints = {design.difficulty_fingerprint for design in designs}
    active_counts = {design.structure.active_count for design in designs}
    heterogeneity = {round(design.structure.heterogeneity, 8) for design in designs}

    assert len(fingerprints) >= 16
    assert len(active_counts) >= 3
    assert len(heterogeneity) == 20


def test_arm_permutation_does_not_change_difficulty_fingerprint() -> None:
    first = _generate(case_seed=17, permutation_seed=101)
    second = _generate(case_seed=17, permutation_seed=202)

    assert first.difficulty_fingerprint == second.difficulty_fingerprint
    assert first.structure == second.structure
    assert sorted(first.empirical_success_counts) == sorted(second.empirical_success_counts)
    assert first.signed_latent_angular_gaps != second.signed_latent_angular_gaps
    assert first.selected_arms != second.selected_arms


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(5))
def test_post_rounding_fixture_strictly_satisfies_top_k_and_gap_promises(
    family: str,
    seed: int,
) -> None:
    design = _generate(family=family, case_seed=seed, table_size=257)
    frozen_means = design.fixture.evaluator.frozen_means

    assert len(design.selected_arms) == 4
    assert all(frozen_means[index] > design.threshold for index in design.selected_arms)
    assert all(
        frozen_means[index] < design.threshold
        for index in range(design.n_arms)
        if index not in set(design.selected_arms)
    )
    assert min(abs(gap) for gap in design.signed_empirical_angular_gaps) >= (
        design.public_gap_floor - 1e-12
    )
    assert design.structure.empirical_boundary_gap >= design.public_gap_floor - 1e-12

    # Reuse the benchmark harness's independent promise validator.
    instance = FrozenQuantumReferenceInstance(
        family_id=family,
        instance_id=f"{family}/{seed}",
        fixture=design.fixture,
        public_threshold=design.threshold,
        public_gap_floor=design.public_gap_floor,
        k=design.k,
    )
    assert instance.k == design.k


def test_family_structures_are_auditable_and_distinct() -> None:
    designs = {family: _generate(family=family, case_seed=37) for family in FAMILIES}

    assert all(design.structure.active_count >= 2 for design in designs.values())
    assert all(design.structure.distinct_empirical_gap_count > 1 for design in designs.values())
    assert designs["equal_gap"].structure.empirical_gap_cv < (
        designs["dyadic_gap"].structure.empirical_gap_cv
    )
    assert len({design.difficulty_fingerprint for design in designs.values()}) == 3


def test_near_endpoint_threshold_and_small_table_remain_strict() -> None:
    design = generate_frozen_quantum_instance_design(
        case_id="endpoint-small-table",
        family="dyadic_gap",
        n_arms=6,
        k=2,
        threshold=0.04,
        public_gap_floor=0.05,
        table_size=17,
        case_seed=13,
    )

    assert len(design.selected_arms) == 2
    assert min(abs(gap) for gap in design.signed_empirical_angular_gaps) >= 0.05 - 1e-12
    assert all(mean != design.threshold for mean in design.fixture.evaluator.frozen_means)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"family": "unknown"}, ValueError, "family"),
        ({"n_arms": 1}, ValueError, "n_arms"),
        ({"k": 0}, ValueError, "k"),
        ({"k": 16}, ValueError, "smaller"),
        ({"threshold": 0.0}, ValueError, "threshold"),
        ({"threshold": 1.0}, ValueError, "threshold"),
        ({"public_gap_floor": 0.0}, ValueError, "public_gap_floor"),
        ({"public_gap_floor": math.pi / 4.0}, ValueError, "threshold margins"),
        ({"table_size": 0}, ValueError, "table_size"),
        ({"case_seed": -1}, ValueError, "case_seed"),
        ({"candidate_ids": ("a", "b")}, ValueError, "candidate_ids"),
    ],
)
def test_invalid_boundaries_are_rejected(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "case_id": "boundary",
        "family": "equal_gap",
        "n_arms": 16,
        "k": 4,
        "threshold": 0.5,
        "public_gap_floor": math.pi / 128.0,
        "table_size": 257,
        "case_seed": 1,
    }
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        generate_frozen_quantum_instance_design(**arguments)  # type: ignore[arg-type]
