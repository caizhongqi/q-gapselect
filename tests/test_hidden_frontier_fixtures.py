from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from qgapselect.hidden_frontier_fixtures import (
    CENTER_GRID,
    F_CLUSTER,
    F_DYADIC,
    F_EQ,
    F_HIDDEN_FRONTIER,
    F_PUBLIC_PARTITION,
    F_TIE_NC,
    F_UNKNOWN_TIME_NC,
    FAMILY_IDS,
    NestedCountProfile,
    algorithmic_fixture_deduplication_key,
    canonical_orbit_hash,
    deduplicate_isomorphic_fixtures,
    generate_hidden_frontier_fixture,
    validate_hidden_frontier_fixture,
)


def _fixture(
    family_id: str = F_HIDDEN_FRONTIER,
    *,
    seed: int = 17,
    permutation_seed: int | None = None,
    n: int = 32,
    k: int = 6,
):
    return generate_hidden_frontier_fixture(
        family_id=family_id,
        n=n,
        k=k,
        design_gap=math.pi / 256.0,
        fixture_seed=seed,
        delta=0.025,
        hard_query_cap=524_288,
        permutation_seed=permutation_seed,
    )


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            keys.update(_all_keys(child))
    return keys


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_every_registered_family_is_strictly_valid_and_bounded(family_id: str) -> None:
    fixture = _fixture(family_id)

    fixture.validate()
    assert fixture.family_id == family_id
    assert len(fixture.angles) == fixture.n == 32
    assert len(fixture.hidden_permutation) == fixture.n
    assert sorted(fixture.hidden_permutation) == list(range(fixture.n))
    assert all(0.0 <= angle <= math.pi / 2.0 for angle in fixture.angles)
    assert fixture.hidden_center in CENTER_GRID
    assert len(fixture.fixture_hash) == len(fixture.orbit_hash) == 64
    assert len(fixture.interface_id) == 64


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_generation_and_hash_replay_are_deterministic(family_id: str) -> None:
    first = _fixture(family_id, seed=91)
    second = _fixture(family_id, seed=91)
    replayed = first.replay()

    assert first == second == replayed
    assert first.trusted_manifest_document() == second.trusted_manifest_document()
    assert json.dumps(first.trusted_manifest_document(), sort_keys=True, allow_nan=False)


def test_seed_and_permutation_provenance_change_only_expected_hashes() -> None:
    first = _fixture(seed=8, permutation_seed=100)
    permuted = _fixture(seed=8, permutation_seed=200)
    another_instance = _fixture(seed=9, permutation_seed=100)

    assert first.angles != permuted.angles
    assert first.fixture_hash != permuted.fixture_hash
    assert first.orbit_hash == permuted.orbit_hash
    assert first.interface_id == permuted.interface_id
    assert first.orbit_hash != another_instance.orbit_hash


def test_canonical_orbit_hash_is_exactly_permutation_invariant() -> None:
    values = (0.2, 0.4, 0.7, 0.9)
    forward = canonical_orbit_hash(values, n=4, k=2, non_unique_output=False)
    reversed_hash = canonical_orbit_hash(
        tuple(reversed(values)), n=4, k=2, non_unique_output=False
    )
    changed = canonical_orbit_hash(
        (0.2, 0.4, 0.7, 0.9000000000000001),
        n=4,
        k=2,
        non_unique_output=False,
    )

    assert forward == reversed_hash
    assert forward != changed


def test_algorithmic_dedup_preserves_first_within_one_interface() -> None:
    first = _fixture(seed=3, permutation_seed=11)
    isomorphic = _fixture(seed=3, permutation_seed=22)
    distinct = _fixture(seed=4, permutation_seed=11)

    unique = deduplicate_isomorphic_fixtures((first, isomorphic, distinct))

    assert unique == (first, distinct)


def test_algorithmic_dedup_retains_equal_orbit_under_distinct_interfaces() -> None:
    hidden = _fixture(F_HIDDEN_FRONTIER, seed=3)
    public = _fixture(F_PUBLIC_PARTITION, seed=3)

    assert hidden.orbit_hash == public.orbit_hash
    assert hidden.interface_id != public.interface_id
    assert algorithmic_fixture_deduplication_key(hidden) != (
        algorithmic_fixture_deduplication_key(public)
    )
    assert deduplicate_isomorphic_fixtures((hidden, public)) == (hidden, public)


def test_hidden_and_public_partition_are_paired_but_use_distinct_interfaces() -> None:
    hidden = _fixture(F_HIDDEN_FRONTIER, seed=44)
    public = _fixture(F_PUBLIC_PARTITION, seed=44)

    assert hidden.angles == public.angles
    assert hidden.latent_angles == public.latent_angles
    assert hidden.hidden_center == public.hidden_center
    assert hidden.hidden_permutation == public.hidden_permutation
    assert hidden.nested_count_profile == public.nested_count_profile
    assert hidden.orbit_hash == public.orbit_hash
    assert hidden.interface_id != public.interface_id
    assert "static_partition" not in hidden.algorithm_view()
    assert "static_partition" in public.algorithm_view()


def test_all_blind_positive_families_have_one_information_matched_interface() -> None:
    blind_families = (
        F_EQ,
        F_DYADIC,
        F_CLUSTER,
        F_HIDDEN_FRONTIER,
        F_UNKNOWN_TIME_NC,
    )
    fixtures = [_fixture(family_id, seed=12) for family_id in blind_families]

    assert len({fixture.interface_id for fixture in fixtures}) == 1
    assert len({fixture.orbit_hash for fixture in fixtures}) == len(blind_families)


def test_tie_control_is_indistinguishable_at_the_blind_interface() -> None:
    positive = _fixture(F_HIDDEN_FRONTIER, seed=12)
    tie = _fixture(F_TIE_NC, seed=12)

    assert tie.algorithm_view() == positive.algorithm_view()
    assert tie.interface_id == positive.interface_id


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_primary_interface_never_supplies_a_numeric_gap(family_id: str) -> None:
    fixture = _fixture(family_id)
    view = fixture.algorithm_view()
    serialized = json.dumps(view, sort_keys=True, allow_nan=False)

    assert "minimum_boundary_gap" not in serialized
    assert "design_gap" not in serialized
    assert "realized_boundary_gap" not in serialized
    assert fixture.design_gap.hex() not in serialized
    assert repr(fixture.design_gap) not in serialized
    assert view["structural_promise"]["boundary_information"] == "not_supplied"


@pytest.mark.parametrize("family_id", (F_HIDDEN_FRONTIER, F_PUBLIC_PARTITION))
def test_interface_is_invariant_to_private_design_gap(family_id: str) -> None:
    common = {
        "family_id": family_id,
        "n": 32,
        "k": 6,
        "fixture_seed": 31,
        "delta": 0.025,
        "hard_query_cap": 524_288,
    }
    coarse = generate_hidden_frontier_fixture(
        **common, design_gap=math.pi / 256.0
    )
    fine = generate_hidden_frontier_fixture(**common, design_gap=math.pi / 512.0)

    assert coarse.algorithm_view() == fine.algorithm_view()
    assert coarse.interface_id == fine.interface_id
    assert coarse.fixture_hash != fine.fixture_hash


@pytest.mark.parametrize("family_id", FAMILY_IDS)
def test_algorithm_view_contains_no_runner_truth_or_schedule(family_id: str) -> None:
    fixture = _fixture(family_id)
    view = fixture.algorithm_view()
    keys = {key.lower() for key in _all_keys(view)}
    forbidden_tokens = (
        "angle",
        "beta",
        "center",
        "ranking",
        "membership",
        "active",
        "schedule",
        "seed",
        "family",
        "permutation",
        "stopping",
        "radius",
    )

    assert not any(
        token in key for token in forbidden_tokens for key in keys
    )
    assert fixture.family_id not in json.dumps(view, sort_keys=True)
    assert "fixture_hash" not in view
    assert "orbit_hash" not in view
    assert set(view) >= {
        "n",
        "k",
        "delta",
        "hard_query_cap",
        "structural_promise",
        "interface_id",
    }
    if family_id == F_PUBLIC_PARTITION:
        assert "static_partition" in view
    else:
        assert "static_partition" not in view


def test_mutating_returned_algorithm_document_cannot_mutate_fixture() -> None:
    fixture = _fixture()
    view = fixture.algorithm_view()
    view["n"] = 999
    view["structural_promise"] = {}

    fresh = fixture.algorithm_view()
    assert fresh["n"] == fixture.n
    assert fresh["structural_promise"]


@pytest.mark.parametrize(
    "family_id",
    (F_EQ, F_DYADIC, F_CLUSTER, F_HIDDEN_FRONTIER, F_PUBLIC_PARTITION, F_UNKNOWN_TIME_NC),
)
def test_positive_families_realize_exact_unique_top_k_gap(family_id: str) -> None:
    fixture = _fixture(family_id)
    ranking = sorted(range(fixture.n), key=lambda index: -fixture.angles[index])
    selected = tuple(sorted(ranking[: fixture.k]))
    realized = fixture.angles[ranking[fixture.k - 1]] - fixture.angles[ranking[fixture.k]]

    assert fixture.top_k_membership == selected
    assert not fixture.non_unique_output
    assert fixture.tie_indices == ()
    assert math.isclose(realized, fixture.design_gap, rel_tol=0.0, abs_tol=2e-14)


def test_family_profiles_have_the_registered_structural_roles() -> None:
    equal = _fixture(F_EQ)
    dyadic = _fixture(F_DYADIC)
    cluster = _fixture(F_CLUSTER)
    hidden = _fixture(F_HIDDEN_FRONTIER)
    unknown = _fixture(F_UNKNOWN_TIME_NC)
    half_gap = equal.design_gap / 2.0

    assert equal.nested_count_profile.radii == pytest.approx((half_gap,))
    assert equal.nested_count_profile.active_counts == (equal.n,)
    assert len(dyadic.nested_count_profile.radii) >= 4
    assert dyadic.nested_count_profile.radii[0] / half_gap > 8.0
    assert max(abs(angle - cluster.hidden_center) for angle in cluster.angles) <= (
        2.35 * half_gap
    )
    assert len(set(cluster.angles)) == cluster.n
    assert len(set(hidden.angles)) == hidden.n
    assert hidden.nested_count_profile.active_counts[0] == hidden.n
    assert hidden.nested_count_profile.active_counts[-1] == 2
    assert sum(hidden.nested_count_profile.new_selected_counts) == hidden.k
    assert len(unknown.nested_count_profile.radii) >= 6
    assert unknown.nested_count_profile.radii[0] / half_gap > 32.0


def test_nested_counts_and_static_annuli_are_recomputed_from_truth() -> None:
    fixture = _fixture()
    profile = fixture.nested_count_profile
    selected = set(fixture.top_k_membership or ())

    for radius, count in zip(profile.radii, profile.active_counts, strict=True):
        assert count == sum(
            abs(angle - fixture.hidden_center) <= radius + 1e-15
            for angle in fixture.angles
        )
    assert sorted(index for block in profile.static_blocks for index in block) == list(
        range(fixture.n)
    )
    assert profile.new_selected_counts == tuple(
        sum(index in selected for index in block) for block in profile.static_blocks
    )


def test_public_partition_is_unordered_and_contains_no_truth_labels() -> None:
    fixture = _fixture(F_PUBLIC_PARTITION)
    view = fixture.algorithm_view()
    blocks = tuple(tuple(block) for block in view["static_partition"])

    assert sorted(index for block in blocks for index in block) == list(range(fixture.n))
    assert blocks == tuple(sorted(blocks, key=lambda block: (block[0], len(block), block)))
    assert "new_selected_counts" not in json.dumps(view, sort_keys=True)
    assert "radii" not in json.dumps(view, sort_keys=True)


def test_tie_control_is_explicitly_nonunique_and_has_zero_realized_gap() -> None:
    fixture = _fixture(F_TIE_NC)
    ranking = fixture.ranking

    assert fixture.non_unique_output
    assert fixture.top_k_membership is None
    assert fixture.realized_boundary_gap == 0.0
    assert len(fixture.tie_indices) == 2
    assert fixture.angles[ranking[fixture.k - 1]] == fixture.angles[ranking[fixture.k]]
    promise = fixture.algorithm_view()["structural_promise"]
    assert promise["relation"] == "complete_strict_top_k_or_fail_closed_rejection"
    assert promise["boundary_information"] == "not_supplied"


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"family_id": "unknown"}, ValueError, "family_id"),
        ({"n": 3}, ValueError, "n"),
        ({"k": 0}, ValueError, "k"),
        ({"k": 32}, ValueError, "smaller"),
        ({"design_gap": 0.0}, ValueError, "positive"),
        ({"design_gap": math.pi / 64.0}, ValueError, "pi/96"),
        ({"fixture_seed": -1}, ValueError, "fixture_seed"),
        ({"delta": 0.0}, ValueError, "positive"),
        ({"delta": 1.0}, ValueError, "delta"),
        ({"hard_query_cap": 0}, ValueError, "hard_query_cap"),
        ({"permutation_seed": -1}, ValueError, "permutation_seed"),
        ({"n": True}, TypeError, "not bool"),
    ],
)
def test_invalid_generator_inputs_fail_closed(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "family_id": F_HIDDEN_FRONTIER,
        "n": 32,
        "k": 6,
        "design_gap": math.pi / 256.0,
        "fixture_seed": 1,
        "delta": 0.05,
        "hard_query_cap": 1024,
    }
    arguments.update(overrides)

    with pytest.raises(error, match=message):
        generate_hidden_frontier_fixture(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tampered",
    [
        lambda fixture: replace(
            fixture, angles=(fixture.angles[1], fixture.angles[0], *fixture.angles[2:])
        ),
        lambda fixture: replace(fixture, fixture_hash="0" * 64),
        lambda fixture: replace(fixture, orbit_hash="0" * 64),
        lambda fixture: replace(fixture, interface_id="0" * 64),
        lambda fixture: replace(
            fixture,
            nested_count_profile=replace(
                fixture.nested_count_profile,
                active_counts=(0, *fixture.nested_count_profile.active_counts[1:]),
            ),
        ),
    ],
)
def test_truth_profile_and_hash_tampering_is_rejected(tampered) -> None:
    fixture = _fixture()

    with pytest.raises((ValueError, RuntimeError)):
        validate_hidden_frontier_fixture(tampered(fixture))


def test_canonical_hash_and_deduplicator_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="exactly n"):
        canonical_orbit_hash((0.2, 0.3), n=4, k=1, non_unique_output=False)
    with pytest.raises(ValueError, match=r"\[0, pi/2\]"):
        canonical_orbit_hash((-0.1, 0.2, 0.3, 0.4), n=4, k=1, non_unique_output=False)
    with pytest.raises(TypeError, match="sequence"):
        deduplicate_isomorphic_fixtures("not-fixtures")  # type: ignore[arg-type]


def test_private_manifest_commits_to_every_hidden_field() -> None:
    fixture = _fixture()
    private = fixture.private_interface_document()
    public = fixture.algorithm_view()

    assert private["family_id"] == fixture.family_id
    assert private["angles"] == list(fixture.angles)
    assert private["hidden_center"] == fixture.hidden_center
    assert private["hidden_permutation"] == list(fixture.hidden_permutation)
    assert private["ranking"] == list(fixture.ranking)
    assert private["top_k_membership"] == list(fixture.top_k_membership or ())
    assert private["nested_count_profile"]["active_counts"] == list(
        fixture.nested_count_profile.active_counts
    )
    assert not set(private).issubset(public)


def test_small_exact_state_grid_is_supported_for_every_family() -> None:
    for family_id in FAMILY_IDS:
        for n in (4, 6, 8):
            for k in range(1, n):
                fixture = _fixture(family_id, n=n, k=k, seed=n * 100 + k)
                fixture.validate()


def test_nested_count_profile_type_cannot_hide_duplicate_indices() -> None:
    fixture = _fixture()
    profile = fixture.nested_count_profile
    duplicated = NestedCountProfile(
        radii=profile.radii,
        active_counts=profile.active_counts,
        new_selected_counts=profile.new_selected_counts,
        static_blocks=((0, 0, *profile.static_blocks[0][1:]), *profile.static_blocks[1:]),
    )

    with pytest.raises(ValueError, match="partition"):
        replace(fixture, nested_count_profile=duplicated).validate()
