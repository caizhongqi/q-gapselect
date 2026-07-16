"""Frozen hidden-frontier fixtures for unknown-boundary Top-k experiments.

The trusted harness owns every value in :class:`FrozenHiddenFrontierFixture`.
An algorithm receives only :meth:`FrozenHiddenFrontierFixture.algorithm_view`.
In particular, the blind interface never serializes the angle vector, center,
permutation, ranking, selected membership, active identities, or stopping
schedule.  ``F-PUBLIC-PARTITION`` is the one stronger-information control: it
publishes an *unordered static partition*, but still withholds all of the
truth and schedule fields above.

This module only generates deterministic synthetic fixtures.  It implements
neither an oracle nor a selection algorithm, and makes no quantum-advantage
claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

F_EQ = "F-EQ"
F_DYADIC = "F-DYADIC"
F_CLUSTER = "F-CLUSTER"
F_HIDDEN_FRONTIER = "F-HIDDEN-FRONTIER"
F_PUBLIC_PARTITION = "F-PUBLIC-PARTITION"
F_UNKNOWN_TIME_NC = "F-UNKNOWN-TIME-NC"
F_TIE_NC = "F-TIE-NC"

FAMILY_IDS = (
    F_EQ,
    F_DYADIC,
    F_CLUSTER,
    F_HIDDEN_FRONTIER,
    F_PUBLIC_PARTITION,
    F_UNKNOWN_TIME_NC,
    F_TIE_NC,
)

CENTER_GRID = tuple(math.pi / 6.0 + index * math.pi / 96.0 for index in range(17))
GENERATOR_VERSION = "qgapselect.hidden-frontier-fixture.v1"
BLIND_INTERFACE_SCHEMA = "qgapselect.blind-canonical-topk-interface.v1"
PUBLIC_PARTITION_INTERFACE_SCHEMA = (
    "qgapselect.public-static-partition-topk-interface.v1"
)


def _canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(document: object) -> str:
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _derived_seed(seed: int, family_id: str, purpose: str) -> int:
    # Hidden and public-partition controls intentionally share one latent
    # construction and permutation for equal user seeds.
    structure_family = (
        F_HIDDEN_FRONTIER
        if family_id in {F_HIDDEN_FRONTIER, F_PUBLIC_PARTITION}
        else family_id
    )
    material = (
        f"{GENERATOR_VERSION}\0{seed}\0{structure_family}\0{purpose}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _float_hexes(values: Iterable[float]) -> list[str]:
    return [float(value).hex() for value in values]


@dataclass(frozen=True, slots=True)
class HiddenFrontierSpec:
    """Public experiment controls used to freeze one trusted fixture."""

    family_id: str
    n: int
    k: int
    design_gap: float
    fixture_seed: int
    delta: float
    hard_query_cap: int
    permutation_seed: int | None = None

    def normalized(self) -> HiddenFrontierSpec:
        if self.family_id not in FAMILY_IDS:
            raise ValueError(f"family_id must be one of {FAMILY_IDS}")
        n = _integer(self.n, "n", minimum=4)
        k = _integer(self.k, "k", minimum=1)
        if k >= n:
            raise ValueError("k must be smaller than n")
        design_gap = _finite_real(self.design_gap, "design_gap", positive=True)
        # beta is always in [pi/6, pi/3].  This conservative cap leaves room
        # for every registered construction on both sides of beta.
        if design_gap > math.pi / 96.0:
            raise ValueError("design_gap must not exceed pi/96 for the frozen families")
        fixture_seed = _integer(self.fixture_seed, "fixture_seed", minimum=0)
        delta = _finite_real(self.delta, "delta", positive=True)
        if delta >= 1.0:
            raise ValueError("delta must lie in (0, 1)")
        hard_query_cap = _integer(
            self.hard_query_cap, "hard_query_cap", minimum=1
        )
        if self.permutation_seed is None:
            permutation_seed = _derived_seed(
                fixture_seed, self.family_id, "permutation"
            )
        else:
            permutation_seed = _integer(
                self.permutation_seed, "permutation_seed", minimum=0
            )
        return HiddenFrontierSpec(
            family_id=self.family_id,
            n=n,
            k=k,
            design_gap=design_gap,
            fixture_seed=fixture_seed,
            delta=delta,
            hard_query_cap=hard_query_cap,
            permutation_seed=permutation_seed,
        )


@dataclass(frozen=True, slots=True)
class NestedCountProfile:
    """Runner-only nested active counts and newly finalized selected counts.

    ``active_counts[r]`` equals the number of arms in
    ``{|theta_i-beta| <= radii[r]}``.  ``new_selected_counts`` counts selected
    arms finalized after the preceding active set is removed; its final entry
    includes every selected arm remaining in the last active set.  Therefore
    it always sums to ``k``.  The blocks are stored in algorithm-index space
    solely for trusted validation and for the public-partition control.
    """

    radii: tuple[float, ...]
    active_counts: tuple[int, ...]
    new_selected_counts: tuple[int, ...]
    static_blocks: tuple[tuple[int, ...], ...]

    def private_document(self) -> dict[str, object]:
        return {
            "radii": list(self.radii),
            "active_counts": list(self.active_counts),
            "new_selected_counts": list(self.new_selected_counts),
            "static_shell_blocks": [list(block) for block in self.static_blocks],
        }


@dataclass(frozen=True, slots=True)
class FrozenHiddenFrontierFixture:
    """One immutable trusted fixture and its non-leaking algorithm interface."""

    family_id: str
    n: int
    k: int
    design_gap: float
    realized_boundary_gap: float
    fixture_seed: int
    permutation_seed: int
    delta: float
    hard_query_cap: int
    hidden_center: float
    hidden_permutation: tuple[int, ...]
    angles: tuple[float, ...]
    latent_angles: tuple[float, ...]
    ranking: tuple[int, ...]
    top_k_membership: tuple[int, ...] | None
    nested_count_profile: NestedCountProfile
    non_unique_output: bool
    tie_indices: tuple[int, ...]
    orbit_hash: str
    interface_id: str
    fixture_hash: str

    def public_interface_document(self) -> dict[str, object]:
        """Serialize exactly what a method may receive before oracle access."""

        structural_promise: dict[str, object] = {
            "relation": "complete_strict_top_k_or_fail_closed_rejection",
            "boundary_information": "not_supplied",
            "valid_index_count": self.n,
            "padding_action": "identity_and_never_returnable",
        }
        schema = BLIND_INTERFACE_SCHEMA
        document: dict[str, object] = {
            "schema": schema,
            "n": self.n,
            "k": self.k,
            "delta": self.delta,
            "hard_query_cap": self.hard_query_cap,
            "oracle_access": {
                "base": "canonical_bernoulli_block_rotation",
                "allowed_calls": [
                    "B",
                    "B_dagger",
                    "controlled_B",
                    "controlled_B_dagger",
                ],
                "logical_query_cost_per_base_call": 1,
            },
            "structural_promise": structural_promise,
        }
        if self.family_id == F_PUBLIC_PARTITION:
            schema = PUBLIC_PARTITION_INTERFACE_SCHEMA
            document["schema"] = schema
            # Block order is canonicalized by public arm IDs and is not an
            # epoch order, ranking, active list, or stopping-time schedule.
            document["static_partition"] = [
                list(block)
                for block in sorted(
                    self.nested_count_profile.static_blocks,
                    key=lambda block: (block[0], len(block), block),
                )
            ]
            document["partition_semantics"] = (
                "unordered_static_blocks_without_truth_or_epoch_order"
            )
        return document

    def algorithm_view(self) -> dict[str, object]:
        """Return a fresh serialization of the method-visible interface."""

        document = self.public_interface_document()
        expected = _interface_id(document)
        if expected != self.interface_id:
            raise RuntimeError("stored interface_id does not match public interface")
        document["interface_id"] = self.interface_id
        _validate_algorithm_document(
            document, allow_static_partition=self.family_id == F_PUBLIC_PARTITION
        )
        return document

    def private_interface_document(self) -> dict[str, object]:
        """Serialize runner-only truth; never pass this document to a method."""

        return {
            "schema": f"{GENERATOR_VERSION}.private-manifest",
            "family_id": self.family_id,
            "n": self.n,
            "k": self.k,
            "design_gap": self.design_gap,
            "realized_boundary_gap": self.realized_boundary_gap,
            "fixture_seed": self.fixture_seed,
            "permutation_seed": self.permutation_seed,
            "delta": self.delta,
            "hard_query_cap": self.hard_query_cap,
            "hidden_center": self.hidden_center,
            "hidden_permutation": list(self.hidden_permutation),
            "angles": list(self.angles),
            "latent_angles": list(self.latent_angles),
            "ranking": list(self.ranking),
            "top_k_membership": (
                None
                if self.top_k_membership is None
                else list(self.top_k_membership)
            ),
            "nested_count_profile": self.nested_count_profile.private_document(),
            "non_unique_output": self.non_unique_output,
            "tie_indices": list(self.tie_indices),
            "orbit_hash": self.orbit_hash,
            "interface_id": self.interface_id,
        }

    def trusted_manifest_document(self) -> dict[str, object]:
        document = self.private_interface_document()
        document["fixture_hash"] = self.fixture_hash
        return document

    def replay(self) -> FrozenHiddenFrontierFixture:
        """Regenerate and verify this fixture from its frozen provenance."""

        replayed = generate_hidden_frontier_fixture(
            family_id=self.family_id,
            n=self.n,
            k=self.k,
            design_gap=self.design_gap,
            fixture_seed=self.fixture_seed,
            delta=self.delta,
            hard_query_cap=self.hard_query_cap,
            permutation_seed=self.permutation_seed,
        )
        if replayed.fixture_hash != self.fixture_hash:
            raise RuntimeError("deterministic fixture replay hash mismatch")
        return replayed

    def validate(self) -> None:
        validate_hidden_frontier_fixture(self)


def canonical_orbit_hash(
    angles: Sequence[float],
    *,
    n: int,
    k: int,
    non_unique_output: bool,
) -> str:
    """Hash a raw geometric permutation orbit, independent of information access.

    This hash deliberately does not commit to ``interface_id``.  It can group
    the hidden/public-partition pair as one angle orbit for a geometry audit,
    but it is not by itself a valid experimental-fixture deduplication key.
    """

    resolved_n = _integer(n, "n", minimum=2)
    resolved_k = _integer(k, "k", minimum=1)
    if resolved_k >= resolved_n:
        raise ValueError("k must be smaller than n")
    values = tuple(_finite_real(value, "angle") for value in angles)
    if len(values) != resolved_n:
        raise ValueError("angles must contain exactly n values")
    if any(value < 0.0 or value > math.pi / 2.0 for value in values):
        raise ValueError("angles must lie in [0, pi/2]")
    if not isinstance(non_unique_output, bool):
        raise TypeError("non_unique_output must be bool")
    return _canonical_hash(
        {
            "schema": f"{GENERATOR_VERSION}.canonical-orbit",
            "n": resolved_n,
            "k": resolved_k,
            "non_unique_output": non_unique_output,
            "sorted_angle_hex": sorted(_float_hexes(values)),
        }
    )


def _interface_id(document: Mapping[str, object]) -> str:
    return _canonical_hash(
        {"schema": f"{GENERATOR_VERSION}.interface-id", "document": document}
    )


def _tiered_side_magnitudes(
    count: int,
    *,
    factors: Sequence[float],
    half_gap: float,
    phase: int,
    jitter: float,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if count == 0:
        return (), ()
    magnitudes = [half_gap]
    tiers = [0]
    for ordinal in range(1, count):
        tier = 1 + ((ordinal - 1 + phase) % (len(factors) - 1))
        # A strictly ordinal jitter is stable, tie-free within each side, and
        # stays well inside the factor separation.  The boundary tier itself
        # is never jittered.
        fraction = (ordinal + phase / len(factors)) / (count + 1)
        multiplier = 1.0 + jitter * (2.0 * fraction - 1.0)
        magnitudes.append(half_gap * factors[tier] * multiplier)
        tiers.append(tier)
    return tuple(magnitudes), tuple(tiers)


def _latent_construction(
    spec: HiddenFrontierSpec,
    center: float,
) -> tuple[tuple[float, ...], tuple[int, ...], bool]:
    """Return latent angles, private shell tiers, and non-unique flag."""

    half_gap = spec.design_gap / 2.0
    selected_count = spec.k
    rejected_count = spec.n - spec.k
    family = spec.family_id

    if family == F_TIE_NC:
        above_count = max(0, selected_count - 1)
        above_max = min(
            half_gap * (2.0 + 0.25 * max(0, above_count - 1)),
            0.80 * (math.pi / 2.0 - center),
        )
        above_step = (
            (above_max - 2.0 * half_gap) / max(1, above_count - 1)
            if above_count
            else 0.0
        )
        above = tuple(
            center + 2.0 * half_gap + above_step * ordinal
            for ordinal in range(above_count)
        )
        tie = (center, center)
        below_count = spec.n - len(above) - len(tie)
        below_max = min(
            half_gap * (2.0 + 0.25 * max(0, below_count - 1)),
            0.80 * center,
        )
        below_step = (
            (below_max - 2.0 * half_gap) / max(1, below_count - 1)
            if below_count
            else 0.0
        )
        below = tuple(
            center - 2.0 * half_gap - below_step * ordinal
            for ordinal in range(below_count)
        )
        # The tie label remains trusted-only.  A coarse private two-annulus
        # profile is sufficient for diagnostics and avoids manufacturing an
        # instance-specific stopping schedule for this negative control.
        tiers = (1,) * len(above) + (0, 0) + (1,) * len(below)
        return above + tie + below, tiers, True

    if family == F_EQ:
        selected_magnitudes = (half_gap,) * selected_count
        rejected_magnitudes = (half_gap,) * rejected_count
        selected_tiers = (0,) * selected_count
        rejected_tiers = (0,) * rejected_count
    else:
        if family == F_DYADIC:
            factors = (1.0, 2.0, 4.0, 8.0, 16.0)
            jitter = 0.04
        elif family == F_CLUSTER:
            factors = (1.0, 1.20, 1.48, 1.82, 2.25)
            jitter = 0.025
        elif family in {F_HIDDEN_FRONTIER, F_PUBLIC_PARTITION}:
            factors = (1.0, 2.0, 4.0, 8.0, 16.0)
            jitter = 0.03
        elif family == F_UNKNOWN_TIME_NC:
            factors = (1.0, 2.25, 4.5, 9.0, 18.0, 36.0, 64.0)
            jitter = 0.025
        else:  # pragma: no cover - normalized spec makes this unreachable.
            raise ValueError(f"unsupported family {family!r}")
        safe_factor = (
            0.90 * min(center, math.pi / 2.0 - center) / half_gap
        )
        factors = tuple(
            factor for factor in factors if factor * (1.0 + jitter) <= safe_factor
        )
        if len(factors) < 3:
            raise ValueError("design_gap leaves too little room for nested family shells")
        selected_magnitudes, selected_tiers = _tiered_side_magnitudes(
            selected_count,
            factors=factors,
            half_gap=half_gap,
            phase=0,
            jitter=jitter,
        )
        rejected_magnitudes, rejected_tiers = _tiered_side_magnitudes(
            rejected_count,
            factors=factors,
            half_gap=half_gap,
            phase=1,
            jitter=jitter,
        )

    latent_angles = tuple(center + value for value in selected_magnitudes) + tuple(
        center - value for value in rejected_magnitudes
    )
    latent_tiers = selected_tiers + rejected_tiers
    return latent_angles, latent_tiers, False


def _nested_profile(
    *,
    angles: tuple[float, ...],
    center: float,
    tiers: tuple[int, ...],
    k: int,
    top_k: tuple[int, ...] | None,
) -> NestedCountProfile:
    tier_values = sorted(set(tiers), reverse=True)
    tier_blocks = tuple(
        tuple(index for index, tier in enumerate(tiers) if tier == tier_value)
        for tier_value in tier_values
    )
    # TIE-NC uses one arm per synthetic tier.  Sorting the exact outer radii
    # also makes its diagnostic profile independent of latent arm ordering.
    radii = tuple(
        sorted(
            {
                max(abs(angles[index] - center) for index in block)
                for block in tier_blocks
            },
            reverse=True,
        )
    )
    annular_blocks: list[tuple[int, ...]] = []
    for position, radius in enumerate(radii):
        next_radius = radii[position + 1] if position + 1 < len(radii) else -1.0
        annular_blocks.append(
            tuple(
                index
                for index, angle in enumerate(angles)
                if next_radius + 1e-15 < abs(angle - center) <= radius + 1e-15
            )
        )
    active_counts = tuple(
        sum(abs(angle - center) <= radius + 1e-15 for angle in angles)
        for radius in radii
    )
    selected = set(top_k or ())
    new_selected = [
        sum(index in selected for index in block) for block in annular_blocks
    ]
    if top_k is None:
        new_selected = [0] * len(annular_blocks)
    elif sum(new_selected) != k:
        raise RuntimeError("nested selected-output profile does not sum to k")
    return NestedCountProfile(
        radii=radii,
        active_counts=active_counts,
        new_selected_counts=tuple(new_selected),
        static_blocks=tuple(annular_blocks),
    )


def generate_hidden_frontier_fixture(
    *,
    family_id: str,
    n: int,
    k: int,
    design_gap: float,
    fixture_seed: int,
    delta: float = 0.05,
    hard_query_cap: int = 1_048_576,
    permutation_seed: int | None = None,
) -> FrozenHiddenFrontierFixture:
    """Generate one deterministic, replayable, runner-owned fixture."""

    spec = HiddenFrontierSpec(
        family_id=family_id,
        n=n,
        k=k,
        design_gap=design_gap,
        fixture_seed=fixture_seed,
        delta=delta,
        hard_query_cap=hard_query_cap,
        permutation_seed=permutation_seed,
    ).normalized()
    assert spec.permutation_seed is not None

    center_rng = random.Random(_derived_seed(spec.fixture_seed, spec.family_id, "center"))
    center = CENTER_GRID[center_rng.randrange(len(CENTER_GRID))]
    latent_angles, latent_tiers, non_unique = _latent_construction(spec, center)
    if any(angle < 0.0 or angle > math.pi / 2.0 for angle in latent_angles):
        raise ValueError("design_gap and family profile place an arm outside [0, pi/2]")

    permutation = list(range(spec.n))
    random.Random(spec.permutation_seed).shuffle(permutation)
    hidden_permutation = tuple(permutation)
    angles = tuple(latent_angles[index] for index in hidden_permutation)
    tiers = tuple(latent_tiers[index] for index in hidden_permutation)
    ranking = tuple(sorted(range(spec.n), key=lambda index: (-angles[index], index)))
    boundary_gap = angles[ranking[spec.k - 1]] - angles[ranking[spec.k]]
    top_k = None if non_unique else tuple(sorted(ranking[: spec.k]))
    boundary_value = angles[ranking[spec.k - 1]]
    tie_indices = tuple(
        sorted(index for index, value in enumerate(angles) if value == boundary_value)
    )
    if not non_unique:
        tie_indices = ()

    profile = _nested_profile(
        angles=angles,
        center=center,
        tiers=tiers,
        k=spec.k,
        top_k=top_k,
    )
    orbit_hash = canonical_orbit_hash(
        angles,
        n=spec.n,
        k=spec.k,
        non_unique_output=non_unique,
    )

    provisional = FrozenHiddenFrontierFixture(
        family_id=spec.family_id,
        n=spec.n,
        k=spec.k,
        design_gap=spec.design_gap,
        realized_boundary_gap=boundary_gap,
        fixture_seed=spec.fixture_seed,
        permutation_seed=spec.permutation_seed,
        delta=spec.delta,
        hard_query_cap=spec.hard_query_cap,
        hidden_center=center,
        hidden_permutation=hidden_permutation,
        angles=angles,
        latent_angles=latent_angles,
        ranking=ranking,
        top_k_membership=top_k,
        nested_count_profile=profile,
        non_unique_output=non_unique,
        tie_indices=tie_indices,
        orbit_hash=orbit_hash,
        interface_id="",
        fixture_hash="",
    )
    interface_id = _interface_id(provisional.public_interface_document())
    with_interface = FrozenHiddenFrontierFixture(
        **{
            field: getattr(provisional, field)
            for field in provisional.__dataclass_fields__
            if field not in {"interface_id", "fixture_hash"}
        },
        interface_id=interface_id,
        fixture_hash="",
    )
    fixture_hash = _canonical_hash(with_interface.private_interface_document())
    fixture = FrozenHiddenFrontierFixture(
        **{
            field: getattr(with_interface, field)
            for field in with_interface.__dataclass_fields__
            if field != "fixture_hash"
        },
        fixture_hash=fixture_hash,
    )
    validate_hidden_frontier_fixture(fixture)
    return fixture


def _walk_keys_and_values(value: object) -> Iterable[tuple[str | None, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_keys_and_values(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield None, child
            yield from _walk_keys_and_values(child)


def _validate_algorithm_document(
    document: Mapping[str, object], *, allow_static_partition: bool
) -> None:
    forbidden_tokens = {
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
        "radii",
    }
    for key, _ in _walk_keys_and_values(document):
        if key is None:
            continue
        lowered = key.lower()
        if any(token in lowered for token in forbidden_tokens):
            # The center *domain* is a public range, not the numeric center;
            # avoid even that key token so leakage audits can remain simple.
            raise RuntimeError(f"algorithm interface leaks forbidden field {key!r}")
    if "static_partition" in document and not allow_static_partition:
        raise RuntimeError("blind interface cannot contain a static partition")
    expected = {
        "schema",
        "n",
        "k",
        "delta",
        "hard_query_cap",
        "oracle_access",
        "structural_promise",
        "interface_id",
    }
    if allow_static_partition:
        expected |= {"static_partition", "partition_semantics"}
    if set(document) != expected:
        raise RuntimeError("algorithm interface contains undeclared top-level fields")


def validate_hidden_frontier_fixture(fixture: FrozenHiddenFrontierFixture) -> None:
    """Fail closed if truth, profile, hashes, or public isolation is inconsistent."""

    if not isinstance(fixture, FrozenHiddenFrontierFixture):
        raise TypeError("fixture must be a FrozenHiddenFrontierFixture")
    spec = HiddenFrontierSpec(
        family_id=fixture.family_id,
        n=fixture.n,
        k=fixture.k,
        design_gap=fixture.design_gap,
        fixture_seed=fixture.fixture_seed,
        delta=fixture.delta,
        hard_query_cap=fixture.hard_query_cap,
        permutation_seed=fixture.permutation_seed,
    ).normalized()
    if len(fixture.angles) != spec.n or len(fixture.latent_angles) != spec.n:
        raise ValueError("angle vectors must contain exactly n values")
    if any(angle < 0.0 or angle > math.pi / 2.0 for angle in fixture.angles):
        raise ValueError("angles must lie in [0, pi/2]")
    if fixture.hidden_center not in CENTER_GRID:
        raise ValueError("hidden_center is outside the frozen center grid")
    if sorted(fixture.hidden_permutation) != list(range(spec.n)):
        raise ValueError("hidden_permutation must be a permutation of range(n)")
    reconstructed = tuple(
        fixture.latent_angles[index] for index in fixture.hidden_permutation
    )
    if reconstructed != fixture.angles:
        raise ValueError("hidden_permutation does not reconstruct angle vector")
    expected_ranking = tuple(
        sorted(range(spec.n), key=lambda index: (-fixture.angles[index], index))
    )
    if fixture.ranking != expected_ranking:
        raise ValueError("ranking is inconsistent with angle vector")
    realized_gap = (
        fixture.angles[expected_ranking[spec.k - 1]]
        - fixture.angles[expected_ranking[spec.k]]
    )
    if not math.isclose(
        realized_gap, fixture.realized_boundary_gap, rel_tol=0.0, abs_tol=1e-14
    ):
        raise ValueError("realized_boundary_gap is inconsistent with angles")

    if fixture.family_id == F_TIE_NC:
        if not fixture.non_unique_output or fixture.top_k_membership is not None:
            raise ValueError("F-TIE-NC must carry a non-unique-output label")
        if abs(realized_gap) > 1e-14 or len(fixture.tie_indices) < 2:
            raise ValueError("F-TIE-NC must tie the kth and (k+1)th arms")
    else:
        if fixture.non_unique_output or fixture.tie_indices:
            raise ValueError("positive families cannot carry a tie label")
        if not math.isclose(
            realized_gap, spec.design_gap, rel_tol=0.0, abs_tol=2e-14
        ):
            raise ValueError("positive-family boundary gap must equal design_gap")
        expected_top_k = tuple(sorted(expected_ranking[: spec.k]))
        if fixture.top_k_membership != expected_top_k:
            raise ValueError("top_k_membership is inconsistent with ranking")

    profile = fixture.nested_count_profile
    if not profile.radii:
        raise ValueError("nested count profile cannot be empty")
    if any(
        left <= right
        for left, right in zip(profile.radii, profile.radii[1:], strict=False)
    ):
        raise ValueError("nested radii must be strictly decreasing")
    expected_active = tuple(
        sum(
            abs(angle - fixture.hidden_center) <= radius + 1e-15
            for angle in fixture.angles
        )
        for radius in profile.radii
    )
    if profile.active_counts != expected_active:
        raise ValueError("nested active counts do not match angle vector")
    if profile.active_counts[0] != spec.n:
        raise ValueError("outer nested radius must contain every arm")
    if any(
        left < right
        for left, right in zip(
            profile.active_counts, profile.active_counts[1:], strict=False
        )
    ):
        raise ValueError("nested active counts must be nonincreasing")
    flattened = tuple(index for block in profile.static_blocks for index in block)
    if sorted(flattened) != list(range(spec.n)) or len(flattened) != spec.n:
        raise ValueError("static blocks must partition all algorithm indices")
    if len(profile.static_blocks) != len(profile.radii):
        raise ValueError("one static annulus is required for each nested radius")
    for position, (radius, block) in enumerate(
        zip(profile.radii, profile.static_blocks, strict=True)
    ):
        next_radius = (
            profile.radii[position + 1]
            if position + 1 < len(profile.radii)
            else -1.0
        )
        if not block or any(
            not (
                next_radius + 1e-15
                < abs(fixture.angles[index] - fixture.hidden_center)
                <= radius + 1e-15
            )
            for index in block
        ):
            raise ValueError("static blocks do not match the nested annuli")
    selected = set(fixture.top_k_membership or ())
    expected_new = tuple(
        sum(index in selected for index in block) for block in profile.static_blocks
    )
    if profile.new_selected_counts != expected_new:
        raise ValueError("new-selected count profile is inconsistent with truth")
    if not fixture.non_unique_output and sum(profile.new_selected_counts) != spec.k:
        raise ValueError("positive nested output counts must sum to k")

    magnitudes = tuple(
        abs(angle - fixture.hidden_center) for angle in fixture.angles
    )
    half_gap = spec.design_gap / 2.0
    if fixture.family_id == F_EQ and any(
        not math.isclose(value, half_gap, rel_tol=0.0, abs_tol=2e-14)
        for value in magnitudes
    ):
        raise ValueError("F-EQ must place every arm on the common boundary scale")
    if fixture.family_id not in {F_EQ, F_TIE_NC} and len(set(fixture.angles)) != spec.n:
        raise ValueError("registered tie-free family contains duplicate arm values")
    if fixture.family_id == F_CLUSTER and max(magnitudes) > 2.35 * half_gap:
        raise ValueError("F-CLUSTER exceeds its registered constant-width cluster")
    if fixture.family_id in {F_HIDDEN_FRONTIER, F_PUBLIC_PARTITION}:
        if not math.isclose(
            profile.radii[-1], half_gap, rel_tol=0.0, abs_tol=2e-14
        ):
            raise ValueError("hidden-frontier final radius must equal design_gap/2")
        if len(profile.radii) < min(3, spec.n - 1):
            raise ValueError("hidden-frontier profile has too few nested epochs")
    if fixture.family_id == F_UNKNOWN_TIME_NC:
        if profile.radii[0] / profile.radii[-1] < 4.0 - 1e-12:
            raise ValueError("unknown-time control must realize a heavy-tailed radius range")

    expected_orbit = canonical_orbit_hash(
        fixture.angles,
        n=spec.n,
        k=spec.k,
        non_unique_output=fixture.non_unique_output,
    )
    if fixture.orbit_hash != expected_orbit:
        raise ValueError("canonical orbit hash mismatch")
    public = fixture.public_interface_document()
    expected_interface = _interface_id(public)
    if fixture.interface_id != expected_interface:
        raise ValueError("public interface hash mismatch")
    view = fixture.algorithm_view()
    _validate_algorithm_document(
        view, allow_static_partition=fixture.family_id == F_PUBLIC_PARTITION
    )
    expected_fixture_hash = _canonical_hash(fixture.private_interface_document())
    if fixture.fixture_hash != expected_fixture_hash:
        raise ValueError("trusted fixture hash mismatch")


def algorithmic_fixture_deduplication_key(
    fixture: FrozenHiddenFrontierFixture,
) -> tuple[str, str]:
    """Return the geometry-and-information key for one algorithmic fixture.

    Equal raw angle orbits under different interfaces are different
    experiments.  In particular, this key retains both members of every
    hidden/public-partition pair.
    """

    if not isinstance(fixture, FrozenHiddenFrontierFixture):
        raise TypeError("fixture must be a FrozenHiddenFrontierFixture")
    return fixture.orbit_hash, fixture.interface_id


def deduplicate_isomorphic_fixtures(
    fixtures: Sequence[FrozenHiddenFrontierFixture],
) -> tuple[FrozenHiddenFrontierFixture, ...]:
    """Preserve the first fixture from each algorithmic isomorphism class.

    Algorithmic isomorphism requires both an equal exact permutation orbit and
    an equal information interface.  Use :func:`canonical_orbit_hash` alone
    only for raw geometric grouping, never to drop a stronger- or
    weaker-information control.
    """

    if isinstance(fixtures, str | bytes) or not isinstance(fixtures, Sequence):
        raise TypeError("fixtures must be a sequence of frozen fixtures")
    unique: list[FrozenHiddenFrontierFixture] = []
    seen: set[tuple[str, str]] = set()
    for fixture in fixtures:
        validate_hidden_frontier_fixture(fixture)
        key = algorithmic_fixture_deduplication_key(fixture)
        if key not in seen:
            seen.add(key)
            unique.append(fixture)
    return tuple(unique)


__all__ = [
    "BLIND_INTERFACE_SCHEMA",
    "CENTER_GRID",
    "FAMILY_IDS",
    "F_CLUSTER",
    "F_DYADIC",
    "F_EQ",
    "F_HIDDEN_FRONTIER",
    "F_PUBLIC_PARTITION",
    "F_TIE_NC",
    "F_UNKNOWN_TIME_NC",
    "FrozenHiddenFrontierFixture",
    "HiddenFrontierSpec",
    "NestedCountProfile",
    "algorithmic_fixture_deduplication_key",
    "canonical_orbit_hash",
    "deduplicate_isomorphic_fixtures",
    "generate_hidden_frontier_fixture",
    "validate_hidden_frontier_fixture",
]
