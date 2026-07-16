"""Non-isomorphic frozen Layer-C instance design in angular coordinates.

The generator varies an entire signed angular-gap vector, rather than merely
permuting a fixed multiset of arm means.  It rounds every arm onto an exact
finite Bernoulli table and revalidates the public threshold/gap promise after
rounding.  The result is a synthetic algorithm fixture only; no LLM or quantum
hardware is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .attack_oracles import FrozenSourceFixture
from .exact_count_fixtures import generate_exact_count_fixture

FAMILIES = ("equal_gap", "dyadic_gap", "clustered_boundary")
CLAIM_SCOPE = "synthetic_nonisomorphic_frozen_layer_c_instance_design"


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


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


def _open_probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _positive_angle(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _canonical_hash(document: Mapping[str, object]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derived_seed(case_seed: int, case_id: str, purpose: str) -> int:
    material = (
        f"qgapselect.frozen-layer-c-instance.v1\0{case_seed}\0{case_id}\0{purpose}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _angle_from_count(count: int, table_size: int) -> float:
    return math.asin(math.sqrt(count / table_size))


def _selected_minimum_count(
    threshold_angle: float,
    public_gap_floor: float,
    table_size: int,
) -> int:
    boundary_mean = math.sin(threshold_angle + public_gap_floor) ** 2
    count = max(0, min(table_size, math.ceil(table_size * boundary_mean - 1e-14)))
    while (
        count <= table_size
        and _angle_from_count(count, table_size) - threshold_angle < public_gap_floor - 1e-14
    ):
        count += 1
    if count > table_size:
        raise ValueError("table_size cannot realize the selected-side gap promise")
    return count


def _rejected_maximum_count(
    threshold_angle: float,
    public_gap_floor: float,
    table_size: int,
) -> int:
    boundary_mean = math.sin(threshold_angle - public_gap_floor) ** 2
    count = max(0, min(table_size, math.floor(table_size * boundary_mean + 1e-14)))
    while (
        count >= 0
        and threshold_angle - _angle_from_count(count, table_size) < public_gap_floor - 1e-14
    ):
        count -= 1
    if count < 0:
        raise ValueError("table_size cannot realize the rejected-side gap promise")
    return count


def _family_multiplier(
    family: str,
    *,
    active: bool,
    ordinal: int,
    heterogeneity: float,
    rng: random.Random,
) -> float:
    if active:
        return 1.0 + rng.uniform(0.0, 0.025 * heterogeneity)
    if family == "equal_gap":
        return 1.035 + rng.uniform(0.0, 0.10 * heterogeneity)
    if family == "dyadic_gap":
        tier = 2.0 if ordinal % 2 == 0 else 4.0
        return tier * rng.uniform(1.0 - 0.035 * heterogeneity, 1.0 + 0.035 * heterogeneity)
    if family == "clustered_boundary":
        centers = (1.20, 1.55, 2.15)
        center = centers[ordinal % len(centers)]
        return center * rng.uniform(1.0 - 0.05 * heterogeneity, 1.0 + 0.05 * heterogeneity)
    raise ValueError(f"unknown family {family!r}")


def _side_latent_gaps(
    *,
    family: str,
    count: int,
    active_count: int,
    public_gap_floor: float,
    side_capacity: float,
    base_scale: float,
    heterogeneity: float,
    rng: random.Random,
) -> tuple[float, ...]:
    available = side_capacity - public_gap_floor
    base = min(
        public_gap_floor * base_scale,
        public_gap_floor + 0.20 * available,
    )
    upper = public_gap_floor + 0.98 * available
    values: list[float] = []
    for ordinal in range(count):
        multiplier = _family_multiplier(
            family,
            active=ordinal < active_count,
            ordinal=max(0, ordinal - active_count),
            heterogeneity=heterogeneity,
            rng=rng,
        )
        values.append(min(upper, max(public_gap_floor, base * multiplier)))
    rng.shuffle(values)
    return tuple(values)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


@dataclass(frozen=True, slots=True)
class AngularGapStructureDescriptor:
    """Auditable latent and post-rounding structure of one instance."""

    family: str
    n_arms: int
    k: int
    selected_active_count: int
    rejected_active_count: int
    active_count: int
    base_gap_scale: float
    heterogeneity: float
    latent_boundary_gap: float
    empirical_boundary_gap: float
    empirical_maximum_gap: float
    empirical_mean_gap: float
    empirical_gap_cv: float
    distinct_empirical_gap_count: int
    selected_empirical_gaps: tuple[float, ...]
    rejected_empirical_gaps: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "n_arms": self.n_arms,
            "k": self.k,
            "selected_active_count": self.selected_active_count,
            "rejected_active_count": self.rejected_active_count,
            "active_count": self.active_count,
            "base_gap_scale": self.base_gap_scale,
            "heterogeneity": self.heterogeneity,
            "latent_boundary_gap": self.latent_boundary_gap,
            "empirical_boundary_gap": self.empirical_boundary_gap,
            "empirical_maximum_gap": self.empirical_maximum_gap,
            "empirical_mean_gap": self.empirical_mean_gap,
            "empirical_gap_cv": self.empirical_gap_cv,
            "distinct_empirical_gap_count": self.distinct_empirical_gap_count,
            "selected_empirical_gaps": list(self.selected_empirical_gaps),
            "rejected_empirical_gaps": list(self.rejected_empirical_gaps),
        }


@dataclass(frozen=True, slots=True)
class FrozenQuantumInstanceDesign:
    """One generated frozen fixture plus permutation-invariant difficulty data."""

    case_id: str
    family: str
    case_seed: int
    permutation_seed: int
    threshold: float
    threshold_angle: float
    public_gap_floor: float
    table_size: int
    candidate_ids: tuple[str, ...]
    signed_latent_angular_gaps: tuple[float, ...]
    signed_empirical_angular_gaps: tuple[float, ...]
    selected_arms: tuple[int, ...]
    empirical_success_counts: tuple[int, ...]
    fixture: FrozenSourceFixture
    structure: AngularGapStructureDescriptor
    difficulty_fingerprint: str
    claim_scope: str = CLAIM_SCOPE

    @property
    def n_arms(self) -> int:
        return len(self.candidate_ids)

    @property
    def k(self) -> int:
        return len(self.selected_arms)

    def manifest_document(self) -> dict[str, object]:
        return {
            "schema": "qgapselect.frozen-layer-c-instance-design.v1",
            "case_id": self.case_id,
            "family": self.family,
            "case_seed": self.case_seed,
            "permutation_seed": self.permutation_seed,
            "threshold": self.threshold,
            "public_gap_floor": self.public_gap_floor,
            "table_size": self.table_size,
            "candidate_ids": list(self.candidate_ids),
            "signed_latent_angular_gaps": list(self.signed_latent_angular_gaps),
            "signed_empirical_angular_gaps": list(self.signed_empirical_angular_gaps),
            "selected_arms": list(self.selected_arms),
            "empirical_success_counts": list(self.empirical_success_counts),
            "fixture_manifest_hash": self.fixture.manifest_hash,
            "difficulty_fingerprint": self.difficulty_fingerprint,
            "structure": self.structure.as_dict(),
            "claim_scope": self.claim_scope,
        }


def _validate_post_rounding(
    *,
    frozen_means: tuple[float, ...],
    threshold: float,
    threshold_angle: float,
    public_gap_floor: float,
    k: int,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    signed_gaps = tuple(math.asin(math.sqrt(mean)) - threshold_angle for mean in frozen_means)
    selected = tuple(index for index, mean in enumerate(frozen_means) if mean > threshold)
    if len(selected) != k:
        raise RuntimeError("post-rounding threshold does not separate exactly k arms")
    if any(mean == threshold for mean in frozen_means):
        raise RuntimeError("post-rounding arm lies exactly on the public threshold")
    if min(abs(gap) for gap in signed_gaps) < public_gap_floor - 1e-12:
        raise RuntimeError("post-rounding angular gap violates public_gap_floor")
    return signed_gaps, selected


def generate_frozen_quantum_instance_design(
    *,
    case_id: str,
    family: str,
    n_arms: int,
    k: int,
    threshold: float,
    public_gap_floor: float,
    table_size: int,
    case_seed: int,
    permutation_seed: int | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> FrozenQuantumInstanceDesign:
    """Generate one deterministic, non-permutation-only Layer-C instance."""

    resolved_case_id = _nonempty_string(case_id, "case_id")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    arm_count = _integer(n_arms, "n_arms", minimum=2)
    selected_count = _integer(k, "k", minimum=1)
    if selected_count >= arm_count:
        raise ValueError("k must be smaller than n_arms")
    resolved_threshold = _open_probability(threshold, "threshold")
    threshold_angle = math.asin(math.sqrt(resolved_threshold))
    gap_floor = _positive_angle(public_gap_floor, "public_gap_floor")
    capacity = min(threshold_angle, math.pi / 2.0 - threshold_angle)
    if gap_floor >= capacity:
        raise ValueError("public_gap_floor must be smaller than both angular threshold margins")
    resolved_table_size = _integer(table_size, "table_size", minimum=1)
    resolved_case_seed = _integer(case_seed, "case_seed", minimum=0)
    if permutation_seed is None:
        resolved_permutation_seed = _derived_seed(
            resolved_case_seed,
            resolved_case_id,
            "permutation",
        )
    else:
        resolved_permutation_seed = _integer(
            permutation_seed,
            "permutation_seed",
            minimum=0,
        )
    if candidate_ids is None:
        resolved_candidate_ids = tuple(f"c{index:04d}" for index in range(arm_count))
    else:
        resolved_candidate_ids = tuple(candidate_ids)
        if len(resolved_candidate_ids) != arm_count:
            raise ValueError("candidate_ids must contain exactly n_arms entries")
        if any(not isinstance(value, str) or not value for value in resolved_candidate_ids):
            raise TypeError("candidate_ids must be non-empty strings")
        if len(set(resolved_candidate_ids)) != arm_count:
            raise ValueError("candidate_ids must be unique")

    structure_rng = random.Random(_derived_seed(resolved_case_seed, resolved_case_id, "structure"))
    selected_active = structure_rng.randint(1, selected_count)
    rejected_active = structure_rng.randint(1, arm_count - selected_count)
    base_scale = structure_rng.uniform(1.04, 1.40)
    heterogeneity = structure_rng.uniform(0.65, 1.35)
    selected_latent = _side_latent_gaps(
        family=family,
        count=selected_count,
        active_count=selected_active,
        public_gap_floor=gap_floor,
        side_capacity=math.pi / 2.0 - threshold_angle,
        base_scale=base_scale,
        heterogeneity=heterogeneity,
        rng=structure_rng,
    )
    rejected_latent = _side_latent_gaps(
        family=family,
        count=arm_count - selected_count,
        active_count=rejected_active,
        public_gap_floor=gap_floor,
        side_capacity=threshold_angle,
        base_scale=base_scale,
        heterogeneity=heterogeneity,
        rng=structure_rng,
    )
    canonical_signed_latent = tuple(selected_latent) + tuple(-gap for gap in rejected_latent)
    arm_assignment = list(range(arm_count))
    random.Random(resolved_permutation_seed).shuffle(arm_assignment)
    signed_latent = tuple(canonical_signed_latent[index] for index in arm_assignment)

    selected_minimum_count = _selected_minimum_count(
        threshold_angle,
        gap_floor,
        resolved_table_size,
    )
    rejected_maximum_count = _rejected_maximum_count(
        threshold_angle,
        gap_floor,
        resolved_table_size,
    )
    success_counts: list[int] = []
    configured_means: dict[str, float] = {}
    for candidate_id, signed_gap in zip(
        resolved_candidate_ids,
        signed_latent,
        strict=True,
    ):
        target_angle = threshold_angle + signed_gap
        target_count = round(resolved_table_size * math.sin(target_angle) ** 2)
        if signed_gap > 0.0:
            count = max(selected_minimum_count, target_count)
        else:
            count = min(rejected_maximum_count, target_count)
        count = max(0, min(resolved_table_size, count))
        success_counts.append(count)
        configured_means[candidate_id] = count / resolved_table_size

    fixture = generate_exact_count_fixture(
        configured_means,
        table_size=resolved_table_size,
        seed=_derived_seed(resolved_case_seed, resolved_case_id, "fixture"),
    )
    frozen_means = fixture.evaluator.frozen_means
    empirical_success_counts = tuple(round(resolved_table_size * mean) for mean in frozen_means)
    if empirical_success_counts != tuple(success_counts):
        raise RuntimeError("exact-count fixture changed a designed success count")
    signed_empirical, selected_arms = _validate_post_rounding(
        frozen_means=frozen_means,
        threshold=resolved_threshold,
        threshold_angle=threshold_angle,
        public_gap_floor=gap_floor,
        k=selected_count,
    )
    empirical_selected = tuple(sorted(gap for gap in signed_empirical if gap > 0.0))
    empirical_rejected = tuple(sorted(-gap for gap in signed_empirical if gap < 0.0))
    all_empirical = empirical_selected + empirical_rejected
    structure = AngularGapStructureDescriptor(
        family=family,
        n_arms=arm_count,
        k=selected_count,
        selected_active_count=selected_active,
        rejected_active_count=rejected_active,
        active_count=selected_active + rejected_active,
        base_gap_scale=base_scale,
        heterogeneity=heterogeneity,
        latent_boundary_gap=min(abs(gap) for gap in signed_latent),
        empirical_boundary_gap=min(all_empirical),
        empirical_maximum_gap=max(all_empirical),
        empirical_mean_gap=statistics.fmean(all_empirical),
        empirical_gap_cv=_coefficient_of_variation(all_empirical),
        distinct_empirical_gap_count=len({round(gap, 15) for gap in all_empirical}),
        selected_empirical_gaps=empirical_selected,
        rejected_empirical_gaps=empirical_rejected,
    )
    canonical_selected_counts = sorted(empirical_success_counts[index] for index in selected_arms)
    selected_set = set(selected_arms)
    canonical_rejected_counts = sorted(
        count for index, count in enumerate(empirical_success_counts) if index not in selected_set
    )
    fingerprint_document: dict[str, object] = {
        "schema": "qgapselect.layer-c-difficulty-fingerprint.v1",
        "family": family,
        "n_arms": arm_count,
        "k": selected_count,
        "threshold": resolved_threshold,
        "public_gap_floor": gap_floor,
        "table_size": resolved_table_size,
        "selected_active_count": selected_active,
        "rejected_active_count": rejected_active,
        "selected_success_counts": canonical_selected_counts,
        "rejected_success_counts": canonical_rejected_counts,
    }
    return FrozenQuantumInstanceDesign(
        case_id=resolved_case_id,
        family=family,
        case_seed=resolved_case_seed,
        permutation_seed=resolved_permutation_seed,
        threshold=resolved_threshold,
        threshold_angle=threshold_angle,
        public_gap_floor=gap_floor,
        table_size=resolved_table_size,
        candidate_ids=resolved_candidate_ids,
        signed_latent_angular_gaps=signed_latent,
        signed_empirical_angular_gaps=signed_empirical,
        selected_arms=selected_arms,
        empirical_success_counts=empirical_success_counts,
        fixture=fixture,
        structure=structure,
        difficulty_fingerprint=_canonical_hash(fingerprint_document),
    )


__all__ = [
    "CLAIM_SCOPE",
    "FAMILIES",
    "AngularGapStructureDescriptor",
    "FrozenQuantumInstanceDesign",
    "generate_frozen_quantum_instance_design",
]
