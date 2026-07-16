"""Fail-closed evidence gate for a possible quantum-advantage claim.

The experimental unit in this module is a preregistered fixture.  Fresh
algorithm seeds are repeated *within* each fixture and are never silently
treated as independent fixtures.  Fixed-cap success uses every attempted run
as its denominator: a timeout, an uncertified output, or a budget violation is
a failure.  Resource summaries likewise include every attempt and never
condition on both methods succeeding.

This module is a claim guard, not a theorem prover or circuit validator.  It
can prevent a claim when required evidence is absent.  A passing report means
only that the supplied, machine-readable evidence satisfies the preregistered
gate; it does not independently prove a quantum advantage or establish that an
external ``PROVED``/``VERIFIED`` assertion is true.
"""

from __future__ import annotations

import hashlib
import math
import operator
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

CLAIM_SCOPE = "claim_guard_only_does_not_independently_prove_quantum_advantage"
CONDITIONING = "all_preregistered_attempts_unconditional"
PREREGISTRATION_PASS_STATUS = "LOCKED_BEFORE_RUN"
THEORY_PASS_STATUS = "PROVED"
IMPLEMENTATION_RESOURCE_PASS_STATUS = "VERIFIED"
THEORY_REQUIREMENTS = (
    "new_upper_bound",
    "same_interface_composition_frontier",
    "matching_lower_bound",
)
IMPLEMENTATION_RESOURCE_REQUIREMENTS = (
    "coherent_index_execution",
    "resource_accounting",
    "strongest_baseline_fidelity",
)


class EvidenceRunLike(Protocol):
    """Record fields consumed by :func:`evaluate_ccfa_evidence_gate`.

    Attribute objects and mappings with these exact keys are both accepted.
    """

    family_id: str
    instance_id: str
    repetition: int
    query_cap: int
    method_id: str
    information_regime: str
    certified_exact: bool
    timeout: bool
    budget_valid: bool
    coherent_queries: int


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _open_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _closed_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _valid_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _mapping_field(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"run mapping is missing required field {name!r}")
        return record[name]
    try:
        return getattr(record, name)
    except AttributeError as error:
        raise TypeError(f"run object is missing required attribute {name!r}") from error


@dataclass(frozen=True, slots=True, order=True)
class PreregisteredFixture:
    """One fixture fixed before outcomes are observed."""

    family_id: str
    instance_id: str

    def __post_init__(self) -> None:
        _nonempty(self.family_id, "family_id")
        _nonempty(self.instance_id, "instance_id")

    def as_dict(self) -> dict[str, str]:
        return {"family_id": self.family_id, "instance_id": self.instance_id}


@dataclass(frozen=True, slots=True)
class CCFAEvidenceGateConfig:
    """Preregistered gate settings; thresholds are not learned from outcomes."""

    candidate_method_id: str
    strongest_baseline_method_ids: tuple[str, ...]
    information_regime: str
    preregistered_fixtures: tuple[PreregisteredFixture, ...]
    preregistered_query_caps: tuple[int, ...]
    repetitions_per_fixture: int
    preregistration_status: str
    preregistration_manifest_sha256: str | None
    minimum_risk_difference: float
    theory_statuses: Mapping[str, str] = field(default_factory=dict)
    implementation_resource_statuses: Mapping[str, str] = field(default_factory=dict)
    familywise_alpha: float = 0.05
    bootstrap_repetitions: int = 10_000
    bootstrap_seed: int = 20_260_715
    minimum_fixtures_per_family: int = 2

    def __post_init__(self) -> None:
        candidate = _nonempty(self.candidate_method_id, "candidate_method_id")
        object.__setattr__(self, "candidate_method_id", candidate)
        if not isinstance(self.strongest_baseline_method_ids, tuple):
            raise TypeError("strongest_baseline_method_ids must be a tuple")
        baselines = tuple(
            _nonempty(value, "strongest_baseline_method_ids item")
            for value in self.strongest_baseline_method_ids
        )
        if not baselines:
            raise ValueError("at least one strongest baseline must be preregistered")
        if len(set(baselines)) != len(baselines):
            raise ValueError("strongest_baseline_method_ids must be unique")
        if candidate in baselines:
            raise ValueError("candidate cannot also be a strongest baseline")
        object.__setattr__(self, "strongest_baseline_method_ids", tuple(sorted(baselines)))
        object.__setattr__(
            self,
            "information_regime",
            _nonempty(self.information_regime, "information_regime"),
        )
        if not isinstance(self.preregistered_fixtures, tuple):
            raise TypeError("preregistered_fixtures must be a tuple")
        if not self.preregistered_fixtures or any(
            not isinstance(item, PreregisteredFixture)
            for item in self.preregistered_fixtures
        ):
            raise TypeError(
                "preregistered_fixtures must be a non-empty tuple of "
                "PreregisteredFixture"
            )
        if len(set(self.preregistered_fixtures)) != len(self.preregistered_fixtures):
            raise ValueError("preregistered_fixtures must be unique")
        fixtures = tuple(sorted(self.preregistered_fixtures))
        object.__setattr__(self, "preregistered_fixtures", fixtures)
        minimum_fixtures = _integer(
            self.minimum_fixtures_per_family,
            "minimum_fixtures_per_family",
            minimum=2,
        )
        object.__setattr__(self, "minimum_fixtures_per_family", minimum_fixtures)
        family_counts: dict[str, int] = defaultdict(int)
        for fixture in fixtures:
            family_counts[fixture.family_id] += 1
        if any(count < minimum_fixtures for count in family_counts.values()):
            raise ValueError(
                "every family must contain at least minimum_fixtures_per_family fixtures"
            )
        if not isinstance(self.preregistered_query_caps, tuple):
            raise TypeError("preregistered_query_caps must be a tuple")
        caps = tuple(
            _integer(value, "preregistered_query_caps item", minimum=1)
            for value in self.preregistered_query_caps
        )
        if not caps:
            raise ValueError("at least one query cap must be preregistered")
        if len(set(caps)) != len(caps):
            raise ValueError("preregistered_query_caps must be unique")
        object.__setattr__(self, "preregistered_query_caps", tuple(sorted(caps)))
        repetitions = _integer(
            self.repetitions_per_fixture,
            "repetitions_per_fixture",
            minimum=2,
        )
        object.__setattr__(self, "repetitions_per_fixture", repetitions)
        object.__setattr__(
            self,
            "preregistration_status",
            _nonempty(self.preregistration_status, "preregistration_status"),
        )
        manifest = self.preregistration_manifest_sha256
        if manifest is not None and not isinstance(manifest, str):
            raise TypeError("preregistration_manifest_sha256 must be str or None")
        object.__setattr__(
            self,
            "minimum_risk_difference",
            _closed_unit(self.minimum_risk_difference, "minimum_risk_difference"),
        )
        object.__setattr__(
            self,
            "familywise_alpha",
            _open_probability(self.familywise_alpha, "familywise_alpha"),
        )
        object.__setattr__(
            self,
            "bootstrap_repetitions",
            _integer(self.bootstrap_repetitions, "bootstrap_repetitions", minimum=1),
        )
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer")
        for name in ("theory_statuses", "implementation_resource_statuses"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            normalized = {
                _nonempty(key, f"{name} key"): _nonempty(status, f"{name} status")
                for key, status in value.items()
            }
            object.__setattr__(self, name, normalized)

    @property
    def method_ids(self) -> tuple[str, ...]:
        return (self.candidate_method_id, *self.strongest_baseline_method_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_method_id": self.candidate_method_id,
            "strongest_baseline_method_ids": list(self.strongest_baseline_method_ids),
            "information_regime": self.information_regime,
            "preregistered_fixtures": [
                item.as_dict() for item in self.preregistered_fixtures
            ],
            "preregistered_query_caps": list(self.preregistered_query_caps),
            "repetitions_per_fixture": self.repetitions_per_fixture,
            "preregistration_status": self.preregistration_status,
            "preregistration_manifest_sha256": self.preregistration_manifest_sha256,
            "minimum_risk_difference": self.minimum_risk_difference,
            "theory_statuses": dict(sorted(self.theory_statuses.items())),
            "implementation_resource_statuses": dict(
                sorted(self.implementation_resource_statuses.items())
            ),
            "familywise_alpha": self.familywise_alpha,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "minimum_fixtures_per_family": self.minimum_fixtures_per_family,
        }


@dataclass(frozen=True, slots=True)
class _Run:
    family_id: str
    instance_id: str
    repetition: int
    query_cap: int
    method_id: str
    information_regime: str
    certified_exact: bool
    timeout: bool
    budget_valid: bool
    coherent_queries: int

    @property
    def budget_violation(self) -> bool:
        return not self.budget_valid or self.coherent_queries > self.query_cap

    @property
    def success(self) -> bool:
        return self.certified_exact and not self.timeout and not self.budget_violation


def _normalize_run(record: EvidenceRunLike | Mapping[str, object]) -> _Run:
    run = _Run(
        family_id=_nonempty(_mapping_field(record, "family_id"), "family_id"),
        instance_id=_nonempty(_mapping_field(record, "instance_id"), "instance_id"),
        repetition=_integer(_mapping_field(record, "repetition"), "repetition"),
        query_cap=_integer(_mapping_field(record, "query_cap"), "query_cap", minimum=1),
        method_id=_nonempty(_mapping_field(record, "method_id"), "method_id"),
        information_regime=_nonempty(
            _mapping_field(record, "information_regime"), "information_regime"
        ),
        certified_exact=_boolean(
            _mapping_field(record, "certified_exact"), "certified_exact"
        ),
        timeout=_boolean(_mapping_field(record, "timeout"), "timeout"),
        budget_valid=_boolean(
            _mapping_field(record, "budget_valid"), "budget_valid"
        ),
        coherent_queries=_integer(
            _mapping_field(record, "coherent_queries"), "coherent_queries"
        ),
    )
    return run


@dataclass(frozen=True, slots=True)
class FixtureMethodSummary:
    family_id: str
    instance_id: str
    query_cap: int
    method_id: str
    attempt_count: int
    certified_exact_count: int
    valid_success_count: int
    timeout_count: int
    budget_violation_count: int
    success_rate: float
    mean_coherent_queries: float
    max_coherent_queries: int
    conditioning: str = CONDITIONING

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FamilyCapMethodSummary:
    family_id: str
    query_cap: int
    method_id: str
    fixture_count: int
    attempt_count: int
    valid_success_count: int
    timeout_count: int
    budget_violation_count: int
    success_rate: float
    mean_fixture_success_rate: float
    between_fixture_success_rate_variance: float
    mean_coherent_queries: float
    conditioning: str = CONDITIONING

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FixedCapMethodSummary:
    query_cap: int
    method_id: str
    fixture_count: int
    attempt_count: int
    valid_success_count: int
    timeout_count: int
    budget_violation_count: int
    success_rate: float
    mean_coherent_queries: float
    conditioning: str = CONDITIONING

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ClusterBootstrapInterval:
    lower: float
    upper: float
    confidence_level: float
    repetitions: int
    seed: int
    resampling_unit: str = "whole_fixture_with_all_seed_repetitions"
    method: str = "paired_fixture_cluster_percentile_bootstrap"

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class PairwiseSuperiorityComparison:
    hypothesis_id: str
    family_id: str
    query_cap: int
    candidate_method_id: str
    baseline_method_id: str
    fixture_count: int
    attempt_pairs: int
    both_success: int
    candidate_only_success: int
    baseline_only_success: int
    neither_success: int
    candidate_success_rate: float
    baseline_success_rate: float
    risk_difference: float
    exact_two_sided_mcnemar_p_value: float
    holm_adjusted_p_value: float
    holm_rejected_at_familywise_alpha: bool
    bootstrap: ClusterBootstrapInterval
    minimum_risk_difference: float
    statistical_superiority_passed: bool
    conditioning: str = CONDITIONING

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["bootstrap"] = self.bootstrap.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class ParetoFrontierPoint:
    method_id: str
    query_cap: int
    success_rate: float
    mean_coherent_queries: float
    attempt_count: int
    dominated: bool
    dominated_by: tuple[str, ...]
    axes: str = "query_cap_minimized_and_unconditional_success_rate_maximized"

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["dominated_by"] = list(self.dominated_by)
        return result


@dataclass(frozen=True, slots=True)
class EvidenceGateCheck:
    check_id: str
    passed: bool
    detail: str
    blocker: str | None

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CCFAEvidenceGateReport:
    config: CCFAEvidenceGateConfig
    fixture_summaries: tuple[FixtureMethodSummary, ...]
    family_cap_summaries: tuple[FamilyCapMethodSummary, ...]
    fixed_cap_summaries: tuple[FixedCapMethodSummary, ...]
    pairwise_comparisons: tuple[PairwiseSuperiorityComparison, ...]
    pareto_frontier: tuple[ParetoFrontierPoint, ...]
    checks: tuple[EvidenceGateCheck, ...]
    blockers: tuple[str, ...]
    advantage_claimable: bool
    claim_scope: str = CLAIM_SCOPE
    interpretation: str = (
        "A claim-prevention gate only; passing does not independently prove a "
        "quantum advantage or validate supplied theorem/implementation statuses."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "fixture_summaries": [item.as_dict() for item in self.fixture_summaries],
            "family_cap_summaries": [
                item.as_dict() for item in self.family_cap_summaries
            ],
            "fixed_cap_summaries": [item.as_dict() for item in self.fixed_cap_summaries],
            "pairwise_comparisons": [
                item.as_dict() for item in self.pairwise_comparisons
            ],
            "pareto_frontier": [item.as_dict() for item in self.pareto_frontier],
            "checks": [item.as_dict() for item in self.checks],
            "blockers": list(self.blockers),
            "advantage_claimable": self.advantage_claimable,
            "claim_scope": self.claim_scope,
            "interpretation": self.interpretation,
        }


def _validate_complete_panel(
    runs: Sequence[_Run], config: CCFAEvidenceGateConfig
) -> dict[tuple[str, str, int, int, str], _Run]:
    if not runs:
        raise ValueError("run panel cannot be empty")
    fixtures = {(item.family_id, item.instance_id) for item in config.preregistered_fixtures}
    caps = set(config.preregistered_query_caps)
    methods = set(config.method_ids)
    expected_repetitions = set(range(config.repetitions_per_fixture))
    indexed: dict[tuple[str, str, int, int, str], _Run] = {}
    for run in runs:
        if (run.family_id, run.instance_id) not in fixtures:
            raise ValueError("run lies outside the preregistered fixture panel")
        if run.query_cap not in caps:
            raise ValueError("run lies outside the preregistered query-cap panel")
        if run.method_id not in methods:
            raise ValueError("run uses a method outside the preregistered method panel")
        if run.repetition not in expected_repetitions:
            raise ValueError("run repetition lies outside the preregistered seed panel")
        key = (
            run.family_id,
            run.instance_id,
            run.query_cap,
            run.repetition,
            run.method_id,
        )
        if key in indexed:
            raise ValueError(f"duplicate run record for panel key {key!r}")
        indexed[key] = run
    expected = {
        (fixture.family_id, fixture.instance_id, cap, repetition, method)
        for fixture in config.preregistered_fixtures
        for cap in config.preregistered_query_caps
        for repetition in range(config.repetitions_per_fixture)
        for method in config.method_ids
    }
    missing = sorted(expected.difference(indexed))
    if missing:
        raise ValueError(
            f"incomplete preregistered panel: {len(missing)} missing records; "
            f"first missing key={missing[0]!r}"
        )
    if len(indexed) != len(expected):
        raise RuntimeError("validated panel size is inconsistent with preregistration")
    return indexed


def _fixture_summary(rows: Sequence[_Run]) -> FixtureMethodSummary:
    first = rows[0]
    successes = sum(run.success for run in rows)
    return FixtureMethodSummary(
        family_id=first.family_id,
        instance_id=first.instance_id,
        query_cap=first.query_cap,
        method_id=first.method_id,
        attempt_count=len(rows),
        certified_exact_count=sum(run.certified_exact for run in rows),
        valid_success_count=successes,
        timeout_count=sum(run.timeout for run in rows),
        budget_violation_count=sum(run.budget_violation for run in rows),
        success_rate=successes / len(rows),
        mean_coherent_queries=statistics.fmean(run.coherent_queries for run in rows),
        max_coherent_queries=max(run.coherent_queries for run in rows),
    )


def _summaries(
    runs: Sequence[_Run],
) -> tuple[
    tuple[FixtureMethodSummary, ...],
    tuple[FamilyCapMethodSummary, ...],
    tuple[FixedCapMethodSummary, ...],
]:
    by_fixture: dict[tuple[str, str, int, str], list[_Run]] = defaultdict(list)
    by_family_cap: dict[tuple[str, int, str], list[_Run]] = defaultdict(list)
    by_cap: dict[tuple[int, str], list[_Run]] = defaultdict(list)
    for run in runs:
        by_fixture[(run.family_id, run.instance_id, run.query_cap, run.method_id)].append(run)
        by_family_cap[(run.family_id, run.query_cap, run.method_id)].append(run)
        by_cap[(run.query_cap, run.method_id)].append(run)
    fixture_summaries = tuple(
        _fixture_summary(sorted(rows, key=lambda item: item.repetition))
        for _, rows in sorted(by_fixture.items())
    )
    fixture_lookup = {
        (row.family_id, row.instance_id, row.query_cap, row.method_id): row
        for row in fixture_summaries
    }
    family_summaries: list[FamilyCapMethodSummary] = []
    for (family, cap, method), rows in sorted(by_family_cap.items()):
        fixtures = sorted({run.instance_id for run in rows})
        rates = [
            fixture_lookup[(family, instance, cap, method)].success_rate
            for instance in fixtures
        ]
        successes = sum(run.success for run in rows)
        family_summaries.append(
            FamilyCapMethodSummary(
                family_id=family,
                query_cap=cap,
                method_id=method,
                fixture_count=len(fixtures),
                attempt_count=len(rows),
                valid_success_count=successes,
                timeout_count=sum(run.timeout for run in rows),
                budget_violation_count=sum(run.budget_violation for run in rows),
                success_rate=successes / len(rows),
                mean_fixture_success_rate=statistics.fmean(rates),
                between_fixture_success_rate_variance=statistics.pvariance(rates),
                mean_coherent_queries=statistics.fmean(
                    run.coherent_queries for run in rows
                ),
            )
        )
    cap_summaries: list[FixedCapMethodSummary] = []
    for (cap, method), rows in sorted(by_cap.items()):
        successes = sum(run.success for run in rows)
        cap_summaries.append(
            FixedCapMethodSummary(
                query_cap=cap,
                method_id=method,
                fixture_count=len({(run.family_id, run.instance_id) for run in rows}),
                attempt_count=len(rows),
                valid_success_count=successes,
                timeout_count=sum(run.timeout for run in rows),
                budget_violation_count=sum(run.budget_violation for run in rows),
                success_rate=successes / len(rows),
                mean_coherent_queries=statistics.fmean(
                    run.coherent_queries for run in rows
                ),
            )
        )
    return fixture_summaries, tuple(family_summaries), tuple(cap_summaries)


def _exact_symmetric_binomial_lower_tail(trials: int, observed: int) -> float:
    if observed < 0:
        return 0.0
    if observed >= trials:
        return 1.0
    log_last = (
        math.lgamma(trials + 1)
        - math.lgamma(observed + 1)
        - math.lgamma(trials - observed + 1)
        - trials * math.log(2.0)
    )
    relative_sum = 1.0
    relative_term = 1.0
    for index in range(observed, 0, -1):
        relative_term *= index / (trials - index + 1)
        relative_sum += relative_term
    log_probability = log_last + math.log(relative_sum)
    if log_probability < math.log(math.nextafter(0.0, 1.0)):
        return 0.0
    return min(1.0, math.exp(log_probability))


def _exact_two_sided_mcnemar(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    return min(
        1.0,
        2.0
        * _exact_symmetric_binomial_lower_tail(
            discordant, min(candidate_only, baseline_only)
        ),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _comparison_seed(master_seed: int, hypothesis_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{hypothesis_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _cluster_bootstrap(
    fixture_differences: Sequence[Sequence[int]],
    *,
    confidence_level: float,
    repetitions: int,
    seed: int,
) -> ClusterBootstrapInterval:
    if len(fixture_differences) < 2:
        raise ValueError("fixture-cluster bootstrap requires at least two fixtures")
    widths = {len(values) for values in fixture_differences}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("fixture clusters must be non-empty and equally calibrated")
    rng = random.Random(seed)
    fixture_count = len(fixture_differences)
    repetition_count = next(iter(widths))
    # Whole-fixture resampling depends only on each cluster total.  Precompute
    # it so 10,000-repetition paper runs do not repeatedly sum the same seed
    # panel; this is algebraically identical to the uncompressed bootstrap.
    cluster_totals = tuple(sum(values) for values in fixture_differences)
    draws: list[float] = []
    for _ in range(repetitions):
        total = 0
        for _ in range(fixture_count):
            total += cluster_totals[rng.randrange(fixture_count)]
        draws.append(total / (fixture_count * repetition_count))
    tail = (1.0 - confidence_level) / 2.0
    return ClusterBootstrapInterval(
        lower=_percentile(draws, tail),
        upper=_percentile(draws, 1.0 - tail),
        confidence_level=confidence_level,
        repetitions=repetitions,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class _ComparisonDraft:
    hypothesis_id: str
    family_id: str
    query_cap: int
    baseline_method_id: str
    fixture_count: int
    attempt_pairs: int
    both_success: int
    candidate_only: int
    baseline_only: int
    neither: int
    candidate_rate: float
    baseline_rate: float
    risk_difference: float
    p_value: float
    bootstrap: ClusterBootstrapInterval


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (hypothesis_id, p_value) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (family_size - rank + 1) * p_value))
        adjusted[hypothesis_id] = running
    return adjusted


def _paired_comparisons(
    indexed: Mapping[tuple[str, str, int, int, str], _Run],
    config: CCFAEvidenceGateConfig,
) -> tuple[PairwiseSuperiorityComparison, ...]:
    families: dict[str, list[str]] = defaultdict(list)
    for fixture in config.preregistered_fixtures:
        families[fixture.family_id].append(fixture.instance_id)
    comparison_count = (
        len(families)
        * len(config.preregistered_query_caps)
        * len(config.strongest_baseline_method_ids)
    )
    confidence_level = 1.0 - config.familywise_alpha / comparison_count
    drafts: list[_ComparisonDraft] = []
    for family in sorted(families):
        for cap in config.preregistered_query_caps:
            for baseline in config.strongest_baseline_method_ids:
                hypothesis_id = (
                    f"family={family}|cap={cap}|candidate={config.candidate_method_id}"
                    f"|baseline={baseline}"
                )
                both = candidate_only = baseline_only = neither = 0
                fixture_differences: list[list[int]] = []
                for instance in sorted(families[family]):
                    differences: list[int] = []
                    for repetition in range(config.repetitions_per_fixture):
                        candidate = indexed[
                            (family, instance, cap, repetition, config.candidate_method_id)
                        ]
                        reference = indexed[(family, instance, cap, repetition, baseline)]
                        candidate_success = candidate.success
                        baseline_success = reference.success
                        differences.append(int(candidate_success) - int(baseline_success))
                        if candidate_success and baseline_success:
                            both += 1
                        elif candidate_success:
                            candidate_only += 1
                        elif baseline_success:
                            baseline_only += 1
                        else:
                            neither += 1
                    fixture_differences.append(differences)
                attempt_pairs = both + candidate_only + baseline_only + neither
                candidate_rate = (both + candidate_only) / attempt_pairs
                baseline_rate = (both + baseline_only) / attempt_pairs
                seed = _comparison_seed(config.bootstrap_seed, hypothesis_id)
                drafts.append(
                    _ComparisonDraft(
                        hypothesis_id=hypothesis_id,
                        family_id=family,
                        query_cap=cap,
                        baseline_method_id=baseline,
                        fixture_count=len(fixture_differences),
                        attempt_pairs=attempt_pairs,
                        both_success=both,
                        candidate_only=candidate_only,
                        baseline_only=baseline_only,
                        neither=neither,
                        candidate_rate=candidate_rate,
                        baseline_rate=baseline_rate,
                        risk_difference=candidate_rate - baseline_rate,
                        p_value=_exact_two_sided_mcnemar(
                            candidate_only, baseline_only
                        ),
                        bootstrap=_cluster_bootstrap(
                            fixture_differences,
                            confidence_level=confidence_level,
                            repetitions=config.bootstrap_repetitions,
                            seed=seed,
                        ),
                    )
                )
    adjusted = _holm_adjust({item.hypothesis_id: item.p_value for item in drafts})
    results: list[PairwiseSuperiorityComparison] = []
    for item in drafts:
        holm_p = adjusted[item.hypothesis_id]
        passed = (
            item.risk_difference >= config.minimum_risk_difference
            and item.bootstrap.lower >= config.minimum_risk_difference
            and holm_p <= config.familywise_alpha
        )
        results.append(
            PairwiseSuperiorityComparison(
                hypothesis_id=item.hypothesis_id,
                family_id=item.family_id,
                query_cap=item.query_cap,
                candidate_method_id=config.candidate_method_id,
                baseline_method_id=item.baseline_method_id,
                fixture_count=item.fixture_count,
                attempt_pairs=item.attempt_pairs,
                both_success=item.both_success,
                candidate_only_success=item.candidate_only,
                baseline_only_success=item.baseline_only,
                neither_success=item.neither,
                candidate_success_rate=item.candidate_rate,
                baseline_success_rate=item.baseline_rate,
                risk_difference=item.risk_difference,
                exact_two_sided_mcnemar_p_value=item.p_value,
                holm_adjusted_p_value=holm_p,
                holm_rejected_at_familywise_alpha=holm_p
                <= config.familywise_alpha,
                bootstrap=item.bootstrap,
                minimum_risk_difference=config.minimum_risk_difference,
                statistical_superiority_passed=passed,
            )
        )
    return tuple(results)


def _pareto_frontier(
    summaries: Sequence[FixedCapMethodSummary],
) -> tuple[ParetoFrontierPoint, ...]:
    points: list[ParetoFrontierPoint] = []
    for current in summaries:
        dominators = sorted(
            f"{other.method_id}@{other.query_cap}"
            for other in summaries
            if other is not current
            and other.query_cap <= current.query_cap
            and other.success_rate >= current.success_rate
            and (
                other.query_cap < current.query_cap
                or other.success_rate > current.success_rate
            )
        )
        points.append(
            ParetoFrontierPoint(
                method_id=current.method_id,
                query_cap=current.query_cap,
                success_rate=current.success_rate,
                mean_coherent_queries=current.mean_coherent_queries,
                attempt_count=current.attempt_count,
                dominated=bool(dominators),
                dominated_by=tuple(dominators),
            )
        )
    return tuple(points)


def _check(check_id: str, passed: bool, detail: str, blocker: str) -> EvidenceGateCheck:
    return EvidenceGateCheck(
        check_id=check_id,
        passed=passed,
        detail=detail,
        blocker=None if passed else blocker,
    )


def evaluate_ccfa_evidence_gate(
    records: Sequence[EvidenceRunLike | Mapping[str, object]],
    config: CCFAEvidenceGateConfig,
) -> CCFAEvidenceGateReport:
    """Evaluate a complete preregistered panel without outcome conditioning.

    Malformed records, duplicate keys, small fixture panels, and missing or
    extra panel cells are rejected rather than converted into a partial report.
    Ordinary evidence failures (for example a timeout, information mismatch,
    non-significant result, or missing theorem status) return a fail-closed
    report with machine-readable blockers.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    if not isinstance(config, CCFAEvidenceGateConfig):
        raise TypeError("config must be CCFAEvidenceGateConfig")
    runs = tuple(_normalize_run(record) for record in records)
    indexed = _validate_complete_panel(runs, config)
    fixture_summaries, family_summaries, cap_summaries = _summaries(runs)
    comparisons = _paired_comparisons(indexed, config)
    pareto = _pareto_frontier(cap_summaries)

    preregistered = (
        config.preregistration_status == PREREGISTRATION_PASS_STATUS
        and _valid_sha256(config.preregistration_manifest_sha256)
    )
    budget_violations = sum(run.budget_violation for run in runs)
    information_mismatches = sum(
        run.information_regime != config.information_regime for run in runs
    )
    statistical_passed = all(
        item.statistical_superiority_passed for item in comparisons
    )
    baseline_ids = set(config.strongest_baseline_method_ids)
    baseline_dominators = [
        (point.method_id, point.query_cap, point.dominated_by)
        for point in pareto
        if point.method_id == config.candidate_method_id
        and any(value.split("@", 1)[0] in baseline_ids for value in point.dominated_by)
    ]
    checks: list[EvidenceGateCheck] = [
        _check(
            "preregistered_fixture_and_query_cap",
            preregistered,
            "fixture/cap manifest must be locked before runs and identified by SHA-256",
            "preregistration_not_locked_or_manifest_missing",
        ),
        _check(
            "complete_fixed_fixture_multiseed_panel",
            True,
            f"exact complete panel with {len(indexed)} records",
            "incomplete_experimental_panel",
        ),
        _check(
            "zero_query_budget_violations",
            budget_violations == 0,
            f"budget violations={budget_violations} across all attempts",
            "query_budget_violation",
        ),
        _check(
            "exact_information_match",
            information_mismatches == 0,
            f"information-regime mismatches={information_mismatches}",
            "information_regime_not_exactly_matched",
        ),
        _check(
            "per_fixture_seed_calibration",
            True,
            f"every fixture/method/cap has repetitions 0..{config.repetitions_per_fixture - 1}",
            "fixture_seed_calibration_incomplete",
        ),
        _check(
            "paired_statistical_superiority",
            statistical_passed,
            (
                f"{sum(item.statistical_superiority_passed for item in comparisons)}/"
                f"{len(comparisons)} family/cap/baseline comparisons pass preregistered RD, "
                "fixture-cluster CI, exact McNemar, and Holm gates"
            ),
            "paired_statistical_superiority_not_established",
        ),
        _check(
            "cross_cap_candidate_frontier",
            not baseline_dominators,
            f"candidate cap points dominated by a baseline point={len(baseline_dominators)}",
            "candidate_is_baseline_dominated_across_query_caps",
        ),
    ]
    for requirement in THEORY_REQUIREMENTS:
        status = config.theory_statuses.get(requirement, "MISSING")
        checks.append(
            _check(
                f"theory_{requirement}",
                status == THEORY_PASS_STATUS,
                f"supplied status={status!r}; required={THEORY_PASS_STATUS!r}",
                f"theory_{requirement}_not_proved",
            )
        )
    for requirement in IMPLEMENTATION_RESOURCE_REQUIREMENTS:
        status = config.implementation_resource_statuses.get(requirement, "MISSING")
        checks.append(
            _check(
                f"implementation_resource_{requirement}",
                status == IMPLEMENTATION_RESOURCE_PASS_STATUS,
                (
                    f"supplied status={status!r}; "
                    f"required={IMPLEMENTATION_RESOURCE_PASS_STATUS!r}"
                ),
                f"{requirement}_not_verified",
            )
        )
    blockers = tuple(check.blocker for check in checks if check.blocker is not None)
    return CCFAEvidenceGateReport(
        config=config,
        fixture_summaries=fixture_summaries,
        family_cap_summaries=family_summaries,
        fixed_cap_summaries=cap_summaries,
        pairwise_comparisons=comparisons,
        pareto_frontier=pareto,
        checks=tuple(checks),
        blockers=blockers,
        advantage_claimable=not blockers,
    )


__all__ = [
    "CLAIM_SCOPE",
    "CONDITIONING",
    "IMPLEMENTATION_RESOURCE_PASS_STATUS",
    "IMPLEMENTATION_RESOURCE_REQUIREMENTS",
    "PREREGISTRATION_PASS_STATUS",
    "THEORY_PASS_STATUS",
    "THEORY_REQUIREMENTS",
    "CCFAEvidenceGateConfig",
    "CCFAEvidenceGateReport",
    "ClusterBootstrapInterval",
    "EvidenceGateCheck",
    "EvidenceRunLike",
    "FamilyCapMethodSummary",
    "FixedCapMethodSummary",
    "FixtureMethodSummary",
    "PairwiseSuperiorityComparison",
    "ParetoFrontierPoint",
    "PreregisteredFixture",
    "evaluate_ccfa_evidence_gate",
]
