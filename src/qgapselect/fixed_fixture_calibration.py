"""Fixed-fixture, multi-seed calibration for synthetic Layer-C experiments.

The fixture, rather than an individual algorithm seed, is the independent
experimental unit.  Anchor fixtures are selected from public structural
metadata only; algorithm outcomes are never consulted by the selector.  Each
selected fixture is then repeated under fresh method-specific measurement
seeds so instance variation and algorithm randomness can be reported
separately.

This module provides finite-sample calibration evidence only.  It does not
replace a worst-case fixed-confidence proof and does not establish quantum
advantage.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

CLAIM_SCOPE = (
    "fixed_fixture_multiseed_layer_c_calibration_"
    "no_worst_case_or_quantum_advantage_claim"
)
SELECTION_RULE = "within_family_mid_quantiles_of_active_count_over_boundary_gap"


class CalibrationRunLike(Protocol):
    family_id: str
    instance_id: str
    method_id: str
    repetition: int
    certified_exact_recovery: bool
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


def _probability(value: object, name: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    interval = "(0, 1)" if open_interval else "[0, 1]"
    if not math.isfinite(result) or not valid:
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if len(result) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a hexadecimal SHA-256 digest") from error
    return result.lower()


def _binomial_upper_tail(total: int, successes: int, probability: float) -> float:
    """Return ``P[Binom(total, probability) >= successes]`` stably."""

    if successes <= 0:
        return 1.0
    if successes > total:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        + index * log_p
        + (total - index) * log_q
        for index in range(successes, total + 1)
    ]
    maximum = max(terms)
    return min(1.0, math.exp(maximum) * math.fsum(math.exp(item - maximum) for item in terms))


def one_sided_clopper_pearson_lower(
    successes: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Return the exact one-sided Clopper--Pearson lower confidence bound."""

    success_count = _integer(successes, "successes")
    trial_count = _integer(total, "total", minimum=1)
    if success_count > trial_count:
        raise ValueError("successes cannot exceed total")
    tail_alpha = _probability(alpha, "alpha", open_interval=True)
    if success_count == 0:
        return 0.0
    lower = 0.0
    upper = success_count / trial_count
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_upper_tail(trial_count, success_count, midpoint) < tail_alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


@dataclass(frozen=True, slots=True)
class AnchorSelectionRecord:
    family_id: str
    instance_id: str
    difficulty_fingerprint: str
    hardness_score: float
    source_rank: int
    source_count: int
    target_quantile: float
    selection_rule: str = SELECTION_RULE
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _nonempty(self.family_id, "family_id")
        _nonempty(self.instance_id, "instance_id")
        _sha256(self.difficulty_fingerprint, "difficulty_fingerprint")
        if not math.isfinite(self.hardness_score) or self.hardness_score <= 0.0:
            raise ValueError("hardness_score must be finite and positive")
        rank = _integer(self.source_rank, "source_rank")
        count = _integer(self.source_count, "source_count", minimum=1)
        if rank >= count:
            raise ValueError("source_rank must be smaller than source_count")
        _probability(self.target_quantile, "target_quantile", open_interval=True)
        if self.selection_rule != SELECTION_RULE:
            raise ValueError("selection_rule is fixed by the preregistered protocol")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("invalid claim_scope")

    def as_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


def _anchor_candidate(document: Mapping[str, object]) -> tuple[str, str, str, float]:
    family_id = _nonempty(document.get("family_id"), "family_id")
    instance_id = _nonempty(document.get("instance_id"), "instance_id")
    fingerprint = _sha256(
        document.get("difficulty_fingerprint"), "difficulty_fingerprint"
    )
    metrics = document.get("structure_metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("structure_metrics must be a mapping")
    active_count = _integer(metrics.get("active_count"), "active_count", minimum=1)
    boundary_gap_raw = metrics.get("empirical_boundary_gap")
    if isinstance(boundary_gap_raw, bool) or not isinstance(
        boundary_gap_raw, (int, float)
    ):
        raise TypeError("empirical_boundary_gap must be numeric")
    boundary_gap = float(boundary_gap_raw)
    if not math.isfinite(boundary_gap) or boundary_gap <= 0.0:
        raise ValueError("empirical_boundary_gap must be finite and positive")
    return family_id, instance_id, fingerprint, active_count / boundary_gap


def select_hardness_quantile_anchors(
    instance_documents: Sequence[Mapping[str, object]],
    *,
    anchors_per_family: int,
    included_families: Sequence[str] | None = None,
) -> tuple[AnchorSelectionRecord, ...]:
    """Select anchors without consulting any algorithm result."""

    anchor_count = _integer(
        anchors_per_family, "anchors_per_family", minimum=1
    )
    if isinstance(instance_documents, (str, bytes)) or not isinstance(
        instance_documents, Sequence
    ):
        raise TypeError("instance_documents must be a sequence")
    if not instance_documents:
        raise ValueError("instance_documents cannot be empty")
    included = None
    if included_families is not None:
        included = tuple(_nonempty(item, "included_families item") for item in included_families)
        if not included or len(set(included)) != len(included):
            raise ValueError("included_families must be non-empty and unique")

    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    seen_ids: set[tuple[str, str]] = set()
    for document in instance_documents:
        if not isinstance(document, Mapping):
            raise TypeError("instance_documents must contain mappings")
        family_id, instance_id, fingerprint, score = _anchor_candidate(document)
        key = (family_id, instance_id)
        if key in seen_ids:
            raise ValueError("instance documents contain duplicate family/instance IDs")
        seen_ids.add(key)
        if included is None or family_id in included:
            grouped[family_id].append((instance_id, fingerprint, score))
    expected_families = set(grouped) if included is None else set(included)
    if set(grouped) != expected_families:
        raise ValueError("an included family has no instance documents")

    records: list[AnchorSelectionRecord] = []
    for family_id in sorted(grouped):
        candidates = sorted(grouped[family_id], key=lambda row: (row[2], row[0]))
        if len(candidates) < anchor_count:
            raise ValueError("anchors_per_family exceeds a family's candidate count")
        chosen_ranks: set[int] = set()
        for anchor_index in range(anchor_count):
            quantile = (anchor_index + 0.5) / anchor_count
            rank = min(len(candidates) - 1, math.floor(quantile * len(candidates)))
            if rank in chosen_ranks:
                raise RuntimeError("quantile rule selected a duplicate source rank")
            chosen_ranks.add(rank)
            instance_id, fingerprint, score = candidates[rank]
            records.append(
                AnchorSelectionRecord(
                    family_id=family_id,
                    instance_id=instance_id,
                    difficulty_fingerprint=fingerprint,
                    hardness_score=score,
                    source_rank=rank,
                    source_count=len(candidates),
                    target_quantile=quantile,
                )
            )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class FixedFixtureCalibrationRecord:
    family_id: str
    instance_id: str
    method_id: str
    successes: int
    repetitions: int
    success_rate: float
    one_sided_lower: float
    simultaneous_alpha: float
    target_success_probability: float
    target_certified: bool
    mean_coherent_queries: float
    median_coherent_queries: float
    within_fixture_seed_variance: float
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _nonempty(self.family_id, "family_id")
        _nonempty(self.instance_id, "instance_id")
        _nonempty(self.method_id, "method_id")
        successes = _integer(self.successes, "successes")
        repetitions = _integer(self.repetitions, "repetitions", minimum=1)
        if successes > repetitions:
            raise ValueError("successes cannot exceed repetitions")
        rate = _probability(self.success_rate, "success_rate")
        if not math.isclose(rate, successes / repetitions, abs_tol=1e-15):
            raise ValueError("success_rate is inconsistent with counts")
        lower = _probability(self.one_sided_lower, "one_sided_lower")
        alpha = _probability(self.simultaneous_alpha, "simultaneous_alpha", open_interval=True)
        target = _probability(
            self.target_success_probability,
            "target_success_probability",
            open_interval=True,
        )
        expected = one_sided_clopper_pearson_lower(
            successes, repetitions, alpha=alpha
        )
        if not math.isclose(lower, expected, abs_tol=1e-12):
            raise ValueError("one_sided_lower is inconsistent with exact calibration")
        if self.target_certified != (lower > target):
            raise ValueError("target_certified is inconsistent with the lower bound")
        for name in ("mean_coherent_queries", "median_coherent_queries"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        variance = float(self.within_fixture_seed_variance)
        if not math.isfinite(variance) or not 0.0 <= variance <= 0.25:
            raise ValueError("within_fixture_seed_variance must lie in [0, 0.25]")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("invalid claim_scope")

    def as_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CalibrationFamilyAggregate:
    family_id: str
    method_id: str
    anchor_count: int
    repetitions_per_anchor: int
    mean_anchor_success_rate: float
    minimum_anchor_success_rate: float
    target_certified_anchor_count: int
    all_anchors_target_certified: bool
    between_fixture_variance: float
    mean_within_fixture_seed_variance: float
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _nonempty(self.family_id, "family_id")
        _nonempty(self.method_id, "method_id")
        anchor_count = _integer(self.anchor_count, "anchor_count", minimum=1)
        _integer(
            self.repetitions_per_anchor,
            "repetitions_per_anchor",
            minimum=1,
        )
        certified_count = _integer(
            self.target_certified_anchor_count,
            "target_certified_anchor_count",
        )
        if certified_count > anchor_count:
            raise ValueError("target_certified_anchor_count exceeds anchor_count")
        for name in (
            "mean_anchor_success_rate",
            "minimum_anchor_success_rate",
        ):
            _probability(getattr(self, name), name)
        if self.minimum_anchor_success_rate > self.mean_anchor_success_rate:
            raise ValueError(
                "minimum_anchor_success_rate cannot exceed the mean"
            )
        if self.all_anchors_target_certified != (
            certified_count == anchor_count
        ):
            raise ValueError(
                "all_anchors_target_certified is inconsistent with counts"
            )
        between = float(self.between_fixture_variance)
        within = float(self.mean_within_fixture_seed_variance)
        if not math.isfinite(between) or not 0.0 <= between <= 0.25:
            raise ValueError("between_fixture_variance must lie in [0, 0.25]")
        if not math.isfinite(within) or not 0.0 <= within <= 0.25:
            raise ValueError(
                "mean_within_fixture_seed_variance must lie in [0, 0.25]"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("invalid claim_scope")

    def as_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class FixedFixtureCalibrationReport:
    records: tuple[FixedFixtureCalibrationRecord, ...]
    aggregates: tuple[CalibrationFamilyAggregate, ...]
    familywise_alpha: float
    simultaneous_alpha: float
    target_success_probability: float
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not self.records or not self.aggregates:
            raise ValueError("calibration report cannot be empty")
        family_alpha = _probability(
            self.familywise_alpha,
            "familywise_alpha",
            open_interval=True,
        )
        simultaneous = _probability(
            self.simultaneous_alpha,
            "simultaneous_alpha",
            open_interval=True,
        )
        _probability(
            self.target_success_probability,
            "target_success_probability",
            open_interval=True,
        )
        expected = family_alpha / len(self.records)
        if not math.isclose(simultaneous, expected, abs_tol=1e-15):
            raise ValueError(
                "simultaneous_alpha must Bonferroni-correct every record"
            )
        if any(record.simultaneous_alpha != simultaneous for record in self.records):
            raise ValueError("record simultaneous alpha differs from report")
        if any(
            record.target_success_probability != self.target_success_probability
            for record in self.records
        ):
            raise ValueError("record target differs from report")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("invalid claim_scope")

    def as_dict(self) -> dict[str, object]:
        return {
            "records": [record.as_dict() for record in self.records],
            "aggregates": [aggregate.as_dict() for aggregate in self.aggregates],
            "familywise_alpha": self.familywise_alpha,
            "simultaneous_alpha": self.simultaneous_alpha,
            "target_success_probability": self.target_success_probability,
            "claim_scope": self.claim_scope,
        }


def summarize_fixed_fixture_calibration(
    runs: Sequence[CalibrationRunLike],
    *,
    target_success_probability: float = 0.95,
    familywise_alpha: float = 0.05,
) -> FixedFixtureCalibrationReport:
    """Aggregate repeated seeds without treating them as independent fixtures."""

    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence) or not runs:
        raise ValueError("runs must be a non-empty sequence")
    target = _probability(
        target_success_probability,
        "target_success_probability",
        open_interval=True,
    )
    family_alpha = _probability(
        familywise_alpha, "familywise_alpha", open_interval=True
    )
    grouped: dict[tuple[str, str, str], list[CalibrationRunLike]] = defaultdict(list)
    panel_methods: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str, int]] = set()
    for run in runs:
        family_id = _nonempty(getattr(run, "family_id", None), "run.family_id")
        instance_id = _nonempty(getattr(run, "instance_id", None), "run.instance_id")
        method_id = _nonempty(getattr(run, "method_id", None), "run.method_id")
        repetition = _integer(getattr(run, "repetition", None), "run.repetition")
        success = getattr(run, "certified_exact_recovery", None)
        if not isinstance(success, bool):
            raise TypeError("run.certified_exact_recovery must be bool")
        queries = _integer(getattr(run, "coherent_queries", None), "run.coherent_queries")
        key = (family_id, instance_id, method_id, repetition)
        if key in seen:
            raise ValueError("duplicate method/repetition calibration run")
        seen.add(key)
        grouped[(family_id, instance_id, method_id)].append(run)
        panel_methods[(family_id, instance_id, repetition)].add(method_id)
        del queries
    method_sets = set(map(frozenset, panel_methods.values()))
    if len(method_sets) != 1:
        raise ValueError("every fixture/repetition panel must contain the same methods")
    repetition_counts = {len(group) for group in grouped.values()}
    if len(repetition_counts) != 1:
        raise ValueError("every fixture/method must have the same repetition count")

    simultaneous_alpha = family_alpha / len(grouped)
    records: list[FixedFixtureCalibrationRecord] = []
    for (family_id, instance_id, method_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda run: run.repetition)
        if [run.repetition for run in ordered] != list(range(len(ordered))):
            raise ValueError("repetition indices must be contiguous from zero")
        successes = sum(run.certified_exact_recovery for run in ordered)
        queries = [run.coherent_queries for run in ordered]
        rate = successes / len(ordered)
        lower = one_sided_clopper_pearson_lower(
            successes,
            len(ordered),
            alpha=simultaneous_alpha,
        )
        records.append(
            FixedFixtureCalibrationRecord(
                family_id=family_id,
                instance_id=instance_id,
                method_id=method_id,
                successes=successes,
                repetitions=len(ordered),
                success_rate=rate,
                one_sided_lower=lower,
                simultaneous_alpha=simultaneous_alpha,
                target_success_probability=target,
                target_certified=lower > target,
                mean_coherent_queries=statistics.fmean(queries),
                median_coherent_queries=float(statistics.median(queries)),
                within_fixture_seed_variance=rate * (1.0 - rate),
            )
        )

    by_family_method: dict[
        tuple[str, str], list[FixedFixtureCalibrationRecord]
    ] = defaultdict(list)
    for record in records:
        by_family_method[(record.family_id, record.method_id)].append(record)
    aggregates: list[CalibrationFamilyAggregate] = []
    for (family_id, method_id), group in sorted(by_family_method.items()):
        rates = [record.success_rate for record in group]
        repetitions = {record.repetitions for record in group}
        if len(repetitions) != 1:
            raise RuntimeError("calibration records changed repetition counts")
        certified_count = sum(record.target_certified for record in group)
        aggregates.append(
            CalibrationFamilyAggregate(
                family_id=family_id,
                method_id=method_id,
                anchor_count=len(group),
                repetitions_per_anchor=repetitions.pop(),
                mean_anchor_success_rate=statistics.fmean(rates),
                minimum_anchor_success_rate=min(rates),
                target_certified_anchor_count=certified_count,
                all_anchors_target_certified=certified_count == len(group),
                between_fixture_variance=statistics.pvariance(rates),
                mean_within_fixture_seed_variance=statistics.fmean(
                    record.within_fixture_seed_variance for record in group
                ),
            )
        )
    return FixedFixtureCalibrationReport(
        records=tuple(records),
        aggregates=tuple(aggregates),
        familywise_alpha=family_alpha,
        simultaneous_alpha=simultaneous_alpha,
        target_success_probability=target,
    )


def calibration_manifest_hash(
    anchors: Sequence[AnchorSelectionRecord],
    *,
    repetitions: int,
    master_seed: int,
) -> str:
    """Hash the immutable anchor/repetition design without outcome data."""

    document = {
        "schema": "qgapselect.fixed-fixture-calibration.v1",
        "master_seed": _integer(master_seed, "master_seed"),
        "repetitions": _integer(repetitions, "repetitions", minimum=1),
        "anchors": [record.as_dict() for record in anchors],
    }
    material = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


__all__ = [
    "CLAIM_SCOPE",
    "SELECTION_RULE",
    "AnchorSelectionRecord",
    "CalibrationFamilyAggregate",
    "CalibrationRunLike",
    "FixedFixtureCalibrationRecord",
    "FixedFixtureCalibrationReport",
    "calibration_manifest_hash",
    "one_sided_clopper_pearson_lower",
    "select_hardness_quantile_anchors",
    "summarize_fixed_fixture_calibration",
]
