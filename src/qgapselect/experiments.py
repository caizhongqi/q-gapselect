"""Reproducible analytic experiments for the Q-GapSelect research program.

The records produced here are evaluations of declared complexity expressions,
not empirical quantum speedups.  The module never substitutes a formula for a
``QueryLedger`` measurement, and every row includes a claim-status field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import exp, isfinite, log, pi
from random import Random

from .baselines import baseline_estimates, partition_baseline_estimates
from .complexity import candidate_layer_profile, topk_gap_profile


@dataclass(frozen=True, slots=True)
class BenchmarkInstance:
    """A deterministic Top-k benchmark with provenance metadata."""

    name: str
    means: tuple[float, ...]
    k: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        topk_gap_profile(self.means, self.k)

    @property
    def n(self) -> int:
        return len(self.means)


@dataclass(frozen=True, slots=True)
class PartitionBenchmark:
    """Independent Top-k groups whose outputs are all required."""

    name: str
    groups: tuple[BenchmarkInstance, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a partition benchmark requires at least one group")

    @property
    def n(self) -> int:
        return sum(group.n for group in self.groups)

    @property
    def k(self) -> int:
        return sum(group.k for group in self.groups)


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One auditable output row from an analytic scaling experiment."""

    scenario: str
    method: str
    n: int
    k: int
    output_size: int
    min_mean_gap: float
    max_mean_gap: float
    min_angular_gap: float
    max_angular_gap: float
    value: float
    unit: str
    claim_status: str
    data_source: str = "analytic_expression"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class ScalingSlope:
    """Descriptive log-log slope; no inferential claim is attached."""

    scenario: str
    method: str
    observations: int
    slope: float
    intercept: float
    claim_status: str = "descriptive_fit_to_analytic_proxy"


def _validate_generation_parameters(n: int, k: int, gap: float) -> None:
    if n < 2 or not 1 <= k < n:
        raise ValueError("expected n >= 2 and 1 <= k < n")
    if not isfinite(gap) or not 0.0 < gap <= 1.0:
        raise ValueError("gap must lie in (0, 1]")


def _permuted(values: Sequence[float], seed: int | None) -> tuple[float, ...]:
    result = list(values)
    if seed is not None:
        Random(seed).shuffle(result)
    return tuple(result)


def equal_gap_instance(
    n: int,
    k: int,
    gap: float,
    *,
    center: float = 0.5,
    seed: int | None = None,
    name: str | None = None,
) -> BenchmarkInstance:
    """Construct exactly k high arms and n-k low arms separated by ``gap``."""

    _validate_generation_parameters(n, k, gap)
    if not gap / 2.0 <= center <= 1.0 - gap / 2.0:
        raise ValueError("center and gap place a Bernoulli mean outside [0, 1]")
    high = center + gap / 2.0
    low = center - gap / 2.0
    means = _permuted((high,) * k + (low,) * (n - k), seed)
    return BenchmarkInstance(
        name=name or f"equal_gap_n{n}_k{k}",
        means=means,
        k=k,
        metadata={
            "family": "equal_gap",
            "requested_boundary_gap": gap,
            "seed": seed,
        },
    )


def _geometric_values(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (start,)
    log_start = log(start)
    log_stop = log(stop)
    return tuple(
        exp(log_start + (log_stop - log_start) * index / (count - 1))
        for index in range(count)
    )


def heterogeneous_gap_instance(
    n: int,
    k: int,
    min_boundary_gap: float,
    *,
    spread: float = 8.0,
    center: float = 0.5,
    seed: int | None = None,
    name: str | None = None,
) -> BenchmarkInstance:
    """Construct geometrically heterogeneous gaps around a strict boundary.

    The nearest selected and rejected arms are each ``min_boundary_gap/2``
    from ``center``, so their difference is exactly ``min_boundary_gap``.
    Other arms range geometrically up to ``spread`` times that offset.
    """

    _validate_generation_parameters(n, k, min_boundary_gap)
    if not isfinite(spread) or spread < 1.0:
        raise ValueError("spread must be finite and at least one")
    min_offset = min_boundary_gap / 2.0
    max_offset = min_offset * spread
    if center - max_offset < 0.0 or center + max_offset > 1.0:
        raise ValueError("spread places a Bernoulli mean outside [0, 1]")

    selected_offsets = _geometric_values(min_offset, max_offset, k)
    rejected_offsets = _geometric_values(min_offset, max_offset, n - k)
    means = _permuted(
        tuple(center + offset for offset in selected_offsets)
        + tuple(center - offset for offset in rejected_offsets),
        seed,
    )
    return BenchmarkInstance(
        name=name or f"heterogeneous_n{n}_k{k}",
        means=means,
        k=k,
        metadata={
            "family": "heterogeneous_gap",
            "requested_boundary_gap": min_boundary_gap,
            "spread": spread,
            "seed": seed,
        },
    )


def partition_direct_sum_instance(
    group_count: int,
    group_size: int,
    gap: float,
    *,
    heterogeneous: bool = False,
    spread: float = 8.0,
    seed: int = 0,
) -> PartitionBenchmark:
    """Build independent one-best groups for a partition direct-sum check."""

    if group_count < 1:
        raise ValueError("group_count must be positive")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    constructor = heterogeneous_gap_instance if heterogeneous else equal_gap_instance
    groups = []
    for group in range(group_count):
        kwargs: dict[str, object] = {
            "seed": seed + group,
            "name": f"partition_group_{group}",
        }
        if heterogeneous:
            kwargs["spread"] = spread
        groups.append(constructor(group_size, 1, gap, **kwargs))
    family = "partition_heterogeneous" if heterogeneous else "partition_equal_gap"
    return PartitionBenchmark(
        name=f"{family}_g{group_count}_s{group_size}",
        groups=tuple(groups),
        metadata={
            "family": family,
            "group_count": group_count,
            "group_size": group_size,
            "seed": seed,
        },
    )


def evaluate_instance(instance: BenchmarkInstance) -> tuple[ExperimentRecord, ...]:
    """Evaluate all registered proxies on a single Top-k instance."""

    profile = topk_gap_profile(instance.means, instance.k)
    layer = candidate_layer_profile(instance.means, instance.k)
    metadata = dict(instance.metadata)
    metadata.update(
        {
            "scaling_group": instance.name,
            "angular_scale_origin": pi / 2.0,
            "candidate_representation": layer.representation,
            "candidate_represented_output_size": len(layer.output_indices),
            "cardinality_smaller_side_size": profile.smaller_side_size,
        }
    )
    return tuple(
        ExperimentRecord(
            scenario=instance.name,
            method=estimate.method,
            n=instance.n,
            k=instance.k,
            output_size=profile.smaller_side_size,
            min_mean_gap=profile.minimum_mean_gap,
            max_mean_gap=max(profile.mean_gaps),
            min_angular_gap=profile.minimum_angular_gap,
            max_angular_gap=max(profile.angular_gaps),
            value=estimate.value,
            unit=estimate.unit,
            claim_status=estimate.claim_status,
            metadata=metadata,
        )
        for estimate in baseline_estimates(instance.means, instance.k)
    )


def evaluate_partition(
    instance: PartitionBenchmark,
) -> tuple[ExperimentRecord, ...]:
    """Evaluate explicitly additive proxies on independent required groups."""

    profiles = tuple(
        topk_gap_profile(group.means, group.k) for group in instance.groups
    )
    group_data = tuple((group.means, group.k) for group in instance.groups)
    metadata = dict(instance.metadata)
    metadata["aggregation"] = "sum_over_required_independent_groups"
    metadata["scaling_group"] = str(metadata["family"])
    metadata["angular_scale_origin"] = pi / 2.0
    return tuple(
        ExperimentRecord(
            scenario=instance.name,
            method=estimate.method,
            n=instance.n,
            k=instance.k,
            output_size=sum(profile.smaller_side_size for profile in profiles),
            min_mean_gap=min(profile.minimum_mean_gap for profile in profiles),
            max_mean_gap=max(max(profile.mean_gaps) for profile in profiles),
            min_angular_gap=min(profile.minimum_angular_gap for profile in profiles),
            max_angular_gap=max(max(profile.angular_gaps) for profile in profiles),
            value=estimate.value,
            unit=estimate.unit,
            claim_status=estimate.claim_status,
            metadata=metadata,
        )
        for estimate in partition_baseline_estimates(group_data)
    )


def canonical_scaling_suite(
    sizes: Iterable[int] = (8, 16, 32, 64, 128),
    *,
    gap: float = 0.125,
    spread: float = 4.0,
    seed: int = 1729,
) -> tuple[ExperimentRecord, ...]:
    """Run the preregistered analytic sanity suite.

    Covered cases are equal and heterogeneous gaps, k=1, k=n/2, and an
    increasing number of independent partition groups.  Values are computed
    from the formulas at call time; no result table is embedded in the code.
    """

    materialized_sizes = tuple(int(size) for size in sizes)
    if not materialized_sizes:
        raise ValueError("at least one size is required")

    records: list[ExperimentRecord] = []
    for index, n in enumerate(materialized_sizes):
        if n < 2 or n % 2:
            raise ValueError("canonical suite sizes must be even and at least two")
        instances = (
            equal_gap_instance(
                n,
                1,
                gap,
                seed=seed + 10 * index,
                name="equal_gap_k1",
            ),
            equal_gap_instance(
                n,
                n // 2,
                gap,
                seed=seed + 10 * index + 1,
                name="equal_gap_half",
            ),
            heterogeneous_gap_instance(
                n,
                1,
                gap,
                spread=spread,
                seed=seed + 10 * index + 2,
                name="heterogeneous_k1",
            ),
            heterogeneous_gap_instance(
                n,
                n // 2,
                gap,
                spread=spread,
                seed=seed + 10 * index + 3,
                name="heterogeneous_half",
            ),
        )
        for instance in instances:
            records.extend(evaluate_instance(instance))

        partition = partition_direct_sum_instance(
            group_count=n // 2,
            group_size=2,
            gap=gap,
            seed=seed + 10 * index + 4,
        )
        records.extend(evaluate_partition(partition))
    return tuple(records)


def fit_loglog_slopes(
    records: Iterable[ExperimentRecord],
) -> tuple[ScalingSlope, ...]:
    """Fit descriptive OLS slopes of log(value) against log(n)."""

    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        if record.n <= 0 or record.value <= 0.0:
            raise ValueError("log-log slopes require positive n and value")
        scaling_group = str(record.metadata.get("scaling_group", record.scenario))
        grouped.setdefault((scaling_group, record.method), []).append(record)

    slopes: list[ScalingSlope] = []
    for (scenario, method), group in sorted(grouped.items()):
        unique_n = {record.n for record in group}
        if len(unique_n) < 2:
            continue
        x = [log(record.n) for record in group]
        y = [log(record.value) for record in group]
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        denominator = sum((item - mean_x) ** 2 for item in x)
        if denominator == 0.0:
            continue
        slope = sum(
            (item_x - mean_x) * (item_y - mean_y)
            for item_x, item_y in zip(x, y, strict=True)
        ) / denominator
        slopes.append(
            ScalingSlope(
                scenario=scenario,
                method=method,
                observations=len(group),
                slope=slope,
                intercept=mean_y - slope * mean_x,
            )
        )
    return tuple(slopes)
