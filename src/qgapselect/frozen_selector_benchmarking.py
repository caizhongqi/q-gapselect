"""Reproducible fixed-tensor benchmarks for classical candidate selectors.

This runner is intentionally limited to synthetic, frozen source-oracle
experiments.  Every method receives a fresh blind oracle over the exact same
reward/cost tensor for a given landscape, trial, and budget.  Ground truth is
held by the runner and consulted only after ``selector.select`` returns.
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
from types import MappingProxyType

from .attack_oracles import (
    BlindSourceRewardOracle,
    CandidateEdge,
    FrozenCandidateGraph,
    FrozenSourceFixture,
    SourceOracleBudget,
    SourceQueryStatus,
    generate_bernoulli_landscape,
)
from .attack_selectors import (
    CostAwareRacingSelector,
    GCGSGreedyGraphSelector,
    InternalCLUCBStyleSelector,
    RandomSelector,
    SelectionResult,
    SuccessiveHalvingSelector,
    UniformTopKSelector,
)
from .attack_statistics import wilson_score_interval

CLAIM_SCOPE = (
    "frozen_source_oracle_classical_selector_benchmark_no_llm_execution_no_quantum_advantage_claim"
)
DEFAULT_SELECTOR_IDS = (
    "random",
    "uniform",
    "successive_halving",
    "cost_aware_racing",
    "clucb_style",
    "gcgs",
)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


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


def _finite_number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
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


def _derived_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(("qgapselect-selector-benchmark-v1", str(master_seed), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _freeze_float_mapping(
    values: Mapping[str, object],
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError(f"{name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = _nonempty_string(raw_key, f"{name} key")
        value = _finite_number(raw_value, f"{name}[{key!r}]", minimum=minimum)
        if maximum is not None and value > maximum:
            raise ValueError(f"{name}[{key!r}] cannot exceed {maximum}")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class SelectorLandscape:
    """One preregistered Bernoulli reward/cost landscape."""

    landscape_id: str
    graph: FrozenCandidateGraph
    means: Mapping[str, float]
    candidate_costs: Mapping[str, float]
    cost_jitter: float = 0.0

    def __post_init__(self) -> None:
        landscape_id = _nonempty_string(self.landscape_id, "landscape_id")
        if not isinstance(self.graph, FrozenCandidateGraph):
            raise TypeError("graph must be a FrozenCandidateGraph")
        means = _freeze_float_mapping(
            self.means,
            name="means",
            minimum=0.0,
            maximum=1.0,
        )
        costs = _freeze_float_mapping(
            self.candidate_costs,
            name="candidate_costs",
            minimum=0.0,
        )
        if any(value <= 0.0 for value in costs.values()):
            raise ValueError("candidate costs must be positive")
        candidate_ids = set(self.graph.candidate_ids)
        if set(means) != candidate_ids or set(costs) != candidate_ids:
            raise ValueError("means and candidate_costs must exactly match graph candidates")
        jitter = _finite_number(self.cost_jitter, "cost_jitter")
        if jitter >= 1.0:
            raise ValueError("cost_jitter must lie in [0, 1)")
        object.__setattr__(self, "landscape_id", landscape_id)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "candidate_costs", costs)
        object.__setattr__(self, "cost_jitter", jitter)

    @classmethod
    def from_means(
        cls,
        landscape_id: str,
        means: Mapping[str, float],
        *,
        candidate_costs: Mapping[str, float] | float = 1.0,
        cost_jitter: float = 0.0,
        edges: Sequence[CandidateEdge] = (),
    ) -> SelectorLandscape:
        candidate_ids = tuple(means)
        graph = FrozenCandidateGraph.from_ids(candidate_ids, edges=tuple(edges))
        if isinstance(candidate_costs, Mapping):
            costs = dict(candidate_costs)
        else:
            value = _finite_number(candidate_costs, "candidate_costs", minimum=0.0)
            costs = {candidate_id: value for candidate_id in candidate_ids}
        return cls(
            landscape_id=landscape_id,
            graph=graph,
            means=means,
            candidate_costs=costs,
            cost_jitter=cost_jitter,
        )

    def manifest_document(self) -> dict[str, object]:
        return {
            "landscape_id": self.landscape_id,
            "graph": self.graph.as_dict(),
            "means": dict(self.means),
            "candidate_costs": dict(self.candidate_costs),
            "cost_jitter": self.cost_jitter,
        }


@dataclass(frozen=True, slots=True)
class SelectorBudget:
    """Named fair budget shared by every selector in one panel."""

    budget_id: str
    max_queries: int
    max_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "budget_id", _nonempty_string(self.budget_id, "budget_id"))
        object.__setattr__(
            self,
            "max_queries",
            _integer(self.max_queries, "max_queries", minimum=0),
        )
        object.__setattr__(
            self,
            "max_cost",
            _finite_number(self.max_cost, "max_cost", minimum=0.0),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "budget_id": self.budget_id,
            "max_queries": self.max_queries,
            "max_cost": self.max_cost,
        }


@dataclass(frozen=True, slots=True)
class SelectorBenchmarkRun:
    """Post-selection metrics and audited resources for one independent run."""

    landscape_id: str
    budget_id: str
    trial: int
    fixture_seed: int
    selector_id: str
    selector_seed: int
    k: int
    fixture_manifest_hash: str
    selected_candidates: tuple[str, ...]
    reference_top_k: tuple[str, ...]
    ranking: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    exact_match: bool
    selected_expected_reward: float
    optimal_expected_reward: float
    top_k_regret: float
    queries_used: int
    cost_used: float
    query_budget: int
    cost_budget: float
    sample_cost_upper_bound: float
    query_utilization: float
    cost_utilization: float
    stop_reason: str
    sample_counts: Mapping[str, int]
    claim_scope: str = CLAIM_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            "landscape_id": self.landscape_id,
            "budget_id": self.budget_id,
            "trial": self.trial,
            "fixture_seed": self.fixture_seed,
            "selector_id": self.selector_id,
            "selector_seed": self.selector_seed,
            "k": self.k,
            "fixture_manifest_hash": self.fixture_manifest_hash,
            "selected_candidates": list(self.selected_candidates),
            "reference_top_k": list(self.reference_top_k),
            "ranking": list(self.ranking),
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "exact_match": self.exact_match,
            "selected_expected_reward": self.selected_expected_reward,
            "optimal_expected_reward": self.optimal_expected_reward,
            "top_k_regret": self.top_k_regret,
            "queries_used": self.queries_used,
            "cost_used": self.cost_used,
            "query_budget": self.query_budget,
            "cost_budget": self.cost_budget,
            "sample_cost_upper_bound": self.sample_cost_upper_bound,
            "query_utilization": self.query_utilization,
            "cost_utilization": self.cost_utilization,
            "stop_reason": self.stop_reason,
            "sample_counts": dict(self.sample_counts),
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class SelectorBenchmarkAggregate:
    """Trial aggregate for one landscape, budget, and selector."""

    landscape_id: str
    budget_id: str
    selector_id: str
    run_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    exact_match_successes: int
    exact_match_rate: float
    exact_match_wilson_lower: float
    exact_match_wilson_upper: float
    mean_top_k_regret: float
    std_top_k_regret: float
    mean_selected_expected_reward: float
    mean_queries_used: float
    mean_cost_used: float
    mean_query_utilization: float
    mean_cost_utilization: float
    claim_scope: str = CLAIM_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()
        }


@dataclass(frozen=True, slots=True)
class FrozenSelectorBenchmarkReport:
    """Complete deterministic benchmark output."""

    manifest_hash: str
    master_seed: int
    trials: int
    k: int
    samples_per_candidate: int
    selector_ids: tuple[str, ...]
    fixture_manifest_hashes: Mapping[str, str]
    runs: tuple[SelectorBenchmarkRun, ...]
    aggregates: tuple[SelectorBenchmarkAggregate, ...]
    claim_scope: str = CLAIM_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "master_seed": self.master_seed,
            "trials": self.trials,
            "k": self.k,
            "samples_per_candidate": self.samples_per_candidate,
            "selector_ids": list(self.selector_ids),
            "fixture_manifest_hashes": dict(self.fixture_manifest_hashes),
            "runs": [run.as_dict() for run in self.runs],
            "aggregates": [aggregate.as_dict() for aggregate in self.aggregates],
            "claim_scope": self.claim_scope,
        }


class _BlindSamplerAdapter:
    """Expose exactly the sample protocol required by selector baselines."""

    __slots__ = ("__oracle", "__selector_id")

    def __init__(self, oracle: BlindSourceRewardOracle, selector_id: str) -> None:
        self.__oracle = oracle
        self.__selector_id = selector_id

    def sample(self, candidate_id: str) -> tuple[float, float]:
        observation = self.__oracle.query(
            candidate_id,
            tag=f"selector:{self.__selector_id}",
        )
        return float(observation.reward), observation.cost


def _build_selector(selector_id: str, seed: int) -> object:
    factories = {
        "random": RandomSelector,
        "uniform": UniformTopKSelector,
        "successive_halving": SuccessiveHalvingSelector,
        "cost_aware_racing": CostAwareRacingSelector,
        "clucb_style": InternalCLUCBStyleSelector,
        "gcgs": GCGSGreedyGraphSelector,
    }
    try:
        factory = factories[selector_id]
    except KeyError:
        raise ValueError(f"unknown selector_id {selector_id!r}") from None
    return factory(seed=seed)


def _adjacency(graph: FrozenCandidateGraph) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(
        {candidate_id: graph.successors(candidate_id) for candidate_id in graph.candidate_ids}
    )


def _run_selector(
    *,
    landscape: SelectorLandscape,
    fixture: FrozenSourceFixture,
    fixture_seed: int,
    budget: SelectorBudget,
    trial: int,
    selector_id: str,
    selector_seed: int,
    k: int,
) -> SelectorBenchmarkRun:
    # ``fixture`` is deliberately runner-owned.  Only the fresh blind oracle
    # and public graph identity are passed into selector code.
    oracle_budget = SourceOracleBudget(budget.max_queries, budget.max_cost)
    oracle = fixture.open_oracle(oracle_budget)
    adapter = _BlindSamplerAdapter(oracle, selector_id)
    selector = _build_selector(selector_id, selector_seed)
    # This bound is derived only from the preregistered public cost model.  It
    # must not be tightened after inspecting the realized frozen streams.
    sample_cost_upper_bound = max(landscape.candidate_costs.values()) * (
        1.0 + landscape.cost_jitter
    )
    result: SelectionResult = selector.select(
        candidates=fixture.tensor.graph.candidate_ids,
        sampler=adapter,
        k=k,
        query_budget=budget.max_queries,
        cost_budget=budget.max_cost,
        sample_cost_upper_bound=sample_cost_upper_bound,
        adjacency=_adjacency(fixture.tensor.graph),
    )

    # The trusted evaluator is first touched after selection has terminated.
    snapshot = oracle.snapshot()
    evaluation = fixture.evaluator.evaluate(result.selected_candidates, k=k)
    if snapshot.manifest_hash != fixture.manifest_hash:
        raise RuntimeError("oracle and fixture manifest hashes differ")
    if snapshot.queries_used != result.resources.queries_used:
        raise RuntimeError("selector and oracle query ledgers differ")
    if not math.isclose(
        snapshot.cost_used,
        result.resources.cost_used,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("selector and oracle cost ledgers differ")
    if not result.resources.within_budget:
        raise RuntimeError("selector returned an over-budget resource ledger")
    if any(record.status is not SourceQueryStatus.ACCEPTED for record in snapshot.records):
        raise RuntimeError("a rejected oracle attempt invalidates the benchmark run")

    query_utilization = snapshot.queries_used / budget.max_queries if budget.max_queries else 0.0
    cost_utilization = snapshot.cost_used / budget.max_cost if budget.max_cost else 0.0
    return SelectorBenchmarkRun(
        landscape_id=landscape.landscape_id,
        budget_id=budget.budget_id,
        trial=trial,
        fixture_seed=fixture_seed,
        selector_id=selector_id,
        selector_seed=selector_seed,
        k=k,
        fixture_manifest_hash=fixture.manifest_hash,
        selected_candidates=tuple(result.selected_candidates),
        reference_top_k=evaluation.reference_top_k,
        ranking=tuple(result.ranking),
        precision_at_k=evaluation.precision_at_k,
        recall_at_k=evaluation.recall_at_k,
        exact_match=evaluation.exact_match,
        selected_expected_reward=evaluation.selected_expected_reward,
        optimal_expected_reward=evaluation.optimal_expected_reward,
        top_k_regret=evaluation.top_k_regret,
        queries_used=snapshot.queries_used,
        cost_used=snapshot.cost_used,
        query_budget=budget.max_queries,
        cost_budget=budget.max_cost,
        sample_cost_upper_bound=sample_cost_upper_bound,
        query_utilization=query_utilization,
        cost_utilization=cost_utilization,
        stop_reason=result.stop_reason,
        sample_counts=MappingProxyType(
            {str(key): int(value) for key, value in result.sample_counts.items()}
        ),
    )


def _aggregate_runs(
    runs: tuple[SelectorBenchmarkRun, ...],
) -> tuple[SelectorBenchmarkAggregate, ...]:
    grouped: dict[tuple[str, str, str], list[SelectorBenchmarkRun]] = defaultdict(list)
    for run in runs:
        grouped[(run.landscape_id, run.budget_id, run.selector_id)].append(run)

    aggregates: list[SelectorBenchmarkAggregate] = []
    for (landscape_id, budget_id, selector_id), group in grouped.items():
        regrets = [run.top_k_regret for run in group]
        exact_match_successes = sum(run.exact_match for run in group)
        exact_match_interval = wilson_score_interval(
            exact_match_successes,
            len(group),
        )
        aggregates.append(
            SelectorBenchmarkAggregate(
                landscape_id=landscape_id,
                budget_id=budget_id,
                selector_id=selector_id,
                run_count=len(group),
                mean_precision_at_k=statistics.fmean(run.precision_at_k for run in group),
                mean_recall_at_k=statistics.fmean(run.recall_at_k for run in group),
                exact_match_successes=exact_match_successes,
                exact_match_rate=exact_match_interval.estimate,
                exact_match_wilson_lower=exact_match_interval.lower,
                exact_match_wilson_upper=exact_match_interval.upper,
                mean_top_k_regret=statistics.fmean(regrets),
                std_top_k_regret=statistics.pstdev(regrets),
                mean_selected_expected_reward=statistics.fmean(
                    run.selected_expected_reward for run in group
                ),
                mean_queries_used=statistics.fmean(run.queries_used for run in group),
                mean_cost_used=statistics.fmean(run.cost_used for run in group),
                mean_query_utilization=statistics.fmean(run.query_utilization for run in group),
                mean_cost_utilization=statistics.fmean(run.cost_utilization for run in group),
            )
        )
    return tuple(aggregates)


def run_frozen_selector_benchmark(
    *,
    landscapes: Sequence[SelectorLandscape],
    budgets: Sequence[SelectorBudget],
    trials: int,
    k: int,
    samples_per_candidate: int,
    master_seed: int = 0,
    selector_ids: Sequence[str] = DEFAULT_SELECTOR_IDS,
) -> FrozenSelectorBenchmarkReport:
    """Execute a complete fixed-tensor classical selector benchmark."""

    landscape_tuple = tuple(landscapes)
    budget_tuple = tuple(budgets)
    selectors = tuple(selector_ids)
    if not landscape_tuple:
        raise ValueError("landscapes cannot be empty")
    if any(not isinstance(item, SelectorLandscape) for item in landscape_tuple):
        raise TypeError("landscapes must contain SelectorLandscape objects")
    if len({item.landscape_id for item in landscape_tuple}) != len(landscape_tuple):
        raise ValueError("landscape IDs must be unique")
    if not budget_tuple:
        raise ValueError("budgets cannot be empty")
    if any(not isinstance(item, SelectorBudget) for item in budget_tuple):
        raise TypeError("budgets must contain SelectorBudget objects")
    if len({item.budget_id for item in budget_tuple}) != len(budget_tuple):
        raise ValueError("budget IDs must be unique")
    if not selectors:
        raise ValueError("selector_ids cannot be empty")
    if len(set(selectors)) != len(selectors):
        raise ValueError("selector_ids must be unique")
    unknown_selectors = set(selectors) - set(DEFAULT_SELECTOR_IDS)
    if unknown_selectors:
        raise ValueError(f"unknown selector IDs: {sorted(unknown_selectors)}")
    trial_count = _integer(trials, "trials", minimum=1)
    top_k = _integer(k, "k", minimum=1)
    sample_count = _integer(
        samples_per_candidate,
        "samples_per_candidate",
        minimum=1,
    )
    seed = _integer(master_seed, "master_seed", minimum=0)
    if any(top_k > len(landscape.graph.candidate_ids) for landscape in landscape_tuple):
        raise ValueError("k cannot exceed any landscape candidate count")
    maximum_query_budget = max(budget.max_queries for budget in budget_tuple)
    if sample_count < maximum_query_budget:
        raise ValueError(
            "samples_per_candidate must cover the largest query budget to prevent "
            "selector-dependent stream exhaustion"
        )

    runs: list[SelectorBenchmarkRun] = []
    fixture_manifests: dict[str, str] = {}
    fixture_manifest_rows: list[dict[str, object]] = []
    for landscape in landscape_tuple:
        for trial in range(trial_count):
            fixture_seed = _derived_seed(seed, "fixture", landscape.landscape_id, trial)
            fixture = generate_bernoulli_landscape(
                landscape.means,
                samples_per_candidate=sample_count,
                seed=fixture_seed,
                candidate_costs=landscape.candidate_costs,
                cost_jitter=landscape.cost_jitter,
                graph=landscape.graph,
            )
            fixture_key = f"{landscape.landscape_id}/trial-{trial}"
            fixture_manifests[fixture_key] = fixture.manifest_hash
            fixture_manifest_rows.append(
                {
                    "landscape_id": landscape.landscape_id,
                    "trial": trial,
                    "fixture_seed": fixture_seed,
                    "fixture_manifest_hash": fixture.manifest_hash,
                }
            )
            for budget in budget_tuple:
                for selector_id in selectors:
                    # Excluding budget from this seed couples randomized choices
                    # across budget levels while preserving independent oracles.
                    selector_seed = _derived_seed(
                        seed,
                        "selector",
                        landscape.landscape_id,
                        trial,
                        selector_id,
                    )
                    runs.append(
                        _run_selector(
                            landscape=landscape,
                            fixture=fixture,
                            fixture_seed=fixture_seed,
                            budget=budget,
                            trial=trial,
                            selector_id=selector_id,
                            selector_seed=selector_seed,
                            k=top_k,
                        )
                    )

    manifest_document: dict[str, object] = {
        "schema": "qgapselect.frozen-selector-benchmark.v1",
        "claim_scope": CLAIM_SCOPE,
        "master_seed": seed,
        "trials": trial_count,
        "k": top_k,
        "samples_per_candidate": sample_count,
        "selector_ids": list(selectors),
        "budgets": [budget.as_dict() for budget in budget_tuple],
        "landscapes": [landscape.manifest_document() for landscape in landscape_tuple],
        "fixtures": fixture_manifest_rows,
    }
    run_tuple = tuple(runs)
    return FrozenSelectorBenchmarkReport(
        manifest_hash=_canonical_hash(manifest_document),
        master_seed=seed,
        trials=trial_count,
        k=top_k,
        samples_per_candidate=sample_count,
        selector_ids=selectors,
        fixture_manifest_hashes=MappingProxyType(fixture_manifests),
        runs=run_tuple,
        aggregates=_aggregate_runs(run_tuple),
    )


__all__ = [
    "CLAIM_SCOPE",
    "DEFAULT_SELECTOR_IDS",
    "FrozenSelectorBenchmarkReport",
    "SelectorBenchmarkAggregate",
    "SelectorBenchmarkRun",
    "SelectorBudget",
    "SelectorLandscape",
    "run_frozen_selector_benchmark",
]
