"""Budget-accounted selector baselines for a frozen attack-candidate pool.

The selectors in this module deliberately see only ``sample(candidate_id)``.
They do not accept a vector of latent means, and consequently cannot use
ground-truth rewards while deciding where to spend the source-model budget.

Cost accounting needs one small but important contract.  A sampler reports its
cost only *after* it has run, so a strict cost budget is impossible unless the
caller supplies an upper bound on one sample.  ``sample_cost_upper_bound`` is
that reservation bound.  A selector starts a query only when the full bound is
still available and rejects a sampler that returns a larger cost.  Thus every
successful ``SelectionResult`` is within both of its declared budgets.
"""

from __future__ import annotations

import math
import operator
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Protocol, runtime_checkable

CLAIM_STATUS = "executed_classical_selector_baseline_no_quantum_advantage_claim"


@runtime_checkable
class CandidateSampler(Protocol):
    """The only reward interface available to a selector."""

    def sample(self, candidate_id: Hashable) -> tuple[float, float]:
        """Return one bounded reward and its realized non-negative cost."""


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        comparator = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_reward(value: object) -> float:
    # Boolean paired-attack events are valid Bernoulli rewards.
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("sample reward must be a real number") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("sample reward must lie in [0, 1]")
    return result


def _seed(value: object) -> int:
    return _integer(value, "seed")


def _candidate_tuple(values: Sequence[Hashable]) -> tuple[Hashable, ...]:
    try:
        candidates = tuple(values)
    except TypeError as error:
        raise TypeError("candidates must be a finite sequence") from error
    if not candidates:
        raise ValueError("candidates cannot be empty")
    try:
        unique_count = len(set(candidates))
    except TypeError as error:
        raise TypeError("candidate identifiers must be hashable") from error
    if unique_count != len(candidates):
        raise ValueError("candidate identifiers must be unique")
    return candidates


def _immutable_mapping(values: Mapping[Hashable, object]) -> Mapping[Hashable, object]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class SelectionTraceRecord:
    """One charged call to ``CandidateSampler.sample``."""

    ordinal: int
    candidate_id: Hashable
    reward: float
    cost: float
    cumulative_queries: int
    cumulative_cost: float


@dataclass(frozen=True, slots=True)
class SelectionResourceLedger:
    """Immutable resource certificate attached to every selection result."""

    query_budget: int
    cost_budget: float
    sample_cost_upper_bound: float
    queries_used: int
    cost_used: float
    per_candidate_queries: Mapping[Hashable, int]
    per_candidate_cost: Mapping[Hashable, float]

    @property
    def remaining_queries(self) -> int:
        return self.query_budget - self.queries_used

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.cost_budget - self.cost_used)

    @property
    def can_afford_another_sample(self) -> bool:
        return (
            self.remaining_queries > 0
            and self.remaining_cost + 1e-12 >= self.sample_cost_upper_bound
        )

    @property
    def within_budget(self) -> bool:
        return (
            0 <= self.queries_used <= self.query_budget
            and -1e-12 <= self.cost_used <= self.cost_budget + 1e-12
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "query_budget": self.query_budget,
            "cost_budget": self.cost_budget,
            "sample_cost_upper_bound": self.sample_cost_upper_bound,
            "queries_used": self.queries_used,
            "cost_used": self.cost_used,
            "remaining_queries": self.remaining_queries,
            "remaining_cost": self.remaining_cost,
            "per_candidate_queries": dict(self.per_candidate_queries),
            "per_candidate_cost": dict(self.per_candidate_cost),
            "within_budget": self.within_budget,
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Common immutable result returned by every selector baseline."""

    selector_id: str
    selected_candidates: tuple[Hashable, ...]
    ranking: tuple[Hashable, ...]
    estimates: Mapping[Hashable, float | None]
    sample_counts: Mapping[Hashable, int]
    resources: SelectionResourceLedger
    trace: tuple[SelectionTraceRecord, ...]
    stop_reason: str
    rounds_completed: int
    eliminated_candidates: tuple[Hashable, ...] = ()
    seed: int = 0
    claim_status: str = CLAIM_STATUS

    @property
    def selected(self) -> tuple[Hashable, ...]:
        return self.selected_candidates

    @property
    def outputs(self) -> tuple[Hashable, ...]:
        return self.selected_candidates

    @property
    def complete(self) -> bool:
        return bool(self.selected_candidates) and self.resources.within_budget


@dataclass(slots=True)
class _CandidateStatistics:
    rewards: dict[Hashable, float]
    counts: dict[Hashable, int]
    costs: dict[Hashable, float]

    @classmethod
    def empty(cls, candidates: tuple[Hashable, ...]) -> _CandidateStatistics:
        return cls(
            rewards={candidate: 0.0 for candidate in candidates},
            counts={candidate: 0 for candidate in candidates},
            costs={candidate: 0.0 for candidate in candidates},
        )

    def mean(self, candidate: Hashable) -> float:
        count = self.counts[candidate]
        return self.rewards[candidate] / count if count else 0.0

    def mean_cost(self, candidate: Hashable) -> float:
        count = self.counts[candidate]
        return self.costs[candidate] / count if count else math.inf


class _BudgetController:
    def __init__(
        self,
        candidates: tuple[Hashable, ...],
        sampler: CandidateSampler,
        *,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float,
    ) -> None:
        if not isinstance(sampler, CandidateSampler):
            raise TypeError("sampler must expose sample(candidate_id)")
        self.candidates = candidates
        self.sampler = sampler
        self.query_budget = _integer(query_budget, "query_budget")
        self.cost_budget = _finite_nonnegative(cost_budget, "cost_budget")
        self.sample_cost_upper_bound = _finite_positive(
            sample_cost_upper_bound, "sample_cost_upper_bound"
        )
        self.statistics = _CandidateStatistics.empty(candidates)
        self.queries_used = 0
        self.cost_used = 0.0
        self.trace: list[SelectionTraceRecord] = []

    def can_sample(self) -> bool:
        return (
            self.queries_used < self.query_budget
            and self.cost_used + self.sample_cost_upper_bound <= self.cost_budget + 1e-12
        )

    def can_complete_pass(self, count: int) -> bool:
        return (
            count >= 0
            and self.queries_used + count <= self.query_budget
            and self.cost_used + count * self.sample_cost_upper_bound <= self.cost_budget + 1e-12
        )

    def draw(self, candidate: Hashable) -> bool:
        if candidate not in self.statistics.counts:
            raise ValueError("selector attempted to query a candidate outside the pool")
        if not self.can_sample():
            return False
        raw = self.sampler.sample(candidate)
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise TypeError("sample must return a (reward, cost) tuple")
        reward = _bounded_reward(raw[0])
        cost = _finite_nonnegative(raw[1], "sample cost")
        if cost > self.sample_cost_upper_bound + 1e-12:
            raise ValueError("sample cost exceeds the declared sample_cost_upper_bound")
        # The reservation check above makes this a sampler-contract assertion,
        # rather than a post-hoc attempt to hide an over-budget query.
        if self.cost_used + cost > self.cost_budget + 1e-12:
            raise RuntimeError("internal budget reservation failed")
        self.queries_used += 1
        self.cost_used += cost
        if self.cost_used > self.cost_budget:
            self.cost_used = self.cost_budget
        stats = self.statistics
        stats.rewards[candidate] += reward
        stats.counts[candidate] += 1
        stats.costs[candidate] += cost
        self.trace.append(
            SelectionTraceRecord(
                ordinal=self.queries_used,
                candidate_id=candidate,
                reward=reward,
                cost=cost,
                cumulative_queries=self.queries_used,
                cumulative_cost=self.cost_used,
            )
        )
        return True

    def snapshot(self) -> SelectionResourceLedger:
        return SelectionResourceLedger(
            query_budget=self.query_budget,
            cost_budget=self.cost_budget,
            sample_cost_upper_bound=self.sample_cost_upper_bound,
            queries_used=self.queries_used,
            cost_used=self.cost_used,
            per_candidate_queries=MappingProxyType(dict(self.statistics.counts)),
            per_candidate_cost=MappingProxyType(dict(self.statistics.costs)),
        )


def _rank(
    candidates: tuple[Hashable, ...],
    statistics: _CandidateStatistics,
) -> tuple[Hashable, ...]:
    ordinal = {candidate: index for index, candidate in enumerate(candidates)}
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                statistics.counts[candidate] == 0,
                -statistics.mean(candidate),
                ordinal[candidate],
            ),
        )
    )


def _racing_allocation_priority(
    candidate: Hashable,
    *,
    statistics: _CandidateStatistics,
    radii: Mapping[Hashable, float],
    ordinal: Mapping[Hashable, int],
) -> tuple[float, int]:
    if statistics.counts[candidate] == 0:
        return (math.inf, -ordinal[candidate])
    return (
        radii[candidate] / max(statistics.mean_cost(candidate), 1e-12),
        -ordinal[candidate],
    )


def _stop_reason(controller: _BudgetController, *, converged: bool = False) -> str:
    if converged:
        return "selection_converged"
    if controller.queries_used >= controller.query_budget:
        return "query_budget_exhausted"
    if not controller.can_sample():
        return "cost_budget_exhausted"
    return "selector_stopped"


def _result(
    *,
    selector_id: str,
    candidates: tuple[Hashable, ...],
    k: int,
    controller: _BudgetController,
    seed: int,
    rounds_completed: int,
    eliminated: Sequence[Hashable] = (),
    ranking: tuple[Hashable, ...] | None = None,
    converged: bool = False,
) -> SelectionResult:
    resolved_ranking = ranking if ranking is not None else _rank(candidates, controller.statistics)
    statistics = controller.statistics
    estimates = {
        candidate: (statistics.mean(candidate) if statistics.counts[candidate] else None)
        for candidate in candidates
    }
    return SelectionResult(
        selector_id=selector_id,
        selected_candidates=resolved_ranking[:k],
        ranking=resolved_ranking,
        estimates=_immutable_mapping(estimates),
        sample_counts=_immutable_mapping(statistics.counts),
        resources=controller.snapshot(),
        trace=tuple(controller.trace),
        stop_reason=_stop_reason(controller, converged=converged),
        rounds_completed=rounds_completed,
        eliminated_candidates=tuple(eliminated),
        seed=seed,
    )


def _prepare(
    *,
    candidates: Sequence[Hashable],
    sampler: CandidateSampler,
    k: int,
    query_budget: int,
    cost_budget: float,
    sample_cost_upper_bound: float,
) -> tuple[tuple[Hashable, ...], int, _BudgetController]:
    pool = _candidate_tuple(candidates)
    top_k = _integer(k, "k", minimum=1)
    if top_k > len(pool):
        raise ValueError("k cannot exceed the number of candidates")
    return (
        pool,
        top_k,
        _BudgetController(
            pool,
            sampler,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        ),
    )


class RandomSelector:
    """Seeded random ranking that spends no reward-oracle budget."""

    selector_id = "random_selector_v1"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _seed(seed)

    def select(
        self,
        *,
        candidates: Sequence[Hashable],
        sampler: CandidateSampler,
        k: int,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float = 1.0,
        adjacency: Mapping[Hashable, Sequence[Hashable]] | None = None,
    ) -> SelectionResult:
        del adjacency
        pool, top_k, controller = _prepare(
            candidates=candidates,
            sampler=sampler,
            k=k,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        )
        ranking = list(pool)
        random.Random(self.seed).shuffle(ranking)
        return _result(
            selector_id=self.selector_id,
            candidates=pool,
            k=top_k,
            controller=controller,
            seed=self.seed,
            rounds_completed=0,
            ranking=tuple(ranking),
            converged=True,
        )


class UniformTopKSelector:
    """Round-robin empirical Top-k with equal completed passes."""

    selector_id = "uniform_empirical_topk_v1"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _seed(seed)

    def select(
        self,
        *,
        candidates: Sequence[Hashable],
        sampler: CandidateSampler,
        k: int,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float = 1.0,
        adjacency: Mapping[Hashable, Sequence[Hashable]] | None = None,
    ) -> SelectionResult:
        del adjacency
        pool, top_k, controller = _prepare(
            candidates=candidates,
            sampler=sampler,
            k=k,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        )
        rounds = 0
        while controller.can_complete_pass(len(pool)):
            for candidate in pool:
                controller.draw(candidate)
            rounds += 1
        # Use a deterministic prefix when the remaining budget cannot fund a
        # complete pass; sample counts can differ by at most one.
        for candidate in pool:
            if not controller.draw(candidate):
                break
        return _result(
            selector_id=self.selector_id,
            candidates=pool,
            k=top_k,
            controller=controller,
            seed=self.seed,
            rounds_completed=rounds,
        )


class SuccessiveHalvingSelector:
    """Fixed-budget Top-k successive halving over empirical rewards."""

    selector_id = "successive_halving_topk_v1"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _seed(seed)

    def select(
        self,
        *,
        candidates: Sequence[Hashable],
        sampler: CandidateSampler,
        k: int,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float = 1.0,
        adjacency: Mapping[Hashable, Sequence[Hashable]] | None = None,
    ) -> SelectionResult:
        del adjacency
        pool, top_k, controller = _prepare(
            candidates=candidates,
            sampler=sampler,
            k=k,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        )
        ordinal = {candidate: index for index, candidate in enumerate(pool)}
        active = list(pool)
        eliminated: list[Hashable] = []
        rounds = 0
        stages = max(1, math.ceil(math.log2(len(pool) / top_k)))

        while len(active) > top_k and controller.can_sample():
            stages_left = max(1, stages - rounds)
            affordable_queries = min(
                controller.query_budget - controller.queries_used,
                int((controller.cost_budget - controller.cost_used) // sample_cost_upper_bound),
            )
            passes = max(1, affordable_queries // (stages_left * len(active)))
            completed = 0
            for _ in range(passes):
                if not controller.can_complete_pass(len(active)):
                    break
                for candidate in active:
                    controller.draw(candidate)
                completed += 1
            if completed == 0:
                break
            rounds += 1
            keep = max(top_k, math.ceil(len(active) / 2))
            ordered = sorted(
                active,
                key=lambda candidate: (-controller.statistics.mean(candidate), ordinal[candidate]),
            )
            eliminated.extend(ordered[keep:])
            active = ordered[:keep]

        # Spend any affordable tail on survivors without allowing eliminated
        # candidates to re-enter merely because they received more samples.
        survivor_index = 0
        while active and controller.can_sample():
            controller.draw(active[survivor_index % len(active)])
            survivor_index += 1
        active_rank = sorted(
            active,
            key=lambda candidate: (-controller.statistics.mean(candidate), ordinal[candidate]),
        )
        eliminated_rank = sorted(
            eliminated,
            key=lambda candidate: (-controller.statistics.mean(candidate), ordinal[candidate]),
        )
        remaining = [
            candidate
            for candidate in pool
            if candidate not in set(active_rank) | set(eliminated_rank)
        ]
        ranking = tuple(active_rank + eliminated_rank + remaining)
        return _result(
            selector_id=self.selector_id,
            candidates=pool,
            k=top_k,
            controller=controller,
            seed=self.seed,
            rounds_completed=rounds,
            eliminated=eliminated,
            ranking=ranking,
            converged=len(active) == top_k,
        )


class CostAwareRacingSelector:
    """Hoeffding racing with uncertainty-per-observed-cost allocation."""

    selector_id = "cost_aware_racing_topk_v1"

    def __init__(self, *, confidence: float = 0.05, seed: int = 0) -> None:
        self.seed = _seed(seed)
        self.confidence = _finite_positive(confidence, "confidence")
        if self.confidence >= 1.0:
            raise ValueError("confidence must lie in (0, 1)")

    def _radius(self, count: int, time: int, n_candidates: int) -> float:
        if count == 0:
            return math.inf
        log_term = math.log(4.0 * n_candidates * max(1, time) ** 2 / self.confidence)
        return math.sqrt(log_term / (2.0 * count))

    def select(
        self,
        *,
        candidates: Sequence[Hashable],
        sampler: CandidateSampler,
        k: int,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float = 1.0,
        adjacency: Mapping[Hashable, Sequence[Hashable]] | None = None,
    ) -> SelectionResult:
        del adjacency
        pool, top_k, controller = _prepare(
            candidates=candidates,
            sampler=sampler,
            k=k,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        )
        ordinal = {candidate: index for index, candidate in enumerate(pool)}
        active = list(pool)
        eliminated: list[Hashable] = []
        rounds = 0

        # One fair initialization pass when affordable.  If it is not, the
        # generic priority rule below deterministically samples the prefix.
        if controller.can_complete_pass(len(active)):
            for candidate in active:
                controller.draw(candidate)
            rounds += 1

        while len(active) > top_k and controller.can_sample():
            time = controller.queries_used + 1
            radii = {
                candidate: self._radius(controller.statistics.counts[candidate], time, len(pool))
                for candidate in active
            }

            priority = max(
                active,
                key=partial(
                    _racing_allocation_priority,
                    statistics=controller.statistics,
                    radii=radii,
                    ordinal=ordinal,
                ),
            )
            controller.draw(priority)

            time = controller.queries_used
            lower = {
                candidate: max(
                    0.0,
                    controller.statistics.mean(candidate)
                    - self._radius(controller.statistics.counts[candidate], time, len(pool)),
                )
                for candidate in active
            }
            upper = {
                candidate: min(
                    1.0,
                    controller.statistics.mean(candidate)
                    + self._radius(controller.statistics.counts[candidate], time, len(pool)),
                )
                for candidate in active
            }
            newly_eliminated = [
                candidate
                for candidate in active
                if sum(lower[other] > upper[candidate] for other in active if other != candidate)
                >= top_k
            ]
            if newly_eliminated:
                newly_eliminated.sort(key=ordinal.__getitem__)
                eliminated.extend(newly_eliminated)
                removed = set(newly_eliminated)
                active = [candidate for candidate in active if candidate not in removed]
                rounds += 1

        active_rank = sorted(
            active,
            key=lambda candidate: (
                controller.statistics.counts[candidate] == 0,
                -controller.statistics.mean(candidate),
                ordinal[candidate],
            ),
        )
        eliminated_rank = sorted(
            eliminated,
            key=lambda candidate: (-controller.statistics.mean(candidate), ordinal[candidate]),
        )
        ranking = tuple(active_rank + eliminated_rank)
        return _result(
            selector_id=self.selector_id,
            candidates=pool,
            k=top_k,
            controller=controller,
            seed=self.seed,
            rounds_completed=rounds,
            eliminated=eliminated,
            ranking=ranking,
            converged=len(active) == top_k,
        )


class InternalCLUCBStyleSelector:
    """Internal CLUCB-style fixed-budget Top-k baseline.

    This is a small in-repository implementation of the characteristic LUCB
    boundary-pair rule; it is not presented as an official implementation of
    any external CLUCB codebase.  At each round it forms an empirical Top-k,
    pairs its least-certain member with the strongest outside challenger, and
    samples only that pair through :class:`CandidateSampler`.
    """

    selector_id = "internal_clucb_style_topk_v1"

    def __init__(self, *, confidence: float = 0.05, seed: int = 0) -> None:
        self.seed = _seed(seed)
        self.confidence = _finite_positive(confidence, "confidence")
        if self.confidence >= 1.0:
            raise ValueError("confidence must lie in (0, 1)")

    def _radius(self, count: int, time: int, n_candidates: int) -> float:
        if count == 0:
            return math.inf
        log_term = math.log(4.0 * n_candidates * max(1, time) ** 2 / self.confidence)
        return math.sqrt(log_term / (2.0 * count))

    def select(
        self,
        *,
        candidates: Sequence[Hashable],
        sampler: CandidateSampler,
        k: int,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float = 1.0,
        adjacency: Mapping[Hashable, Sequence[Hashable]] | None = None,
    ) -> SelectionResult:
        del adjacency
        pool, top_k, controller = _prepare(
            candidates=candidates,
            sampler=sampler,
            k=k,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        )
        ordinal = {candidate: index for index, candidate in enumerate(pool)}
        if top_k == len(pool):
            return _result(
                selector_id=self.selector_id,
                candidates=pool,
                k=top_k,
                controller=controller,
                seed=self.seed,
                rounds_completed=0,
                ranking=pool,
                converged=True,
            )

        rounds = 0
        # A complete initialization pass avoids assigning a fabricated mean to
        # an unseen arm.  Under a smaller budget the deterministic prefix is
        # initialized and the first outside challenger rule reaches unseen arms.
        if controller.can_complete_pass(len(pool)):
            for candidate in pool:
                controller.draw(candidate)
            rounds += 1
        else:
            for candidate in pool:
                if not controller.draw(candidate):
                    break

        converged = False
        while controller.can_sample():
            ranking = _rank(pool, controller.statistics)
            incumbent = ranking[:top_k]
            outside = ranking[top_k:]
            time = controller.queries_used + 1
            radii = {
                candidate: self._radius(controller.statistics.counts[candidate], time, len(pool))
                for candidate in pool
            }
            lower = {
                candidate: max(
                    0.0,
                    controller.statistics.mean(candidate) - radii[candidate],
                )
                for candidate in pool
            }
            upper = {
                candidate: min(
                    1.0,
                    controller.statistics.mean(candidate) + radii[candidate],
                )
                for candidate in pool
            }
            weakest = min(
                incumbent,
                key=lambda candidate: (lower[candidate], ordinal[candidate]),
            )
            challenger = max(
                outside,
                key=lambda candidate: (upper[candidate], -ordinal[candidate]),
            )
            separated = lower[weakest] > upper[challenger] or (
                lower[weakest] == upper[challenger] and ordinal[weakest] < ordinal[challenger]
            )
            if separated:
                converged = True
                break

            boundary_pair = (weakest, challenger)
            if controller.can_complete_pass(2):
                controller.draw(weakest)
                controller.draw(challenger)
                rounds += 1
                continue

            # A one-query tail cannot execute a full LUCB pair.  Spend it on
            # the less certain boundary arm, with frozen-order tie-breaking.
            tail = max(
                boundary_pair,
                key=lambda candidate: (radii[candidate], -ordinal[candidate]),
            )
            if controller.draw(tail):
                rounds += 1
            break

        return _result(
            selector_id=self.selector_id,
            candidates=pool,
            k=top_k,
            controller=controller,
            seed=self.seed,
            rounds_completed=rounds,
            converged=converged,
        )


def _normalize_adjacency(
    candidates: tuple[Hashable, ...],
    adjacency: Mapping[Hashable, Sequence[Hashable]] | None,
) -> Mapping[Hashable, tuple[Hashable, ...]]:
    if adjacency is None:
        raise ValueError("adjacency is required for GCGS-style graph search")
    pool = set(candidates)
    unknown_keys = set(adjacency) - pool
    if unknown_keys:
        raise ValueError("adjacency contains a key outside the candidate pool")
    ordinal = {candidate: index for index, candidate in enumerate(candidates)}
    normalized: dict[Hashable, tuple[Hashable, ...]] = {}
    for candidate in candidates:
        try:
            raw_neighbors = tuple(adjacency.get(candidate, ()))
            neighbor_set = set(raw_neighbors)
        except TypeError as error:
            raise TypeError("adjacency neighbors must be hashable sequences") from error
        if not neighbor_set.issubset(pool):
            raise ValueError("adjacency contains a neighbor outside the candidate pool")
        neighbor_set.discard(candidate)
        normalized[candidate] = tuple(sorted(neighbor_set, key=ordinal.__getitem__))
    return MappingProxyType(normalized)


class GCGSGreedyGraphSelector:
    """Seeded greedy local search on a frozen candidate-transformation graph.

    At each vertex, the method samples the vertex and every affordable
    neighbor once, moves to the empirically best strict improvement, and
    restarts from the next seeded root at a local optimum.  The final Top-k is
    ranked only by observations gathered through the sampler.
    """

    selector_id = "gcgs_style_greedy_graph_topk_v1"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = _seed(seed)

    def select(
        self,
        *,
        candidates: Sequence[Hashable],
        sampler: CandidateSampler,
        k: int,
        query_budget: int,
        cost_budget: float,
        sample_cost_upper_bound: float = 1.0,
        adjacency: Mapping[Hashable, Sequence[Hashable]] | None = None,
    ) -> SelectionResult:
        pool, top_k, controller = _prepare(
            candidates=candidates,
            sampler=sampler,
            k=k,
            query_budget=query_budget,
            cost_budget=cost_budget,
            sample_cost_upper_bound=sample_cost_upper_bound,
        )
        graph = _normalize_adjacency(pool, adjacency)
        ordinal = {candidate: index for index, candidate in enumerate(pool)}
        roots = list(pool)
        random.Random(self.seed).shuffle(roots)
        root_cursor = 0
        visited: set[Hashable] = set()
        current: Hashable | None = None
        rounds = 0

        while controller.can_sample():
            if current is None:
                while root_cursor < len(roots) and roots[root_cursor] in visited:
                    root_cursor += 1
                if root_cursor >= len(roots):
                    # Every component has been explored; refine the empirical
                    # leaders round-robin with the remaining strict budget.
                    ranking = _rank(pool, controller.statistics)
                    controller.draw(ranking[controller.queries_used % len(ranking)])
                    continue
                current = roots[root_cursor]
                root_cursor += 1

            local = (current, *graph[current])
            drew = False
            for candidate in local:
                if controller.draw(candidate):
                    drew = True
                    visited.add(candidate)
                else:
                    break
            if not drew:
                break
            rounds += 1
            observed = [
                candidate for candidate in local if controller.statistics.counts[candidate] > 0
            ]
            best = min(
                observed,
                key=lambda candidate: (-controller.statistics.mean(candidate), ordinal[candidate]),
            )
            if best != current and controller.statistics.mean(best) > controller.statistics.mean(
                current
            ):
                current = best
            else:
                current = None

        return _result(
            selector_id=self.selector_id,
            candidates=pool,
            k=top_k,
            controller=controller,
            seed=self.seed,
            rounds_completed=rounds,
        )


# A concise alias for experiment configurations that use the paper acronym.
GCGSSelector = GCGSGreedyGraphSelector
GCGSStyleGreedyGraphSelector = GCGSGreedyGraphSelector
UniformEmpiricalTopKSelector = UniformTopKSelector


__all__ = [
    "CLAIM_STATUS",
    "CandidateSampler",
    "CostAwareRacingSelector",
    "GCGSGreedyGraphSelector",
    "GCGSSelector",
    "GCGSStyleGreedyGraphSelector",
    "InternalCLUCBStyleSelector",
    "RandomSelector",
    "SelectionResourceLedger",
    "SelectionResult",
    "SelectionTraceRecord",
    "SuccessiveHalvingSelector",
    "UniformEmpiricalTopKSelector",
    "UniformTopKSelector",
]
