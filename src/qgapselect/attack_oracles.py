"""Frozen, non-peeking source reward oracles for selector experiments.

The trusted experiment runner owns a :class:`FrozenSourceFixture`.  A selector
receives only the :class:`BlindSourceRewardOracle` returned by
``fixture.open_oracle``.  The selector-facing object exposes candidate graph
structure, observations already paid for, and an audit snapshot; it does not
expose reward streams, configured means, empirical means, or a random seed.

This is an API separation for reproducible experiments, not a hostile Python
sandbox.  Selectors under evaluation must be passed the oracle, never the
fixture, tensor, or evaluator object.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import random
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType


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


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bernoulli_outcome(value: object, name: str) -> int:
    try:
        result = _integer(value, name, minimum=0)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer Bernoulli outcome in {{0, 1}}") from error
    if result > 1:
        raise ValueError(f"{name} must be a Bernoulli outcome in {{0, 1}}")
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


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """Immutable identity of one strategy in the fixed candidate pool.

    ``payload_hash`` should commit to the exact prompt/strategy artifact kept by
    the runner.  The artifact itself is intentionally outside the oracle.
    """

    candidate_id: str
    payload_hash: str = ""
    family: str = "unspecified"

    def __post_init__(self) -> None:
        _nonempty_string(self.candidate_id, "candidate_id")
        if not isinstance(self.payload_hash, str):
            raise TypeError("payload_hash must be a string")
        _nonempty_string(self.family, "family")

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "payload_hash": self.payload_hash,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    """Immutable directed edge in the preregistered strategy graph."""

    source: str
    target: str
    relation: str = "transform"

    def __post_init__(self) -> None:
        _nonempty_string(self.source, "source")
        _nonempty_string(self.target, "target")
        _nonempty_string(self.relation, "relation")

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
        }


@dataclass(frozen=True, slots=True)
class FrozenCandidateGraph:
    """Fixed candidate pool and directed transformation graph."""

    candidates: tuple[SourceCandidate, ...]
    edges: tuple[CandidateEdge, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        edges = tuple(self.edges)
        if not candidates:
            raise ValueError("candidate graph cannot be empty")
        if any(not isinstance(item, SourceCandidate) for item in candidates):
            raise TypeError("candidates must contain SourceCandidate objects")
        if any(not isinstance(item, CandidateEdge) for item in edges):
            raise TypeError("edges must contain CandidateEdge objects")
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        known = set(candidate_ids)
        for edge in edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"edge {edge.source!r}->{edge.target!r} references an unknown candidate"
                )
        edge_keys = tuple((edge.source, edge.target, edge.relation) for edge in edges)
        if len(set(edge_keys)) != len(edge_keys):
            raise ValueError("candidate graph cannot contain duplicate edges")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_ids(
        cls,
        candidate_ids: Sequence[str],
        *,
        edges: Sequence[CandidateEdge] = (),
    ) -> FrozenCandidateGraph:
        """Build a graph when candidate payloads are managed separately."""

        return cls(
            tuple(SourceCandidate(candidate_id) for candidate_id in candidate_ids),
            tuple(edges),
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)

    def successors(self, candidate_id: str) -> tuple[str, ...]:
        candidate_id = _nonempty_string(candidate_id, "candidate_id")
        if candidate_id not in set(self.candidate_ids):
            raise KeyError(candidate_id)
        return tuple(edge.target for edge in self.edges if edge.source == candidate_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "edges": [edge.as_dict() for edge in self.edges],
        }


@dataclass(frozen=True, slots=True)
class SourceOracleBudget:
    """Hard accepted-query and evaluation-cost caps for one selector run."""

    max_queries: int
    max_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_queries",
            _integer(self.max_queries, "max_queries", minimum=0),
        )
        cost = _finite_number(self.max_cost, "max_cost")
        if cost < 0.0:
            raise ValueError("max_cost must be non-negative")
        object.__setattr__(self, "max_cost", cost)


@dataclass(frozen=True, slots=True)
class FrozenRewardCostTensor:
    """Trusted-runner artifact containing aligned, immutable sample streams."""

    graph: FrozenCandidateGraph
    reward_streams: tuple[tuple[int, ...], ...]
    cost_streams: tuple[tuple[float, ...], ...]
    truth_commitment: str
    metadata: tuple[tuple[str, str], ...] = ()
    manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.graph, FrozenCandidateGraph):
            raise TypeError("graph must be a FrozenCandidateGraph")
        rewards = tuple(tuple(stream) for stream in self.reward_streams)
        costs = tuple(tuple(stream) for stream in self.cost_streams)
        n_candidates = len(self.graph.candidates)
        if len(rewards) != n_candidates or len(costs) != n_candidates:
            raise ValueError("reward and cost streams must align with every candidate")
        normalized_rewards: list[tuple[int, ...]] = []
        normalized_costs: list[tuple[float, ...]] = []
        for index, (reward_stream, cost_stream) in enumerate(
            zip(rewards, costs, strict=True)
        ):
            if not reward_stream:
                raise ValueError(f"candidate stream {index} cannot be empty")
            if len(reward_stream) != len(cost_stream):
                raise ValueError(f"reward and cost stream {index} lengths differ")
            normalized_rewards.append(
                tuple(
                    _bernoulli_outcome(value, f"reward_streams[{index}]")
                    for value in reward_stream
                )
            )
            normalized_costs.append(
                tuple(
                    _finite_number(value, f"cost_streams[{index}]", positive=True)
                    for value in cost_stream
                )
            )
        truth_commitment = _nonempty_string(
            self.truth_commitment, "truth_commitment"
        )
        metadata = tuple((str(key), str(value)) for key, value in self.metadata)
        if any(not key for key, _ in metadata):
            raise ValueError("metadata keys cannot be empty")
        if len({key for key, _ in metadata}) != len(metadata):
            raise ValueError("metadata keys must be unique")
        metadata = tuple(sorted(metadata))
        object.__setattr__(self, "reward_streams", tuple(normalized_rewards))
        object.__setattr__(self, "cost_streams", tuple(normalized_costs))
        object.__setattr__(self, "truth_commitment", truth_commitment)
        object.__setattr__(self, "metadata", metadata)
        document: dict[str, object] = {
            "schema": "qgapselect.frozen-source-oracle.v1",
            "graph": self.graph.as_dict(),
            "reward_streams": [list(stream) for stream in normalized_rewards],
            "cost_streams": [list(stream) for stream in normalized_costs],
            "truth_commitment": truth_commitment,
            "metadata": dict(metadata),
        }
        object.__setattr__(self, "manifest_hash", _canonical_hash(document))

    @property
    def stream_lengths(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                candidate_id: len(self.reward_streams[index])
                for index, candidate_id in enumerate(self.graph.candidate_ids)
            }
        )


class SourceQueryStatus(str, Enum):
    """Outcome of one selector query attempt."""

    ACCEPTED = "accepted"
    QUERY_BUDGET_EXHAUSTED = "query_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    STREAM_EXHAUSTED = "stream_exhausted"


@dataclass(frozen=True, slots=True)
class SourceObservation:
    """One paid observation returned to a selector."""

    candidate_id: str
    reward: int
    cost: float
    candidate_query_index: int
    global_query_index: int
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class SourceQueryRecord:
    """Immutable accepted or rejected query-attempt audit record."""

    attempt_index: int
    candidate_id: str
    status: SourceQueryStatus
    tag: str | None
    candidate_query_index: int | None = None
    global_query_index: int | None = None
    reward: int | None = None
    cost: float | None = None


@dataclass(frozen=True, slots=True)
class SourceOracleSnapshot:
    """Immutable point-in-time source-oracle ledger."""

    manifest_hash: str
    budget: SourceOracleBudget
    queries_used: int
    cost_used: float
    cursors: Mapping[str, int]
    records: tuple[SourceQueryRecord, ...]

    @property
    def attempts(self) -> int:
        return len(self.records)

    @property
    def remaining_queries(self) -> int:
        return self.budget.max_queries - self.queries_used

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.budget.max_cost - self.cost_used)


class SourceOracleError(RuntimeError):
    """Base class for a rejected source-oracle query."""


class SourceBudgetExhaustedError(SourceOracleError):
    """Raised before consumption when a hard query or cost cap would be crossed."""

    def __init__(self, candidate_id: str, status: SourceQueryStatus) -> None:
        self.candidate_id = candidate_id
        self.status = status
        super().__init__(f"{status.value} for candidate {candidate_id!r}")


class SourceStreamExhaustedError(SourceOracleError):
    """Raised before consumption when a candidate's frozen stream is empty."""

    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.status = SourceQueryStatus.STREAM_EXHAUSTED
        super().__init__(f"frozen reward stream exhausted for candidate {candidate_id!r}")


class BlindSourceRewardOracle:
    """Selector-facing, cursor-based access to one frozen source tensor.

    Each candidate has an independent cursor.  A rejected attempt is recorded,
    but it consumes neither a query, cost, nor a frozen outcome.
    """

    __slots__ = (
        "__budget",
        "__cost_streams",
        "__cost_used",
        "__cost_used_decimal",
        "__cursors",
        "__graph",
        "__id_to_index",
        "__lock",
        "__manifest_hash",
        "__queries_used",
        "__records",
        "__reward_streams",
    )

    def __init__(
        self,
        tensor: FrozenRewardCostTensor,
        budget: SourceOracleBudget,
    ) -> None:
        if not isinstance(tensor, FrozenRewardCostTensor):
            raise TypeError("tensor must be a FrozenRewardCostTensor")
        if not isinstance(budget, SourceOracleBudget):
            raise TypeError("budget must be a SourceOracleBudget")
        self.__graph = tensor.graph
        self.__manifest_hash = tensor.manifest_hash
        self.__reward_streams = tensor.reward_streams
        self.__cost_streams = tensor.cost_streams
        self.__id_to_index = {
            candidate_id: index
            for index, candidate_id in enumerate(tensor.graph.candidate_ids)
        }
        self.__budget = budget
        self.__cursors = [0] * len(tensor.graph.candidates)
        self.__queries_used = 0
        self.__cost_used = 0.0
        self.__cost_used_decimal = Decimal("0")
        self.__records: list[SourceQueryRecord] = []
        self.__lock = threading.RLock()

    @property
    def graph(self) -> FrozenCandidateGraph:
        return self.__graph

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self.__graph.candidate_ids

    @property
    def n_candidates(self) -> int:
        return len(self.__graph.candidates)

    @property
    def manifest_hash(self) -> str:
        return self.__manifest_hash

    @property
    def budget(self) -> SourceOracleBudget:
        return self.__budget

    def _record_rejection(
        self,
        candidate_id: str,
        status: SourceQueryStatus,
        tag: str | None,
    ) -> None:
        self.__records.append(
            SourceQueryRecord(
                attempt_index=len(self.__records) + 1,
                candidate_id=candidate_id,
                status=status,
                tag=tag,
            )
        )

    def query(self, candidate_id: str, *, tag: str | None = None) -> SourceObservation:
        """Consume exactly one frozen observation if both hard budgets permit it."""

        candidate_id = _nonempty_string(candidate_id, "candidate_id")
        if tag is not None and not isinstance(tag, str):
            raise TypeError("tag must be a string or None")
        try:
            candidate_index = self.__id_to_index[candidate_id]
        except KeyError:
            raise KeyError(candidate_id) from None
        with self.__lock:
            if self.__queries_used >= self.__budget.max_queries:
                status = SourceQueryStatus.QUERY_BUDGET_EXHAUSTED
                self._record_rejection(candidate_id, status, tag)
                raise SourceBudgetExhaustedError(candidate_id, status)
            cursor = self.__cursors[candidate_index]
            if cursor >= len(self.__reward_streams[candidate_index]):
                self._record_rejection(
                    candidate_id, SourceQueryStatus.STREAM_EXHAUSTED, tag
                )
                raise SourceStreamExhaustedError(candidate_id)
            cost = self.__cost_streams[candidate_index][cursor]
            proposed_cost = self.__cost_used_decimal + Decimal(str(cost))
            if proposed_cost > Decimal(str(self.__budget.max_cost)):
                status = SourceQueryStatus.COST_BUDGET_EXHAUSTED
                self._record_rejection(candidate_id, status, tag)
                raise SourceBudgetExhaustedError(candidate_id, status)
            reward = self.__reward_streams[candidate_index][cursor]
            self.__cursors[candidate_index] += 1
            self.__queries_used += 1
            self.__cost_used_decimal = proposed_cost
            self.__cost_used = float(proposed_cost)
            observation = SourceObservation(
                candidate_id=candidate_id,
                reward=reward,
                cost=cost,
                candidate_query_index=cursor,
                global_query_index=self.__queries_used,
                manifest_hash=self.__manifest_hash,
            )
            self.__records.append(
                SourceQueryRecord(
                    attempt_index=len(self.__records) + 1,
                    candidate_id=candidate_id,
                    status=SourceQueryStatus.ACCEPTED,
                    tag=tag,
                    candidate_query_index=cursor,
                    global_query_index=self.__queries_used,
                    reward=reward,
                    cost=cost,
                )
            )
            return observation

    def snapshot(self) -> SourceOracleSnapshot:
        """Return an immutable audit view without unconsumed rewards or means."""

        with self.__lock:
            return SourceOracleSnapshot(
                manifest_hash=self.__manifest_hash,
                budget=self.__budget,
                queries_used=self.__queries_used,
                cost_used=self.__cost_used,
                cursors=MappingProxyType(
                    {
                        candidate_id: self.__cursors[index]
                        for candidate_id, index in self.__id_to_index.items()
                    }
                ),
                records=tuple(self.__records),
            )


@dataclass(frozen=True, slots=True)
class SourceSelectionEvaluation:
    """Evaluator-only Top-k quality report for one selector output."""

    selected_ids: tuple[str, ...]
    reference_top_k: tuple[str, ...]
    hits: int
    precision_at_k: float
    recall_at_k: float
    selected_expected_reward: float
    optimal_expected_reward: float
    top_k_regret: float
    exact_match: bool


@dataclass(frozen=True, slots=True)
class SourceGroundTruth:
    """Ground truth held by the runner and never passed to a selector."""

    candidate_ids: tuple[str, ...]
    configured_means: tuple[float, ...]
    frozen_means: tuple[float, ...]
    truth_commitment: str
    manifest_hash: str

    def __post_init__(self) -> None:
        candidate_ids = tuple(self.candidate_ids)
        configured = tuple(
            _finite_number(value, "configured mean") for value in self.configured_means
        )
        frozen = tuple(_finite_number(value, "frozen mean") for value in self.frozen_means)
        if len(candidate_ids) != len(configured) or len(candidate_ids) != len(frozen):
            raise ValueError("ground-truth arrays must align with candidate IDs")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("ground-truth candidate IDs must be unique")
        if any(not 0.0 <= value <= 1.0 for value in configured + frozen):
            raise ValueError("ground-truth Bernoulli means must be in [0, 1]")
        expected_commitment = _canonical_hash(
            {
                "candidate_ids": list(candidate_ids),
                "configured_means": list(configured),
                "frozen_means": list(frozen),
            }
        )
        if self.truth_commitment != expected_commitment:
            raise ValueError("ground-truth values do not match truth_commitment")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "configured_means", configured)
        object.__setattr__(self, "frozen_means", frozen)

    @property
    def expected_mean_by_candidate(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.candidate_ids, self.configured_means, strict=True))
        )

    @property
    def frozen_mean_by_candidate(self) -> Mapping[str, float]:
        return MappingProxyType(
            dict(zip(self.candidate_ids, self.frozen_means, strict=True))
        )

    def top_k(self, k: int) -> tuple[str, ...]:
        k = _integer(k, "k", minimum=1)
        if k > len(self.candidate_ids):
            raise ValueError("k cannot exceed the candidate count")
        order = sorted(
            range(len(self.candidate_ids)),
            key=lambda index: (-self.configured_means[index], index),
        )
        return tuple(self.candidate_ids[index] for index in order[:k])

    def evaluate(
        self,
        selected_ids: Sequence[str],
        *,
        k: int,
    ) -> SourceSelectionEvaluation:
        """Evaluate up to ``k`` unique outputs against configured expected means."""

        k = _integer(k, "k", minimum=1)
        selected = tuple(selected_ids)
        if len(selected) > k:
            raise ValueError("selector returned more than k candidates")
        if len(set(selected)) != len(selected):
            raise ValueError("selector output must contain unique candidate IDs")
        means = self.expected_mean_by_candidate
        unknown = tuple(candidate_id for candidate_id in selected if candidate_id not in means)
        if unknown:
            raise ValueError(f"selector returned unknown candidates: {unknown}")
        reference = self.top_k(k)
        hits = len(set(selected) & set(reference))
        selected_reward = math.fsum(means[candidate_id] for candidate_id in selected)
        optimal_reward = math.fsum(means[candidate_id] for candidate_id in reference)
        return SourceSelectionEvaluation(
            selected_ids=selected,
            reference_top_k=reference,
            hits=hits,
            precision_at_k=hits / len(selected) if selected else 0.0,
            recall_at_k=hits / k,
            selected_expected_reward=selected_reward,
            optimal_expected_reward=optimal_reward,
            top_k_regret=max(0.0, optimal_reward - selected_reward),
            exact_match=len(selected) == k and set(selected) == set(reference),
        )


@dataclass(frozen=True, slots=True)
class FrozenSourceFixture:
    """Trusted-runner bundle; pass only ``open_oracle(...)`` to selectors."""

    tensor: FrozenRewardCostTensor
    evaluator: SourceGroundTruth

    def __post_init__(self) -> None:
        if self.tensor.manifest_hash != self.evaluator.manifest_hash:
            raise ValueError("tensor and evaluator manifest hashes differ")
        if self.tensor.truth_commitment != self.evaluator.truth_commitment:
            raise ValueError("tensor and evaluator truth commitments differ")

    @property
    def manifest_hash(self) -> str:
        return self.tensor.manifest_hash

    def open_oracle(self, budget: SourceOracleBudget) -> BlindSourceRewardOracle:
        return BlindSourceRewardOracle(self.tensor, budget)


def _aligned_mapping(
    values: Mapping[str, Sequence[object]],
    candidate_ids: tuple[str, ...],
    name: str,
) -> tuple[tuple[object, ...], ...]:
    if set(values) != set(candidate_ids):
        missing = sorted(set(candidate_ids) - set(values))
        extra = sorted(set(values) - set(candidate_ids))
        raise ValueError(f"{name} keys differ from candidate IDs; missing={missing}, extra={extra}")
    return tuple(tuple(values[candidate_id]) for candidate_id in candidate_ids)


def freeze_source_streams(
    graph: FrozenCandidateGraph,
    reward_streams: Mapping[str, Sequence[int]],
    cost_streams: Mapping[str, Sequence[float]],
    *,
    configured_means: Mapping[str, float] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> FrozenSourceFixture:
    """Freeze exact streams and return separated selector/evaluator artifacts."""

    if not isinstance(graph, FrozenCandidateGraph):
        raise TypeError("graph must be a FrozenCandidateGraph")
    candidate_ids = graph.candidate_ids
    rewards_untyped = _aligned_mapping(reward_streams, candidate_ids, "reward_streams")
    costs_untyped = _aligned_mapping(cost_streams, candidate_ids, "cost_streams")
    rewards = tuple(
        tuple(
            _bernoulli_outcome(value, f"reward_streams[{stream_index}]")
            for value in stream
        )
        for stream_index, stream in enumerate(rewards_untyped)
    )
    costs = tuple(
        tuple(
            _finite_number(value, f"cost_streams[{stream_index}]", positive=True)
            for value in stream
        )
        for stream_index, stream in enumerate(costs_untyped)
    )
    if any(not stream for stream in rewards):
        raise ValueError("candidate streams cannot be empty")
    if any(
        len(reward) != len(cost)
        for reward, cost in zip(rewards, costs, strict=True)
    ):
        raise ValueError("reward and cost stream lengths differ")
    frozen_means = tuple(math.fsum(stream) / len(stream) for stream in rewards)
    if configured_means is None:
        expected = frozen_means
    else:
        if set(configured_means) != set(candidate_ids):
            raise ValueError("configured_means keys must exactly match candidate IDs")
        expected = tuple(
            _finite_number(configured_means[candidate_id], "configured mean")
            for candidate_id in candidate_ids
        )
        if any(not 0.0 <= value <= 1.0 for value in expected):
            raise ValueError("configured Bernoulli means must be in [0, 1]")
    truth_document: dict[str, object] = {
        "candidate_ids": list(candidate_ids),
        "configured_means": list(expected),
        "frozen_means": list(frozen_means),
    }
    truth_commitment = _canonical_hash(truth_document)
    tensor = FrozenRewardCostTensor(
        graph=graph,
        reward_streams=rewards,
        cost_streams=costs,
        truth_commitment=truth_commitment,
        metadata=tuple((str(key), str(value)) for key, value in (metadata or {}).items()),
    )
    evaluator = SourceGroundTruth(
        candidate_ids=candidate_ids,
        configured_means=expected,
        frozen_means=frozen_means,
        truth_commitment=truth_commitment,
        manifest_hash=tensor.manifest_hash,
    )
    return FrozenSourceFixture(tensor=tensor, evaluator=evaluator)


def _derived_rng(seed: int, candidate_id: str, purpose: str) -> random.Random:
    material = f"qgapselect\0{seed}\0{candidate_id}\0{purpose}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(derived_seed)


def generate_bernoulli_landscape(
    means: Mapping[str, float],
    *,
    samples_per_candidate: int,
    seed: int,
    candidate_costs: Mapping[str, float] | float = 1.0,
    cost_jitter: float = 0.0,
    graph: FrozenCandidateGraph | None = None,
    edges: Sequence[CandidateEdge] = (),
) -> FrozenSourceFixture:
    """Generate and freeze independent Bernoulli reward and cost streams.

    Candidate-specific RNGs make each stream invariant to query order and to
    the ordering of unrelated candidates.  ``cost_jitter`` produces an
    independently frozen multiplicative uniform perturbation around each
    candidate's base cost.
    """

    if not isinstance(means, Mapping) or not means:
        raise TypeError("means must be a non-empty mapping")
    samples_per_candidate = _integer(
        samples_per_candidate, "samples_per_candidate", minimum=1
    )
    seed = _integer(seed, "seed", minimum=0)
    jitter = _finite_number(cost_jitter, "cost_jitter")
    if not 0.0 <= jitter < 1.0:
        raise ValueError("cost_jitter must be in [0, 1)")
    if graph is None:
        graph = FrozenCandidateGraph.from_ids(tuple(means), edges=tuple(edges))
    elif edges:
        raise ValueError("pass edges through graph when graph is supplied")
    candidate_ids = graph.candidate_ids
    if set(means) != set(candidate_ids):
        raise ValueError("means keys must exactly match graph candidate IDs")
    normalized_means = {
        candidate_id: _finite_number(means[candidate_id], "mean")
        for candidate_id in candidate_ids
    }
    if any(not 0.0 <= mean <= 1.0 for mean in normalized_means.values()):
        raise ValueError("Bernoulli means must be in [0, 1]")
    if isinstance(candidate_costs, Mapping):
        if set(candidate_costs) != set(candidate_ids):
            raise ValueError("candidate_costs keys must exactly match graph candidate IDs")
        base_costs = {
            candidate_id: _finite_number(
                candidate_costs[candidate_id], "candidate cost", positive=True
            )
            for candidate_id in candidate_ids
        }
    else:
        base_cost = _finite_number(candidate_costs, "candidate_costs", positive=True)
        base_costs = dict.fromkeys(candidate_ids, base_cost)
    reward_streams: dict[str, tuple[int, ...]] = {}
    cost_streams: dict[str, tuple[float, ...]] = {}
    for candidate_id in candidate_ids:
        reward_rng = _derived_rng(seed, candidate_id, "reward")
        reward_streams[candidate_id] = tuple(
            int(reward_rng.random() < normalized_means[candidate_id])
            for _ in range(samples_per_candidate)
        )
        base_cost = base_costs[candidate_id]
        if jitter == 0.0:
            cost_streams[candidate_id] = (base_cost,) * samples_per_candidate
        else:
            cost_rng = _derived_rng(seed, candidate_id, "cost")
            cost_streams[candidate_id] = tuple(
                base_cost * (1.0 + jitter * (2.0 * cost_rng.random() - 1.0))
                for _ in range(samples_per_candidate)
            )
    return freeze_source_streams(
        graph,
        reward_streams,
        cost_streams,
        configured_means=normalized_means,
        metadata={
            "generator": "candidate-independent-bernoulli-v1",
            "samples_per_candidate": str(samples_per_candidate),
            "seed": str(seed),
            "cost_jitter": repr(jitter),
        },
    )
