"""Executable small-state primitives for coherent Q-GapSelect experiments.

The routines in this module compose charged reward-oracle experiments with an
exact reversible phase marker and exact-state Grover search.  They are useful
for falsifying circuit semantics and resource accounting on small instances.
The current boundary routine first obtains a classical certificate and then
compiles its already-known set into the phase marker.  Consequently the Grover
stage is certificate enumeration, not coherent discovery of an unknown Top-k
set.  These routines do **not** claim the proposed heterogeneous-gap algorithm,
complexity bound, or scalable physical implementation.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .contracts import CoherentOracleContract
from .oracles import QueryKind, QueryLedger, QuerySnapshot

ComplexState = NDArray[np.complex128]
BACKEND_NAME = "numpy_exact_statevector_small_scale"
CLAIM_STATUS = "certificate_compiled_reference_not_quantum_discovery_or_theorem"


def _integer(value: object, name: str) -> int:
    """Accept exact integer-like inputs without silently truncating floats."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


class ExactRewardOracle(Protocol):
    """Shared subset implemented by canonical and natural exact-state oracles."""

    @property
    def contract(self) -> CoherentOracleContract: ...

    @property
    def n_arms(self) -> int: ...

    @property
    def index_dimension(self) -> int: ...

    def index_superposition(
        self,
        indices: Sequence[int] | None = None,
        *,
        controlled: bool = False,
        active_control: bool = True,
    ) -> ComplexState: ...

    def apply(
        self,
        state: ComplexState,
        *,
        inverse: bool = False,
        controlled: bool = False,
        tag: str | None = None,
    ) -> ComplexState: ...

    def query_snapshot(self) -> QuerySnapshot: ...


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _zero_query_counts() -> dict[str, int]:
    result = {kind.value: 0 for kind in QueryKind}
    result.update(coherent_total=0, classical_total=0, total=0)
    return result


@dataclass(frozen=True, slots=True)
class PrimitiveResources:
    """Auditable logical resources, separate from statevector wall-clock work."""

    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    phase_oracle_queries: int
    depth: int
    qubits: int
    workspace_qubits: int
    uncompute_residual: float
    backend: str = BACKEND_NAME

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))

    @property
    def gates(self) -> int:
        return sum(self.gate_counts.values())


class _GateCounter:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.depth = 0
        self.phase_oracle_queries = 0
        self.max_qubits = 0
        self.max_workspace_qubits = 0
        self.max_uncompute_residual = 0.0

    def add(self, name: str, count: int = 1, *, depth: int | None = None) -> None:
        if count < 0:
            raise ValueError("gate count cannot be negative")
        self.counts[name] += count
        self.depth += count if depth is None else depth

    def observe_registers(self, qubits: int, workspace_qubits: int = 0) -> None:
        self.max_qubits = max(self.max_qubits, qubits)
        self.max_workspace_qubits = max(
            self.max_workspace_qubits,
            workspace_qubits,
        )

    def merge(self, resources: PrimitiveResources) -> None:
        self.counts.update(resources.gate_counts)
        self.depth += resources.depth
        self.phase_oracle_queries += resources.phase_oracle_queries
        self.max_qubits = max(self.max_qubits, resources.qubits)
        self.max_workspace_qubits = max(
            self.max_workspace_qubits,
            resources.workspace_qubits,
        )
        self.max_uncompute_residual = max(
            self.max_uncompute_residual,
            resources.uncompute_residual,
        )

    def resources(
        self,
        query_counts: Mapping[str, int] | None = None,
    ) -> PrimitiveResources:
        return PrimitiveResources(
            query_counts=_immutable_counts(
                _zero_query_counts() if query_counts is None else query_counts
            ),
            gate_counts=_immutable_counts(self.counts),
            phase_oracle_queries=self.phase_oracle_queries,
            depth=self.depth,
            qubits=self.max_qubits,
            workspace_qubits=self.max_workspace_qubits,
            uncompute_residual=self.max_uncompute_residual,
        )


@dataclass(frozen=True, slots=True)
class BoundaryArmInterval:
    """Measured confidence interval for one reward arm."""

    arm: int
    successes: int
    shots: int
    estimate: float
    lower: float
    upper: float
    angular_lower: float
    angular_upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class BoundaryCertificate:
    """A simultaneous separation certificate for one exact Top-k set."""

    selected: tuple[int, ...]
    rejected: tuple[int, ...]
    mean_threshold: float | None
    angular_threshold: float | None
    mean_margin: float
    angular_margin: float
    confidence: float
    complete: bool = True


@dataclass(frozen=True, slots=True)
class BoundaryRound:
    """One cumulative sampling round of :class:`QBoundaryEstimator`."""

    round_index: int
    target_shots_per_arm: int
    intervals: tuple[BoundaryArmInterval, ...]
    candidate_selected: tuple[int, ...]
    candidate_rejected: tuple[int, ...]
    separated: bool


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    """Output of charged boundary experiments; no hidden mean is read."""

    k: int
    intervals: tuple[BoundaryArmInterval, ...]
    candidate_selected: tuple[int, ...]
    candidate_rejected: tuple[int, ...]
    order_statistic_mean_interval: tuple[float, float]
    order_statistic_angular_interval: tuple[float, float]
    certificate: BoundaryCertificate | None
    complete: bool
    rounds: tuple[BoundaryRound, ...]
    resources: PrimitiveResources
    minimum_angular_margin: float = 0.0
    backend: str = BACKEND_NAME
    claim_status: str = CLAIM_STATUS


class QBoundaryEstimator:
    """Resumable fixed-confidence boundary construction from oracle experiments.

    Each shot prepares ``|arm, 0>`` and calls the supplied coherent reward
    oracle once.  A measurement is then sampled from the resulting exact
    statevector.  Confidence intervals use a finite-round Hoeffding union bound.
    This deliberately conservative estimator is executable but is not claimed
    to be the final quantum boundary algorithm.
    """

    def __init__(
        self,
        oracle: ExactRewardOracle,
        k: int,
        *,
        confidence: float = 0.05,
        shots_per_round: int = 32,
        max_rounds: int = 8,
        minimum_angular_margin: float = 0.0,
        seed: int | None = None,
        tag: str = "qboundary",
    ) -> None:
        k = _integer(k, "k")
        shots_per_round = _integer(shots_per_round, "shots_per_round")
        max_rounds = _integer(max_rounds, "max_rounds")
        if not 1 <= k <= oracle.n_arms:
            raise ValueError("k must be in {1, ..., number of arms}")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if shots_per_round <= 0:
            raise ValueError("shots_per_round must be positive")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if isinstance(minimum_angular_margin, bool):
            raise TypeError("minimum_angular_margin must be a real number")
        try:
            minimum_angular_margin = float(minimum_angular_margin)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "minimum_angular_margin must be a real number"
            ) from error
        if (
            not math.isfinite(minimum_angular_margin)
            or not 0.0 <= minimum_angular_margin <= math.pi / 2.0
        ):
            raise ValueError(
                "minimum_angular_margin must lie in [0, pi/2]"
            )

        self.oracle = oracle
        self.k = k
        self.confidence = float(confidence)
        self.shots_per_round = shots_per_round
        self.max_rounds = max_rounds
        self.minimum_angular_margin = minimum_angular_margin
        self.tag = str(tag)
        self._rng = np.random.default_rng(seed)
        self._successes = np.zeros(oracle.n_arms, dtype=np.int64)
        self._shots = np.zeros(oracle.n_arms, dtype=np.int64)
        self._rounds: list[BoundaryRound] = []
        self._certificate: BoundaryCertificate | None = None
        self._before_queries = oracle.query_snapshot()
        # Count only calls made inside this estimator.  A live difference from
        # the construction snapshot would incorrectly absorb later searches
        # or other users of the shared oracle between resumptions.
        self._owned_queries: Counter[str] = Counter()
        self._gates = _GateCounter()
        self._gates.observe_registers(
            oracle.contract.index_qubits
            + oracle.contract.workspace_qubits
            + 1,
            oracle.contract.workspace_qubits,
        )

        if self.k == oracle.n_arms:
            self._certificate = BoundaryCertificate(
                selected=tuple(range(oracle.n_arms)),
                rejected=(),
                mean_threshold=None,
                angular_threshold=None,
                mean_margin=math.inf,
                angular_margin=math.inf,
                confidence=self.confidence,
            )

    @property
    def complete(self) -> bool:
        return self._certificate is not None

    @property
    def exhausted(self) -> bool:
        return self.complete or len(self._rounds) >= self.max_rounds

    def _measure_arm_once(self, arm: int, round_index: int) -> int:
        prepared = self.oracle.index_superposition((arm,))
        self._gates.add("index_state_preparation")
        output = self.oracle.apply(
            prepared,
            tag=f"{self.tag}_round_{round_index}",
        )
        self._gates.add("coherent_reward_oracle")

        local_dimension = output.size // self.oracle.index_dimension
        if local_dimension % 2 != 0:
            raise RuntimeError("reward qubit is not the final local register")
        shaped = output.reshape(
            self.oracle.index_dimension,
            local_dimension // 2,
            2,
        )
        probability = float(np.sum(np.abs(shaped[..., 1]) ** 2))
        probability = min(max(probability, 0.0), 1.0)
        self._gates.add("reward_measurement")
        return int(self._rng.random() < probability)

    def _intervals(self) -> tuple[BoundaryArmInterval, ...]:
        if not np.all(self._shots > 0):
            return tuple(
                BoundaryArmInterval(
                    arm=arm,
                    successes=0,
                    shots=0,
                    estimate=0.5,
                    lower=0.0,
                    upper=1.0,
                    angular_lower=0.0,
                    angular_upper=math.pi / 2.0,
                )
                for arm in range(self.oracle.n_arms)
            )

        log_term = math.log(
            2.0 * self.oracle.n_arms * self.max_rounds / self.confidence
        )
        intervals: list[BoundaryArmInterval] = []
        for arm in range(self.oracle.n_arms):
            shots = int(self._shots[arm])
            successes = int(self._successes[arm])
            estimate = successes / shots
            radius = math.sqrt(log_term / (2.0 * shots))
            lower = max(0.0, estimate - radius)
            upper = min(1.0, estimate + radius)
            intervals.append(
                BoundaryArmInterval(
                    arm=arm,
                    successes=successes,
                    shots=shots,
                    estimate=estimate,
                    lower=lower,
                    upper=upper,
                    angular_lower=math.asin(math.sqrt(lower)),
                    angular_upper=math.asin(math.sqrt(upper)),
                )
            )
        return tuple(intervals)

    def _classify(
        self,
        intervals: tuple[BoundaryArmInterval, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], BoundaryCertificate | None]:
        ranking = sorted(
            range(self.oracle.n_arms),
            key=lambda arm: (-intervals[arm].estimate, arm),
        )
        selected = tuple(sorted(ranking[: self.k]))
        rejected = tuple(sorted(ranking[self.k :]))
        if not rejected:
            return selected, rejected, self._certificate

        selected_mean_floor = min(intervals[arm].lower for arm in selected)
        rejected_mean_ceiling = max(intervals[arm].upper for arm in rejected)
        selected_angle_floor = min(
            intervals[arm].angular_lower for arm in selected
        )
        rejected_angle_ceiling = max(
            intervals[arm].angular_upper for arm in rejected
        )
        angular_margin = selected_angle_floor - rejected_angle_ceiling
        if (
            selected_mean_floor <= rejected_mean_ceiling
            or angular_margin < self.minimum_angular_margin
        ):
            return selected, rejected, None

        return (
            selected,
            rejected,
            BoundaryCertificate(
                selected=selected,
                rejected=rejected,
                mean_threshold=0.5
                * (selected_mean_floor + rejected_mean_ceiling),
                angular_threshold=0.5
                * (selected_angle_floor + rejected_angle_ceiling),
                mean_margin=selected_mean_floor - rejected_mean_ceiling,
                angular_margin=angular_margin,
                confidence=self.confidence,
            ),
        )

    def step(self) -> BoundaryResult:
        """Execute one doubling round and preserve state for a later resume."""

        if self.exhausted:
            return self.result()
        before_step = self.oracle.query_snapshot()
        round_index = len(self._rounds) + 1
        target_shots = self.shots_per_round * (2 ** (round_index - 1))
        for arm in range(self.oracle.n_arms):
            additional = target_shots - int(self._shots[arm])
            for _ in range(additional):
                self._successes[arm] += self._measure_arm_once(arm, round_index)
            self._shots[arm] += additional
        step_delta = QueryLedger.difference(
            self.oracle.query_snapshot(), before_step
        )
        for kind in QueryKind:
            self._owned_queries[kind.value] += int(step_delta[kind.value])

        intervals = self._intervals()
        selected, rejected, certificate = self._classify(intervals)
        if certificate is not None:
            self._certificate = certificate
        self._rounds.append(
            BoundaryRound(
                round_index=round_index,
                target_shots_per_arm=target_shots,
                intervals=intervals,
                candidate_selected=selected,
                candidate_rejected=rejected,
                separated=certificate is not None,
            )
        )
        return self.result()

    def run(self) -> BoundaryResult:
        while not self.exhausted:
            self.step()
        return self.result()

    def result(self) -> BoundaryResult:
        intervals = self._intervals()
        selected, rejected, current_certificate = self._classify(intervals)
        certificate = self._certificate or current_certificate
        lower_order = sorted(
            (interval.lower for interval in intervals),
            reverse=True,
        )[self.k - 1]
        upper_order = sorted(
            (interval.upper for interval in intervals),
            reverse=True,
        )[self.k - 1]
        lower_angle = sorted(
            (interval.angular_lower for interval in intervals),
            reverse=True,
        )[self.k - 1]
        upper_angle = sorted(
            (interval.angular_upper for interval in intervals),
            reverse=True,
        )[self.k - 1]
        query_counts = {
            kind.value: int(self._owned_queries[kind.value]) for kind in QueryKind
        }
        query_counts["coherent_total"] = sum(
            query_counts[kind.value]
            for kind in (
                QueryKind.FORWARD,
                QueryKind.INVERSE,
                QueryKind.CONTROLLED_FORWARD,
                QueryKind.CONTROLLED_INVERSE,
            )
        )
        query_counts["classical_total"] = query_counts[
            QueryKind.CLASSICAL_SAMPLE.value
        ]
        query_counts["total"] = (
            query_counts["coherent_total"] + query_counts["classical_total"]
        )
        return BoundaryResult(
            k=self.k,
            intervals=intervals,
            candidate_selected=selected,
            candidate_rejected=rejected,
            order_statistic_mean_interval=(lower_order, upper_order),
            order_statistic_angular_interval=(lower_angle, upper_angle),
            certificate=certificate,
            complete=certificate is not None,
            rounds=tuple(self._rounds),
            resources=self._gates.resources(query_counts),
            minimum_angular_margin=self.minimum_angular_margin,
        )


def qboundary(
    oracle: ExactRewardOracle,
    k: int,
    *,
    confidence: float = 0.05,
    shots_per_round: int = 32,
    max_rounds: int = 8,
    minimum_angular_margin: float = 0.0,
    seed: int | None = None,
    tag: str = "qboundary",
) -> BoundaryResult:
    """Run the executable charged boundary estimator to completion or timeout."""

    return QBoundaryEstimator(
        oracle,
        k,
        confidence=confidence,
        shots_per_round=shots_per_round,
        max_rounds=max_rounds,
        minimum_angular_margin=minimum_angular_margin,
        seed=seed,
        tag=tag,
    ).run()


@dataclass(frozen=True, slots=True)
class GapFlagResult:
    """Result of one reversible phase-marking operation."""

    state: ComplexState
    marked_indices: tuple[int, ...]
    resources: PrimitiveResources
    backend: str = BACKEND_NAME
    claim_status: str = CLAIM_STATUS


class QGapFlag:
    r"""Exact reversible phase marker with one explicitly uncomputed ancilla.

    The implementation applies ``C_f``, ``Z`` on the flag workspace, and
    ``C_f^{-1}``.  For a zero-initialized workspace this realizes
    :math:`|i,0\rangle\mapsto(-1)^{f(i)}|i,0\rangle`.  The complete operation on
    arbitrary workspace input is unitary and self-inverse.
    """

    def __init__(
        self,
        index_dimension: int,
        marked_indices: Sequence[int],
        *,
        valid_indices: Sequence[int] | None = None,
    ) -> None:
        index_dimension = _integer(index_dimension, "index_dimension")
        if index_dimension < 1 or index_dimension & (index_dimension - 1):
            raise ValueError("index_dimension must be a positive power of two")
        marked = tuple(
            sorted({_integer(index, "marked index") for index in marked_indices})
        )
        valid = (
            tuple(range(index_dimension))
            if valid_indices is None
            else tuple(
                sorted({_integer(index, "valid index") for index in valid_indices})
            )
        )
        if any(not 0 <= index < index_dimension for index in valid):
            raise IndexError("a valid index is outside the index register")
        if any(index not in set(valid) for index in marked):
            raise ValueError("every marked index must also be valid")
        self.index_dimension = index_dimension
        self.valid_indices = valid
        self._marked = frozenset(marked)
        self._gates = _GateCounter()
        self._gates.observe_registers(
            int(math.log2(index_dimension)) + 1,
            workspace_qubits=1,
        )

    @property
    def marked_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._marked))

    @property
    def marked_count(self) -> int:
        return len(self._marked)

    def is_marked(
        self,
        index: int,
        *,
        excluded: Sequence[int] = (),
    ) -> bool:
        return index in self._marked and index not in set(excluded)

    def evaluate(
        self,
        index: int,
        *,
        excluded: Sequence[int] = (),
    ) -> bool:
        """Evaluate and measure the compiled predicate with an explicit charge.

        This is the classical-output form of one reversible flag computation.
        It is used to verify measured Grover candidates.  The logical flag-query
        counter is incremented even though the exact simulator already stores
        the predicate needed to execute the reversible gate.
        """

        index = _integer(index, "index")
        if not 0 <= index < self.index_dimension:
            raise IndexError("index is outside the index register")
        self._gates.add("qgapflag_verification_compute")
        self._gates.add("qgapflag_verification_measurement")
        self._gates.phase_oracle_queries += 1
        return self.is_marked(index, excluded=excluded)

    def apply_workspace(
        self,
        state: ComplexState,
        *,
        excluded: Sequence[int] = (),
    ) -> ComplexState:
        """Apply the full compute/phase/uncompute unitary."""

        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != 2 * self.index_dimension:
            raise ValueError(
                "workspace state must have flattened shape "
                f"({self.index_dimension}, 2)"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        excluded_set = set(excluded)
        marked = tuple(index for index in self._marked if index not in excluded_set)
        view = values.copy().reshape(self.index_dimension, 2)

        # Compute f(i) into the flag workspace.
        if marked:
            view[np.asarray(marked, dtype=int)] = view[
                np.asarray(marked, dtype=int)
            ][:, ::-1]
        self._gates.add("qgapflag_compute")
        # Phase kickback on the computed flag.
        view[:, 1] *= -1.0
        self._gates.add("qgapflag_phase_flip")
        # Exact inverse computation restores the workspace.
        if marked:
            view[np.asarray(marked, dtype=int)] = view[
                np.asarray(marked, dtype=int)
            ][:, ::-1]
        self._gates.add("qgapflag_uncompute")
        self._gates.phase_oracle_queries += 1
        return view.reshape(-1)

    def apply_index(
        self,
        state: ComplexState,
        *,
        excluded: Sequence[int] = (),
    ) -> ComplexState:
        """Phase-mark an index state using a zeroed, verified workspace."""

        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.size != self.index_dimension:
            raise ValueError(
                f"expected an index state of length {self.index_dimension}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        workspace = np.zeros((self.index_dimension, 2), dtype=np.complex128)
        workspace[:, 0] = values
        output = self.apply_workspace(workspace.reshape(-1), excluded=excluded)
        shaped = output.reshape(self.index_dimension, 2)
        residual = float(np.linalg.norm(shaped[:, 1]))
        self._gates.max_uncompute_residual = max(
            self._gates.max_uncompute_residual,
            residual,
        )
        if residual > 1e-10:
            raise RuntimeError("QGapFlag failed to uncompute its workspace")
        return shaped[:, 0].copy()

    def resources(self) -> PrimitiveResources:
        return self._gates.resources()


def qgap_flag(
    state: ComplexState,
    marked_indices: Sequence[int],
    *,
    index_dimension: int | None = None,
    valid_indices: Sequence[int] | None = None,
) -> GapFlagResult:
    """Convenience wrapper for one workspace-clean phase-marking call."""

    values = np.asarray(state, dtype=np.complex128)
    dimension = values.size if index_dimension is None else index_dimension
    flag = QGapFlag(
        dimension,
        marked_indices,
        valid_indices=valid_indices,
    )
    output = flag.apply_index(values)
    return GapFlagResult(
        state=output,
        marked_indices=flag.marked_indices,
        resources=flag.resources(),
    )


@dataclass(frozen=True, slots=True)
class BatchExtractResult:
    """Unique, verified outputs from repeated exact-state Grover searches."""

    outputs: tuple[int, ...]
    complete: bool
    verified: bool
    attempts: int
    expected_count: int
    strategy: str
    resources: PrimitiveResources
    failure_reason: str | None = None
    backend: str = BACKEND_NAME
    claim_status: str = CLAIM_STATUS


def _uniform_state(index_dimension: int, valid_indices: Sequence[int]) -> ComplexState:
    state = np.zeros(index_dimension, dtype=np.complex128)
    state[np.asarray(valid_indices, dtype=int)] = 1.0 / math.sqrt(len(valid_indices))
    return state


def _diffuse_about_uniform(
    state: ComplexState,
    valid_indices: Sequence[int],
) -> ComplexState:
    uniform = _uniform_state(state.size, valid_indices)
    return 2.0 * uniform * np.vdot(uniform, state) - state


def _sample_index(state: ComplexState, rng: np.random.Generator) -> int:
    probabilities = np.abs(state) ** 2
    probabilities /= float(np.sum(probabilities))
    return int(rng.choice(state.size, p=probabilities))


def qbatch_extract(
    flag: QGapFlag,
    expected_count: int | None = None,
    *,
    strategy: str = "known",
    seed: int | None = None,
    max_attempts_per_output: int = 24,
    initial_outputs: Sequence[int] = (),
) -> BatchExtractResult:
    """Enumerate marked indices with known-count Grover or the BBHT schedule."""

    if strategy not in {"known", "bbht"}:
        raise ValueError("strategy must be 'known' or 'bbht'")
    expected = (
        flag.marked_count
        if expected_count is None
        else _integer(expected_count, "expected_count")
    )
    if not 0 <= expected <= flag.marked_count:
        raise ValueError("expected_count must lie between zero and marked_count")
    max_attempts_per_output = _integer(
        max_attempts_per_output,
        "max_attempts_per_output",
    )
    if max_attempts_per_output <= 0:
        raise ValueError("max_attempts_per_output must be positive")

    rng = np.random.default_rng(seed)
    gates = _GateCounter()
    gates.observe_registers(
        int(math.log2(flag.index_dimension)) + 1,
        workspace_qubits=1,
    )
    flag_before = flag.resources()
    outputs: list[int] = []
    for raw_index in initial_outputs:
        index = _integer(raw_index, "initial output")
        if not flag.evaluate(index):
            raise ValueError("an initial output is not marked")
        if index not in outputs:
            outputs.append(index)
    if len(outputs) > expected:
        raise ValueError("initial_outputs exceeds expected_count")

    attempts = 0
    failure_reason: str | None = None
    domain_size = len(flag.valid_indices)

    while len(outputs) < expected:
        remaining = expected - len(outputs)
        if remaining == 0:
            failure_reason = "no_unseen_marked_index"
            break

        found = False
        bbht_bound = 1.0
        for _ in range(max_attempts_per_output):
            attempts += 1
            if strategy == "known":
                theta = math.asin(math.sqrt(remaining / domain_size))
                iterations = max(0, int(round(math.pi / (4.0 * theta) - 0.5)))
            else:
                upper = max(1, int(math.ceil(bbht_bound)))
                iterations = int(rng.integers(0, upper))
                bbht_bound = min(
                    (6.0 / 5.0) * bbht_bound,
                    math.sqrt(domain_size),
                )

            state = _uniform_state(flag.index_dimension, flag.valid_indices)
            gates.add("uniform_state_preparation")
            for _ in range(iterations):
                state = flag.apply_index(state, excluded=outputs)
                state = _diffuse_about_uniform(state, flag.valid_indices)
                gates.add("grover_diffusion")
            candidate = _sample_index(state, rng)
            gates.add("index_measurement")
            if flag.evaluate(candidate, excluded=outputs):
                outputs.append(candidate)
                found = True
                break
        if not found:
            failure_reason = "attempt_budget_exhausted"
            break

    ordered = tuple(sorted(outputs))
    verified = len(ordered) == len(set(ordered)) and all(
        flag.evaluate(index) for index in ordered
    )
    flag_after = flag.resources()
    flag_gate_delta = {
        name: int(flag_after.gate_counts.get(name, 0))
        - int(flag_before.gate_counts.get(name, 0))
        for name in set(flag_before.gate_counts) | set(flag_after.gate_counts)
    }
    gates.counts.update(flag_gate_delta)
    gates.depth += flag_after.depth - flag_before.depth
    gates.phase_oracle_queries += (
        flag_after.phase_oracle_queries - flag_before.phase_oracle_queries
    )
    gates.max_uncompute_residual = max(
        gates.max_uncompute_residual,
        flag_after.uncompute_residual,
    )
    complete = verified and len(ordered) == expected
    return BatchExtractResult(
        outputs=ordered,
        complete=complete,
        verified=verified,
        attempts=attempts,
        expected_count=expected,
        strategy=strategy,
        resources=gates.resources(),
        failure_reason=None if complete else failure_reason,
    )


@dataclass(frozen=True, slots=True)
class OrientationCertificate:
    """State of one selected/rejected-complement output certificate."""

    orientation: str
    expected_count: int
    extracted: tuple[int, ...]
    boundary: BoundaryCertificate | None
    verified: bool
    complete: bool
    margin: float | None


@dataclass(frozen=True, slots=True)
class DovetailResult:
    """Current resumable state of the two-orientation controller."""

    selected: tuple[int, ...]
    rejected: tuple[int, ...]
    complete: bool
    winning_orientation: str | None
    steps: int
    certificates: tuple[OrientationCertificate, ...]
    resources: PrimitiveResources
    status: str
    backend: str = BACKEND_NAME
    claim_status: str = CLAIM_STATUS


@dataclass(slots=True)
class _OrientationState:
    name: str
    estimator: QBoundaryEstimator
    expected_count: int
    extracted: tuple[int, ...] = ()
    verified: bool = False
    complete: bool = False
    batch_resumes: int = 0


class DovetailTopKController:
    """Resumable fair scheduler for selected and rejected-complement searches.

    The boundary certificate explicitly contains the classically identified
    selected/rejected sets; one of those known sets is compiled into ``QGapFlag``.
    A branch is successful only after (1) a simultaneous boundary certificate,
    (2) unique Grover enumeration of the requested side, and (3) verification
    of every output against the compiled reversible marker.  Partial empirical
    rankings are never returned as a completed Top-k answer.
    """

    def __init__(
        self,
        oracle: ExactRewardOracle,
        k: int,
        *,
        confidence: float = 0.05,
        shots_per_round: int = 32,
        max_boundary_rounds: int = 8,
        batch_strategy: str = "known",
        seed: int | None = None,
        max_batch_resumes: int = 4,
        max_attempts_per_output: int = 24,
    ) -> None:
        k = _integer(k, "k")
        max_batch_resumes = _integer(max_batch_resumes, "max_batch_resumes")
        max_attempts_per_output = _integer(
            max_attempts_per_output,
            "max_attempts_per_output",
        )
        if not 1 <= k <= oracle.n_arms:
            raise ValueError("k must be in {1, ..., number of arms}")
        if batch_strategy not in {"known", "bbht"}:
            raise ValueError("batch_strategy must be 'known' or 'bbht'")
        if max_batch_resumes <= 0 or max_attempts_per_output <= 0:
            raise ValueError("batch retry budgets must be positive")
        self.oracle = oracle
        self.k = k
        self.batch_strategy = batch_strategy
        self.max_batch_resumes = max_batch_resumes
        self.max_attempts_per_output = max_attempts_per_output
        self._rng = np.random.default_rng(seed)
        selected_seed = int(self._rng.integers(0, 2**32))
        rejected_seed = int(self._rng.integers(0, 2**32))
        self._states = (
            _OrientationState(
                name="selected",
                estimator=QBoundaryEstimator(
                    oracle,
                    k,
                    confidence=confidence / 2.0,
                    shots_per_round=shots_per_round,
                    max_rounds=max_boundary_rounds,
                    seed=selected_seed,
                    tag="dovetail_selected_boundary",
                ),
                expected_count=k,
            ),
            _OrientationState(
                name="rejected_complement",
                estimator=QBoundaryEstimator(
                    oracle,
                    k,
                    confidence=confidence / 2.0,
                    shots_per_round=shots_per_round,
                    max_rounds=max_boundary_rounds,
                    seed=rejected_seed,
                    tag="dovetail_rejected_boundary",
                ),
                expected_count=oracle.n_arms - k,
            ),
        )
        self._next_orientation = 0
        self._winner: str | None = None
        self._steps = 0
        self._before_queries = oracle.query_snapshot()
        self._batch_resources: list[PrimitiveResources] = []

    def _can_advance(self, state: _OrientationState) -> bool:
        if state.complete:
            return False
        boundary = state.estimator.result()
        if not boundary.complete:
            return not state.estimator.exhausted
        return state.batch_resumes < self.max_batch_resumes

    def _advance(self, state: _OrientationState) -> None:
        boundary = state.estimator.result()
        if not boundary.complete:
            state.estimator.step()
            return
        if boundary.certificate is None:
            return

        targets = (
            boundary.certificate.selected
            if state.name == "selected"
            else boundary.certificate.rejected
        )
        flag = QGapFlag(
            self.oracle.index_dimension,
            targets,
            valid_indices=range(self.oracle.n_arms),
        )
        batch_seed = int(self._rng.integers(0, 2**32))
        batch = qbatch_extract(
            flag,
            state.expected_count,
            strategy=self.batch_strategy,
            seed=batch_seed,
            max_attempts_per_output=self.max_attempts_per_output,
            initial_outputs=state.extracted,
        )
        state.batch_resumes += 1
        state.extracted = batch.outputs
        state.verified = batch.verified
        self._batch_resources.append(batch.resources)
        state.complete = (
            batch.complete
            and batch.verified
            and boundary.certificate.complete
            and len(batch.outputs) == state.expected_count
        )
        if state.complete:
            self._winner = state.name

    def step(self) -> DovetailResult:
        """Advance exactly one available orientation and return current state."""

        if self._winner is not None:
            return self.result()
        for offset in range(2):
            index = (self._next_orientation + offset) % 2
            state = self._states[index]
            if self._can_advance(state):
                self._advance(state)
                self._next_orientation = (index + 1) % 2
                self._steps += 1
                break
        return self.result()

    def run(self, max_steps: int | None = None) -> DovetailResult:
        """Run an additional step budget; call again later to resume."""

        if max_steps is not None:
            max_steps = _integer(max_steps, "max_steps")
            if max_steps < 0:
                raise ValueError("max_steps cannot be negative")
        budget = (
            2
            * (
                self._states[0].estimator.max_rounds
                + self.max_batch_resumes
            )
            if max_steps is None
            else max_steps
        )
        for _ in range(budget):
            if self._winner is not None or not any(
                self._can_advance(state) for state in self._states
            ):
                break
            self.step()
        return self.result()

    # An explicit spelling makes resumability visible in experiment code.
    resume = run

    def result(self) -> DovetailResult:
        certificates: list[OrientationCertificate] = []
        for state in self._states:
            boundary = state.estimator.result().certificate
            certificates.append(
                OrientationCertificate(
                    orientation=state.name,
                    expected_count=state.expected_count,
                    extracted=state.extracted,
                    boundary=boundary,
                    verified=state.verified,
                    complete=state.complete,
                    margin=None if boundary is None else boundary.angular_margin,
                )
            )

        selected: tuple[int, ...] = ()
        rejected: tuple[int, ...] = ()
        if self._winner is not None:
            winner = next(
                state for state in self._states if state.name == self._winner
            )
            boundary = winner.estimator.result().certificate
            if boundary is None or not winner.complete:
                raise RuntimeError("winner lacks a complete certificate")
            selected = boundary.selected
            rejected = boundary.rejected

        gates = _GateCounter()
        for state in self._states:
            gates.merge(state.estimator.result().resources)
        for resources in self._batch_resources:
            gates.merge(resources)
        query_counts = QueryLedger.difference(
            self.oracle.query_snapshot(),
            self._before_queries,
        )
        resources = gates.resources(query_counts)
        if self._winner is not None:
            status = "complete_certificate"
        elif any(self._can_advance(state) for state in self._states):
            status = "paused_resumable"
        else:
            status = "incomplete_budgets_exhausted"
        return DovetailResult(
            selected=selected,
            rejected=rejected,
            complete=self._winner is not None,
            winning_orientation=self._winner,
            steps=self._steps,
            certificates=tuple(certificates),
            resources=resources,
            status=status,
        )
