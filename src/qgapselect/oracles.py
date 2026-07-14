"""Canonical stochastic-oracle simulators and auditable query accounting."""

from __future__ import annotations

import math
import operator
import random
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


def _integer_argument(value: object, name: str) -> int:
    """Accept exact integer-like values without truncating floats or strings."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


class QueryKind(str, Enum):
    """Kinds of canonical oracle access charged by :class:`QueryLedger`."""

    FORWARD = "forward"
    INVERSE = "inverse"
    CONTROLLED_FORWARD = "controlled_forward"
    CONTROLLED_INVERSE = "controlled_inverse"
    CLASSICAL_SAMPLE = "classical_sample"


@dataclass(frozen=True, slots=True)
class QuerySnapshot:
    """Immutable aggregate of query counts."""

    counts: Mapping[str, int]
    by_arm: Mapping[int, Mapping[str, int]]
    by_tag: Mapping[str, Mapping[str, int]]

    @property
    def coherent_total(self) -> int:
        return sum(
            self.counts.get(kind.value, 0)
            for kind in (
                QueryKind.FORWARD,
                QueryKind.INVERSE,
                QueryKind.CONTROLLED_FORWARD,
                QueryKind.CONTROLLED_INVERSE,
            )
        )

    @property
    def classical_total(self) -> int:
        return self.counts.get(QueryKind.CLASSICAL_SAMPLE.value, 0)

    @property
    def total(self) -> int:
        return self.coherent_total + self.classical_total

    def flat(self) -> dict[str, int]:
        result = {kind.value: int(self.counts.get(kind.value, 0)) for kind in QueryKind}
        result["coherent_total"] = self.coherent_total
        result["classical_total"] = self.classical_total
        result["total"] = self.total
        return result


class QueryLedger:
    """Thread-safe ledger for canonical oracle calls.

    The ledger aggregates counts instead of storing an event per shot so large
    experiments do not create an accidental memory bottleneck.
    """

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._by_arm: defaultdict[int, Counter[str]] = defaultdict(Counter)
        self._by_tag: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._lock = threading.RLock()

    def record(
        self,
        kind: QueryKind,
        count: int = 1,
        *,
        arm: int | None = None,
        tag: str | None = None,
    ) -> None:
        if not isinstance(kind, QueryKind):
            raise TypeError("kind must be a QueryKind")
        count = _integer_argument(count, "count")
        if count < 0:
            raise ValueError("query count cannot be negative")
        if count == 0:
            return
        with self._lock:
            self._counts[kind.value] += int(count)
            if arm is not None:
                arm = _integer_argument(arm, "arm")
                self._by_arm[arm][kind.value] += count
            if tag is not None:
                self._by_tag[str(tag)][kind.value] += int(count)

    def snapshot(self) -> QuerySnapshot:
        with self._lock:
            return QuerySnapshot(
                counts=MappingProxyType(dict(self._counts)),
                by_arm=MappingProxyType(
                    {
                        arm: MappingProxyType(dict(counts))
                        for arm, counts in self._by_arm.items()
                    }
                ),
                by_tag=MappingProxyType(
                    {
                        tag: MappingProxyType(dict(counts))
                        for tag, counts in self._by_tag.items()
                    }
                ),
            )

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._by_arm.clear()
            self._by_tag.clear()

    @staticmethod
    def difference(after: QuerySnapshot, before: QuerySnapshot) -> dict[str, int]:
        """Return a flat non-negative count difference between snapshots."""

        keys = {kind.value for kind in QueryKind}
        delta = {
            key: int(after.counts.get(key, 0) - before.counts.get(key, 0))
            for key in keys
        }
        if any(value < 0 for value in delta.values()):
            raise ValueError("the 'after' snapshot predates the 'before' snapshot")
        delta["coherent_total"] = sum(
            delta[kind.value]
            for kind in (
                QueryKind.FORWARD,
                QueryKind.INVERSE,
                QueryKind.CONTROLLED_FORWARD,
                QueryKind.CONTROLLED_INVERSE,
            )
        )
        delta["classical_total"] = delta[QueryKind.CLASSICAL_SAMPLE.value]
        delta["total"] = delta["coherent_total"] + delta["classical_total"]
        return delta


class CanonicalBernoulliOracleSimulator:
    r"""Canonical garbage-free Bernoulli rotation oracle.

    For arm ``i`` with mean :math:`\mu_i`, the complete reward-qubit unitary is

    .. math::

       O_\mu = \sum_i |i\rangle\!\langle i|\otimes
       \begin{pmatrix}
       \sqrt{1-\mu_i}&-\sqrt{\mu_i}\\
       \sqrt{\mu_i}& \sqrt{1-\mu_i}
       \end{pmatrix}.

    On the reward qubit this is exactly :math:`R_y(2\theta_i)` with
    :math:`\theta_i=\arcsin\sqrt{\mu_i}`.  No arm-dependent phase or garbage
    register is exposed.

    This class is an analytic measurement *simulator* of that access model.  It never
    represents a physical speed-up; it only produces measurement distributions
    and records the logical calls that the corresponding circuit makes.  Its
    public algorithm-facing methods return measurements or immutable query
    snapshots, never amplitudes, matrices, or hidden means.  Exact construction
    diagnostics live in :mod:`qgapselect.diagnostics` and require the caller to
    supply a mean explicitly.  It accepts one classical arm index per experiment;
    it is not the coherent index-register gate needed by the proposed batch
    algorithm.
    """

    __slots__ = ("__ledger", "__means", "__rng")

    def __init__(
        self,
        means: Sequence[float],
        *,
        seed: int | None = None,
    ) -> None:
        values = tuple(float(mu) for mu in means)
        if not values:
            raise ValueError("the oracle must contain at least one arm")
        if any(not math.isfinite(mu) or not 0.0 <= mu <= 1.0 for mu in values):
            raise ValueError("all Bernoulli means must be finite and in [0, 1]")
        # Construction belongs to the trusted benchmark harness.  Algorithms
        # receive the completed oracle object and no reference to ``values``.
        self.__means = values
        self.__rng = random.Random(seed)
        self.__ledger = QueryLedger()

    @property
    def n_arms(self) -> int:
        return len(self.__means)

    def query_snapshot(self) -> QuerySnapshot:
        """Return an immutable copy of the audit ledger.

        The mutable ledger is intentionally not exposed: an algorithm may
        inspect costs but cannot reset them or manufacture entries through the
        oracle interface.
        """

        return self.__ledger.snapshot()

    def _validate_arm(self, arm: int) -> int:
        arm = _integer_argument(arm, "arm")
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm {arm} is outside [0, {self.n_arms})")
        return arm

    def sample(
        self,
        arm: int,
        shots: int = 1,
        *,
        tag: str | None = None,
    ) -> int:
        """Make basis-state classical queries and return the number of successes."""

        arm = self._validate_arm(arm)
        shots = _integer_argument(shots, "shots")
        if shots <= 0:
            raise ValueError("shots must be positive")
        self.__ledger.record(QueryKind.CLASSICAL_SAMPLE, shots, arm=arm, tag=tag)
        probability = self.__means[arm]
        return sum(self.__rng.random() < probability for _ in range(shots))

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool = False,
        tag: str | None = None,
    ) -> int:
        """Analytically sample measurements of ``Q**m A|0>``.

        Each shot is charged ``m + 1`` forward and ``m`` inverse calls.  The
        reward-marking reflection and zero-state reflection are not calls to
        the Bernoulli oracle.  Runtime of this classical method is simulator
        runtime and must be reported separately from the logical ledger.  If
        ``controlled`` is true, the control is assumed to be active and only
        the query category changes; this method does not simulate a control in
        superposition or implement a controlled circuit.
        """

        arm = self._validate_arm(arm)
        grover_power = _integer_argument(grover_power, "grover_power")
        shots = _integer_argument(shots, "shots")
        if not isinstance(controlled, bool):
            raise TypeError("controlled must be bool")
        if grover_power < 0:
            raise ValueError("grover_power cannot be negative")
        if shots <= 0:
            raise ValueError("shots must be positive")

        forward_kind = (
            QueryKind.CONTROLLED_FORWARD if controlled else QueryKind.FORWARD
        )
        inverse_kind = (
            QueryKind.CONTROLLED_INVERSE if controlled else QueryKind.INVERSE
        )
        self.__ledger.record(
            forward_kind,
            shots * (grover_power + 1),
            arm=arm,
            tag=tag,
        )
        self.__ledger.record(
            inverse_kind,
            shots * grover_power,
            arm=arm,
            tag=tag,
        )
        theta = math.asin(math.sqrt(self.__means[arm]))
        probability = math.sin((2 * grover_power + 1) * theta) ** 2
        return sum(self.__rng.random() < probability for _ in range(shots))
