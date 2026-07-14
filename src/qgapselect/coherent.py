"""Exact statevector implementation of the canonical Bernoulli oracle.

This module is deliberately small-scale.  It applies the *complete* block
rotation to arbitrary normalized superpositions, including a genuine control
qubit branch, and records every logical oracle call.  The implementation is a
reference semantics for experiments; it is not an efficient circuit compiler
and it does not establish a query-complexity theorem.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .contracts import CoherentOracleContract, OracleModel
from .oracles import QueryKind, QueryLedger, QuerySnapshot

ComplexState = NDArray[np.complex128]


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True, slots=True)
class CanonicalResourceSnapshot:
    """Logical resources used by an exact canonical-oracle simulation."""

    oracle_queries: Mapping[str, int]
    good_reflections: int
    statevector_dimension: int
    index_qubits: int
    workspace_qubits: int = 0
    reward_qubits: int = 1
    backend: str = "numpy_exact_statevector_small_scale"


class CanonicalRyStatevectorOracle:
    r"""Canonical phase-fixed :math:`R_y` reward oracle on an index register.

    For every valid arm ``i`` the complete local block is

    .. math::

       \begin{pmatrix}
       \sqrt{1-\mu_i} & -\sqrt{\mu_i}\\
       \sqrt{\mu_i} & \sqrt{1-\mu_i}
       \end{pmatrix}.

    Invalid computational-basis indices (which occur when the number of arms
    is not a power of two) are fixed by the identity.  Statevectors use the
    flattened register order ``(index, reward)`` or, for a controlled call,
    ``(control, index, reward)``.  Only the ``control == 1`` branch is acted on.

    Hidden construction parameters are not exposed through an algorithm-facing
    mean, angle, or dense-matrix accessor.  Algorithms must call :meth:`apply`
    or :meth:`reward_experiment`, both of which charge the query ledger.
    """

    __slots__ = (
        "__blocks",
        "__good_reflections",
        "__ledger",
        "__lock",
        "__rng",
        "_contract",
        "_index_dimension",
        "_n_arms",
    )

    def __init__(
        self,
        means: Sequence[float],
        *,
        seed: int | None = None,
    ) -> None:
        values = tuple(float(mean) for mean in means)
        if not values:
            raise ValueError("at least one arm is required")
        if any(
            not math.isfinite(mean) or not 0.0 <= mean <= 1.0
            for mean in values
        ):
            raise ValueError("all Bernoulli means must be finite and in [0, 1]")

        self._n_arms = len(values)
        self._index_dimension = _next_power_of_two(self._n_arms)
        blocks: list[ComplexState] = []
        for mean in values:
            cosine = math.sqrt(1.0 - mean)
            sine = math.sqrt(mean)
            blocks.append(
                np.asarray(
                    ((cosine, -sine), (sine, cosine)),
                    dtype=np.complex128,
                )
            )
        self.__blocks = tuple(blocks)
        self.__ledger = QueryLedger()
        self.__rng = random.Random(seed)
        self.__lock = threading.RLock()
        self.__good_reflections = 0
        self._contract = CoherentOracleContract(
            n_arms=self._n_arms,
            index_qubits=int(math.log2(self._index_dimension)),
            workspace_qubits=0,
            model=OracleModel.CANONICAL_ROTATION,
            full_unitary_specified=True,
            inverse_available=True,
            controlled_available=True,
            reward_projector_available=True,
            claim_status="exact_small_state_implementation_no_complexity_theorem",
        )

    @property
    def contract(self) -> CoherentOracleContract:
        return self._contract

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def index_qubits(self) -> int:
        return self.contract.index_qubits

    @property
    def index_dimension(self) -> int:
        return self._index_dimension

    @property
    def statevector_dimension(self) -> int:
        return 2 * self.index_dimension

    # Compatibility with the canonical analytic backend's terminology.
    @property
    def state_dimension(self) -> int:
        return self.statevector_dimension

    def query_snapshot(self) -> QuerySnapshot:
        return self.__ledger.snapshot()

    def resource_snapshot(self) -> CanonicalResourceSnapshot:
        with self.__lock:
            return CanonicalResourceSnapshot(
                oracle_queries=MappingProxyType(
                    dict(self.__ledger.snapshot().flat())
                ),
                good_reflections=self.__good_reflections,
                statevector_dimension=self.statevector_dimension,
                index_qubits=self.index_qubits,
            )

    def zero_state(self, *, controlled: bool = False) -> ComplexState:
        """Return a normalized all-zero state of the requested register shape."""

        if not isinstance(controlled, bool):
            raise TypeError("controlled must be bool")
        dimension = self.statevector_dimension * (2 if controlled else 1)
        state = np.zeros(dimension, dtype=np.complex128)
        state[0] = 1.0
        return state

    def index_superposition(
        self,
        indices: Sequence[int] | None = None,
        *,
        controlled: bool = False,
        active_control: bool = True,
    ) -> ComplexState:
        """Prepare uniform valid indices with the reward qubit initialized to zero."""

        selected = tuple(range(self.n_arms)) if indices is None else tuple(indices)
        if not selected:
            raise ValueError("indices cannot be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("indices must be unique")
        if any(
            not isinstance(index, int) or not 0 <= index < self.n_arms
            for index in selected
        ):
            raise IndexError("an index is outside the declared arm range")
        if not isinstance(controlled, bool) or not isinstance(active_control, bool):
            raise TypeError("controlled and active_control must be bool")

        shape = (
            (2, self.index_dimension, 2)
            if controlled
            else (self.index_dimension, 2)
        )
        state = np.zeros(shape, dtype=np.complex128)
        amplitude = 1.0 / math.sqrt(len(selected))
        if controlled:
            state[int(active_control), selected, 0] = amplitude
        else:
            state[selected, 0] = amplitude
        return state.reshape(-1)

    def _validated_view(self, state: ComplexState, controlled: bool) -> ComplexState:
        values = np.asarray(state, dtype=np.complex128)
        expected = self.statevector_dimension * (2 if controlled else 1)
        if values.ndim != 1 or values.size != expected:
            raise ValueError(
                f"expected a flat statevector of length {expected}, got {values.shape}"
            )
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-10):
            raise ValueError("statevector must be normalized")
        shape = (
            (2, self.index_dimension, 2)
            if controlled
            else (self.index_dimension, 2)
        )
        return values.copy().reshape(shape)

    def apply(
        self,
        state: ComplexState,
        *,
        inverse: bool = False,
        controlled: bool = False,
        tag: str | None = None,
    ) -> ComplexState:
        """Apply one complete forward/inverse oracle call to an arbitrary state."""

        if not isinstance(inverse, bool) or not isinstance(controlled, bool):
            raise TypeError("inverse and controlled must be bool")
        view = self._validated_view(state, controlled)
        active = view[1] if controlled else view
        for arm, block in enumerate(self.__blocks):
            operator = block.conj().T if inverse else block
            active[arm] = operator @ active[arm]

        kind = {
            (False, False): QueryKind.FORWARD,
            (True, False): QueryKind.INVERSE,
            (False, True): QueryKind.CONTROLLED_FORWARD,
            (True, True): QueryKind.CONTROLLED_INVERSE,
        }[(inverse, controlled)]
        self.__ledger.record(kind, tag=tag)
        return view.reshape(-1)

    def reflect_good(
        self,
        state: ComplexState,
        *,
        controlled: bool = False,
    ) -> ComplexState:
        """Flip every reward-one basis state in the active control branch."""

        if not isinstance(controlled, bool):
            raise TypeError("controlled must be bool")
        view = self._validated_view(state, controlled)
        active = view[1] if controlled else view
        active[..., 1] *= -1.0
        with self.__lock:
            self.__good_reflections += 1
        return view.reshape(-1)

    def measure_reward(
        self,
        state: ComplexState,
        *,
        controlled: bool = False,
    ) -> int:
        """Sample the reward qubit without mutating the supplied statevector."""

        view = self._validated_view(state, controlled)
        probability = float(np.sum(np.abs(view[..., 1]) ** 2))
        with self.__lock:
            return int(self.__rng.random() < min(max(probability, 0.0), 1.0))

    def reward_experiment(
        self,
        arm: int,
        shots: int = 1,
        *,
        tag: str | None = None,
    ) -> int:
        """Run measured one-query experiments for an arm.

        This routine intentionally rebuilds and applies the statevector for
        every shot.  It never computes the success probability from a hidden
        mean, so callers obtain information only through charged experiments.
        """

        if not isinstance(arm, int) or isinstance(arm, bool):
            raise TypeError("arm must be an integer")
        if not 0 <= arm < self.n_arms:
            raise IndexError("arm is outside the declared range")
        if not isinstance(shots, int) or isinstance(shots, bool):
            raise TypeError("shots must be an integer")
        if shots <= 0:
            raise ValueError("shots must be positive")

        successes = 0
        for _ in range(shots):
            prepared = self.index_superposition((arm,))
            output = self.apply(prepared, tag=tag)
            successes += self.measure_reward(output)
        return successes
