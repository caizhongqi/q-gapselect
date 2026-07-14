"""Exact-state natural-purification oracle for small research instances.

The canonical one-qubit rotation is not a free replacement for a reversible
generator with work garbage.  This module therefore implements the broader
state-preparation interface directly.  It is intentionally limited to exact
small-state simulation; its memory cost is exponential in the declared qubits.
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
class NaturalArmDistribution:
    """Finite classical randomness and reward map for one oracle arm."""

    probabilities: tuple[float, ...]
    rewards: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        probabilities: Sequence[float],
        rewards: Sequence[int | bool],
    ) -> NaturalArmDistribution:
        return cls(
            tuple(float(value) for value in probabilities),
            tuple(int(value) for value in rewards),
        )

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("an arm distribution needs at least one outcome")
        if len(self.probabilities) != len(self.rewards):
            raise ValueError("probabilities and rewards must have equal length")
        if any(
            not math.isfinite(probability) or probability < 0.0
            for probability in self.probabilities
        ):
            raise ValueError("probabilities must be finite and non-negative")
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        if any(reward not in (0, 1) for reward in self.rewards):
            raise ValueError("rewards must be bits")


@dataclass(frozen=True, slots=True)
class NaturalResourceSnapshot:
    """Resource counters separated from statevector wall-clock work."""

    oracle_queries: Mapping[str, int]
    good_reflections: int
    statevector_dimension: int
    index_qubits: int
    workspace_qubits: int
    reward_qubits: int = 1


class NaturalPurificationStatevectorOracle:
    r"""Block state-preparation oracle with explicit work and reward registers.

    For each arm, the requested first column is

    .. math::

       \sum_w \sqrt{p_i(w)}\lvert w\rangle\lvert R_i(w)\rangle.

    A deterministic Householder completion maps the all-zero local basis state
    to that vector and fixes every other column.  Hence the complete unitary is
    specified, and forward/inverse calls are executable on arbitrary input
    states.  The completion is a simulator convention, not a claim that a real
    reversible generator has unit Householder cost.
    """

    __slots__ = (
        "__blocks",
        "__good_reflections",
        "__ledger",
        "__lock",
        "__rng",
        "_contract",
        "_index_dim",
        "_n_arms",
        "_work_dim",
    )

    def __init__(
        self,
        arms: Sequence[NaturalArmDistribution],
        *,
        seed: int | None = None,
    ) -> None:
        specifications = tuple(arms)
        if not specifications:
            raise ValueError("at least one arm is required")
        self._n_arms = len(specifications)
        self._index_dim = _next_power_of_two(self._n_arms)
        self._work_dim = _next_power_of_two(
            max(2, max(len(arm.probabilities) for arm in specifications))
        )
        local_dimension = 2 * self._work_dim
        zero = np.zeros(local_dimension, dtype=np.complex128)
        zero[0] = 1.0
        blocks: list[ComplexState] = []
        for arm in specifications:
            target = np.zeros(local_dimension, dtype=np.complex128)
            for work, (probability, reward) in enumerate(
                zip(arm.probabilities, arm.rewards, strict=True)
            ):
                target[2 * work + reward] = math.sqrt(probability)
            difference = zero - target
            norm = float(np.linalg.norm(difference))
            if norm <= 1e-14:
                block = np.eye(local_dimension, dtype=np.complex128)
            else:
                direction = difference / norm
                block = np.eye(local_dimension, dtype=np.complex128) - 2.0 * np.outer(
                    direction, direction.conj()
                )
            blocks.append(block)
        self.__blocks = tuple(blocks)
        self.__ledger = QueryLedger()
        self.__good_reflections = 0
        self.__lock = threading.RLock()
        self.__rng = random.Random(seed)
        self._contract = CoherentOracleContract(
            n_arms=self._n_arms,
            index_qubits=int(math.log2(self._index_dim)),
            workspace_qubits=int(math.log2(self._work_dim)),
            model=OracleModel.NATURAL_PURIFICATION,
            full_unitary_specified=True,
            inverse_available=True,
            controlled_available=True,
            reward_projector_available=True,
            claim_status="exact_small_state_householder_simulator",
        )

    @property
    def contract(self) -> CoherentOracleContract:
        return self._contract

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def index_dimension(self) -> int:
        return self._index_dim

    @property
    def workspace_dimension(self) -> int:
        return self._work_dim

    @property
    def statevector_dimension(self) -> int:
        return self._index_dim * self._work_dim * 2

    def query_snapshot(self) -> QuerySnapshot:
        return self.__ledger.snapshot()

    def resource_snapshot(self) -> NaturalResourceSnapshot:
        with self.__lock:
            return NaturalResourceSnapshot(
                oracle_queries=MappingProxyType(
                    dict(self.__ledger.snapshot().flat())
                ),
                good_reflections=self.__good_reflections,
                statevector_dimension=self.statevector_dimension,
                index_qubits=self.contract.index_qubits,
                workspace_qubits=self.contract.workspace_qubits,
            )

    def zero_state(self, *, controlled: bool = False) -> ComplexState:
        """Return the normalized all-zero register state."""

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
        """Prepare a uniform index state with zeroed work and reward registers."""

        if not isinstance(controlled, bool) or not isinstance(active_control, bool):
            raise TypeError("controlled and active_control must be bool")
        selected = tuple(range(self.n_arms)) if indices is None else tuple(indices)
        if not selected:
            raise ValueError("indices cannot be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("indices must be unique")
        if any(not isinstance(index, int) or not 0 <= index < self.n_arms for index in selected):
            raise IndexError("an index is outside the declared arm range")
        local_dimension = 2 * self.workspace_dimension
        shape = (
            (2, self.index_dimension, local_dimension)
            if controlled
            else (self.index_dimension, local_dimension)
        )
        state = np.zeros(shape, dtype=np.complex128)
        amplitude = 1.0 / math.sqrt(len(selected))
        if controlled:
            control = int(active_control)
            state[control, selected, 0] = amplitude
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
        local_dimension = 2 * self.workspace_dimension
        shape = (
            (2, self.index_dimension, local_dimension)
            if controlled
            else (self.index_dimension, local_dimension)
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
        """Apply the complete forward or inverse sampler to an arbitrary state."""

        if not isinstance(inverse, bool) or not isinstance(controlled, bool):
            raise TypeError("inverse and controlled must be bool")
        view = self._validated_view(state, controlled)
        control_slice = view[1] if controlled else view
        for arm, block in enumerate(self.__blocks):
            operator = block.conj().T if inverse else block
            control_slice[arm] = operator @ control_slice[arm]
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
        """Apply the known reflection that flips exactly reward-one basis states."""

        if not isinstance(controlled, bool):
            raise TypeError("controlled must be bool")
        view = self._validated_view(state, controlled)
        if controlled:
            shaped = view[1].reshape(
                self.index_dimension,
                self.workspace_dimension,
                2,
            )
        else:
            shaped = view.reshape(
                self.index_dimension,
                self.workspace_dimension,
                2,
            )
        shaped[..., 1] *= -1.0
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
        if controlled:
            shaped = view.reshape(
                2,
                self.index_dimension,
                self.workspace_dimension,
                2,
            )
        else:
            shaped = view.reshape(
                self.index_dimension,
                self.workspace_dimension,
                2,
            )
        probability = float(np.sum(np.abs(shaped[..., 1]) ** 2))
        return int(self.__rng.random() < min(max(probability, 0.0), 1.0))
