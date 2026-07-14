"""Non-executable contracts for coherent reward-oracle implementations.

The analytic simulator in :mod:`qgapselect.oracles` intentionally does not
implement these protocols.  A paper-grade coherent Q-GapSelect backend must
append gates acting on an index *register* in superposition and must support
inverse/controlled composition with complete workspace uncomputation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class OracleModel(str, Enum):
    """Reward access models tracked separately by the research protocol."""

    CANONICAL_ROTATION = "canonical_rotation"
    NATURAL_PURIFICATION = "natural_purification"


@dataclass(frozen=True, slots=True)
class CoherentOracleContract:
    """Auditable metadata required before a backend can claim coherent access."""

    n_arms: int
    index_qubits: int
    workspace_qubits: int
    model: OracleModel
    full_unitary_specified: bool
    inverse_available: bool
    controlled_available: bool
    reward_projector_available: bool
    claim_status: str

    def __post_init__(self) -> None:
        if self.n_arms < 1:
            raise ValueError("n_arms must be positive")
        if self.index_qubits < math.ceil(math.log2(self.n_arms)):
            raise ValueError("index_qubits cannot encode every arm")
        if self.workspace_qubits < 0:
            raise ValueError("workspace_qubits cannot be negative")
        if self.model is OracleModel.NATURAL_PURIFICATION and self.workspace_qubits == 0:
            raise ValueError("a natural purification must declare its workspace")


@runtime_checkable
class ReversibleCircuitBuilder(Protocol):
    """Small adapter boundary for Qiskit, PennyLane, or a custom circuit IR."""

    def append_named_gate(
        self,
        name: str,
        *,
        targets: tuple[object, ...],
        controls: tuple[object, ...] = (),
    ) -> None: ...


@runtime_checkable
class CoherentRewardGate(Protocol):
    """Gate-level interface required by the proposed coherent algorithm."""

    @property
    def contract(self) -> CoherentOracleContract: ...

    def append_forward(
        self,
        circuit: ReversibleCircuitBuilder,
        *,
        index_register: object,
        workspace_register: object,
        reward_qubit: object,
        controls: tuple[object, ...] = (),
    ) -> None: ...

    def append_inverse(
        self,
        circuit: ReversibleCircuitBuilder,
        *,
        index_register: object,
        workspace_register: object,
        reward_qubit: object,
        controls: tuple[object, ...] = (),
    ) -> None: ...


def canonical_rotation_contract(n_arms: int) -> CoherentOracleContract:
    """Return the required contract for the phase-fixed synthetic model."""

    return CoherentOracleContract(
        n_arms=n_arms,
        index_qubits=max(0, math.ceil(math.log2(n_arms))),
        workspace_qubits=0,
        model=OracleModel.CANONICAL_ROTATION,
        full_unitary_specified=True,
        inverse_available=True,
        controlled_available=True,
        reward_projector_available=True,
        claim_status="interface_contract_not_an_implementation",
    )
