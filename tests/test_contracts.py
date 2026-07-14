from __future__ import annotations

import pytest

from qgapselect.contracts import (
    CoherentOracleContract,
    CoherentRewardGate,
    OracleModel,
    canonical_rotation_contract,
)
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


def test_analytic_measurement_backend_is_not_a_coherent_gate() -> None:
    simulator = CanonicalBernoulliOracleSimulator((0.2, 0.8), seed=4)

    assert not isinstance(simulator, CoherentRewardGate)


def test_canonical_contract_requires_enough_index_qubits() -> None:
    contract = canonical_rotation_contract(9)

    assert contract.index_qubits == 4
    assert contract.workspace_qubits == 0
    assert contract.full_unitary_specified
    assert contract.claim_status == "interface_contract_not_an_implementation"


def test_natural_purification_cannot_hide_an_undeclared_workspace() -> None:
    with pytest.raises(ValueError, match="workspace"):
        CoherentOracleContract(
            n_arms=4,
            index_qubits=2,
            workspace_qubits=0,
            model=OracleModel.NATURAL_PURIFICATION,
            full_unitary_specified=True,
            inverse_available=True,
            controlled_available=True,
            reward_projector_available=True,
            claim_status="extension_proof_obligation",
        )
