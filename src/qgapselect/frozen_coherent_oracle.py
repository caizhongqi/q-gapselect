"""Blind coherent-oracle emulation derived from a frozen empirical tensor.

The trusted harness constructs this capability from a
:class:`~qgapselect.attack_oracles.FrozenSourceFixture` and gives only the
completed oracle to an algorithm.  Each arm amplitude is the empirical success
frequency in that candidate's already-frozen Bernoulli stream.  Grover
experiments are then sampled from the canonical analytic measurement law.

This module does not execute a quantum circuit, access hardware, or emulate an
LLM coherently.  Its deliberately narrow claim scope is exported as
``CLAIM_SCOPE`` and attached to every capability descriptor.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .attack_oracles import FrozenSourceFixture
from .oracles import CanonicalBernoulliOracleSimulator, QuerySnapshot

CLAIM_SCOPE = "empirical_tensor_coherent_oracle_emulation_no_hardware_claim"
BACKEND = "canonical_analytic_measurement_law_from_frozen_empirical_tensor"


def _optional_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("seed must be an integer or None, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError("seed must be an integer or None") from error


@runtime_checkable
class FrozenCoherentOracleProtocol(Protocol):
    """Minimum algorithm-side capability consumed by QGapSelect/analytic IAE."""

    claim_scope: str

    @property
    def n_arms(self) -> int: ...

    def query_snapshot(self) -> QuerySnapshot: ...

    def run_grover_experiment(
        self,
        arm: int,
        grover_power: int,
        shots: int,
        *,
        controlled: bool = False,
        tag: str | None = None,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class FrozenCoherentOracleDescriptor:
    """Public construction commitment with no rewards or latent means."""

    manifest_hash: str
    candidate_ids: tuple[str, ...]
    stream_lengths: tuple[int, ...]
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    hardware_claimable: bool = False

    def __post_init__(self) -> None:
        if not self.manifest_hash:
            raise ValueError("manifest_hash cannot be empty")
        if not self.candidate_ids:
            raise ValueError("candidate_ids cannot be empty")
        if len(self.candidate_ids) != len(self.stream_lengths):
            raise ValueError("candidate IDs and stream lengths must align")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if any(length <= 0 for length in self.stream_lengths):
            raise ValueError("stream lengths must be positive")

    @property
    def stream_length_by_candidate(self) -> Mapping[str, int]:
        return MappingProxyType(
            dict(zip(self.candidate_ids, self.stream_lengths, strict=True))
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "candidate_ids": list(self.candidate_ids),
            "stream_lengths": list(self.stream_lengths),
            "backend": self.backend,
            "claim_scope": self.claim_scope,
            "hardware_claimable": self.hardware_claimable,
        }


class FrozenEmpiricalCoherentOracle(CanonicalBernoulliOracleSimulator):
    """Algorithm capability for a frozen tensor's empirical arm frequencies.

    Subclassing the canonical simulator is intentional: the existing analytic
    IAE threshold baseline validates that exact runtime type family.  The
    constructed object stores only the canonical simulator's private rotation
    parameters plus public commitments.  It does not retain the fixture,
    reward/cost tensors, evaluator, configured means, or generator seed.
    """

    __slots__ = ("__candidate_ids", "__descriptor", "__manifest_hash")

    claim_scope = CLAIM_SCOPE
    backend = BACKEND
    hardware_claimable = False

    def __init__(
        self,
        fixture: FrozenSourceFixture,
        *,
        measurement_seed: int | None = None,
    ) -> None:
        if not isinstance(fixture, FrozenSourceFixture):
            raise TypeError("fixture must be a FrozenSourceFixture")
        measurement_seed = _optional_seed(measurement_seed)

        # This trusted-construction block is the only place where the frozen
        # reward tensor is inspected.  No fixture/tensor reference is retained.
        reward_streams = fixture.tensor.reward_streams
        empirical_frequencies = tuple(
            math.fsum(stream) / len(stream) for stream in reward_streams
        )
        candidate_ids = fixture.tensor.graph.candidate_ids
        stream_lengths = tuple(len(stream) for stream in reward_streams)
        manifest_hash = fixture.manifest_hash

        super().__init__(empirical_frequencies, seed=measurement_seed)
        self.__candidate_ids = candidate_ids
        self.__manifest_hash = manifest_hash
        self.__descriptor = FrozenCoherentOracleDescriptor(
            manifest_hash=manifest_hash,
            candidate_ids=candidate_ids,
            stream_lengths=stream_lengths,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self.__candidate_ids

    @property
    def manifest_hash(self) -> str:
        return self.__manifest_hash

    @property
    def descriptor(self) -> FrozenCoherentOracleDescriptor:
        return self.__descriptor

    def candidate_id(self, arm: int) -> str:
        if isinstance(arm, bool):
            raise TypeError("arm must be an integer, not bool")
        try:
            index = int(operator.index(arm))
        except TypeError as error:
            raise TypeError("arm must be an integer") from error
        if not 0 <= index < self.n_arms:
            raise IndexError(f"arm {index} is outside [0, {self.n_arms})")
        return self.__candidate_ids[index]

    def arm_index(self, candidate_id: str) -> int:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TypeError("candidate_id must be a non-empty string")
        try:
            return self.__candidate_ids.index(candidate_id)
        except ValueError:
            raise KeyError(candidate_id) from None


def build_frozen_empirical_coherent_oracle(
    fixture: FrozenSourceFixture,
    *,
    measurement_seed: int | None = None,
) -> FrozenEmpiricalCoherentOracle:
    """Trusted-runner factory for the blind algorithm capability."""

    return FrozenEmpiricalCoherentOracle(
        fixture,
        measurement_seed=measurement_seed,
    )


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "FrozenCoherentOracleDescriptor",
    "FrozenCoherentOracleProtocol",
    "FrozenEmpiricalCoherentOracle",
    "build_frozen_empirical_coherent_oracle",
]
