"""Core immutable contracts for Q-COLLIDE calibration fixtures.

The module deliberately separates endpoint-local collision instances from
arbitrary pair-oracle instances. The former may use claw/collision structure;
the latter may only use unstructured pair-search bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Side = Literal["A", "B"]


@dataclass(frozen=True)
class EndpointRecord:
    """A fully specified endpoint record available to every compared method."""

    index: int
    side: Side
    signature: tuple[int, ...]
    prefixes: tuple[tuple[int, ...], ...]
    control: tuple[float, ...]
    payload: tuple[float, ...]
    behavior: float
    valid: bool = True
    stage_costs: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if not self.signature:
            raise ValueError("signature must be non-empty")
        if any(cost <= 0.0 for cost in self.stage_costs):
            raise ValueError("stage costs must be strictly positive")
        if self.prefixes and len(self.prefixes) != len(self.stage_costs):
            raise ValueError("prefix and stage-cost lengths must match")

    @property
    def cumulative_stage_costs(self) -> tuple[float, ...]:
        return tuple(float(x) for x in np.cumsum(self.stage_costs))


@dataclass(frozen=True)
class CollisionCriteria:
    """Thresholds defining a conditional neural collision edge."""

    control_epsilon: float
    payload_delta: float
    behavior_gamma: float

    def __post_init__(self) -> None:
        if self.control_epsilon < 0.0:
            raise ValueError("control_epsilon must be non-negative")
        if self.payload_delta < 0.0:
            raise ValueError("payload_delta must be non-negative")
        if self.behavior_gamma < 0.0:
            raise ValueError("behavior_gamma must be non-negative")


@dataclass(frozen=True)
class CollisionInstance:
    """A frozen bipartite Q-COLLIDE instance.

    ``endpoint_local`` is a hard contract. When false, the valid relation is
    supplied only by ``explicit_pair_marks`` and collision/claw algorithms are
    not entitled to infer it from endpoint records.
    """

    name: str
    left: tuple[EndpointRecord, ...]
    right: tuple[EndpointRecord, ...]
    criteria: CollisionCriteria
    endpoint_local: bool
    explicit_pair_marks: frozenset[tuple[int, int]] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.left or not self.right:
            raise ValueError("both endpoint domains must be non-empty")
        if {record.side for record in self.left} != {"A"}:
            raise ValueError("left records must use side='A'")
        if {record.side for record in self.right} != {"B"}:
            raise ValueError("right records must use side='B'")
        if [record.index for record in self.left] != list(range(len(self.left))):
            raise ValueError("left indices must be dense and ordered")
        if [record.index for record in self.right] != list(range(len(self.right))):
            raise ValueError("right indices must be dense and ordered")
        for i, j in self.explicit_pair_marks:
            if not (0 <= i < len(self.left) and 0 <= j < len(self.right)):
                raise ValueError("explicit pair mark is outside the domain")
        if self.endpoint_local and self.explicit_pair_marks:
            raise ValueError("endpoint-local instances must not carry hidden pair marks")
        if not self.endpoint_local and not self.explicit_pair_marks:
            raise ValueError("pair-oracle instances require at least one explicit mark")

    @property
    def n_left(self) -> int:
        return len(self.left)

    @property
    def n_right(self) -> int:
        return len(self.right)
