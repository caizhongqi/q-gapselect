"""Exact-count frozen Bernoulli fixtures for purely synthetic experiments.

For candidate ``i`` with configured mean ``mu_i`` and table size ``M``, the
generator creates exactly ``round(M * mu_i)`` successes.  It then shuffles that
binary table with a stable candidate-specific seed.  Thus finite-table noise in
the empirical mean is controlled explicitly, while query order and the ordering
of unrelated candidates cannot alter an arm's frozen stream.

This module constructs synthetic source fixtures only.  It does not connect to
an LLM, model API, or quantum hardware.
"""

from __future__ import annotations

import hashlib
import math
import operator
import random
from collections.abc import Mapping

from .attack_oracles import (
    FrozenCandidateGraph,
    FrozenSourceFixture,
    freeze_source_streams,
)

GENERATOR = "exact_count_then_seeded_shuffle"


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _mean(value: object, candidate_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"mean for candidate {candidate_id!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(
            f"mean for candidate {candidate_id!r} must be finite and in [0, 1]"
        )
    return result


def _candidate_seed(seed: int, candidate_id: str) -> int:
    material = (
        f"qgapselect.exact-count-fixture.v1\0{seed}\0{candidate_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _exact_count_stream(
    mean: float,
    *,
    table_size: int,
    seed: int,
    candidate_id: str,
) -> tuple[int, ...]:
    successes = int(round(table_size * mean))
    stream = [1] * successes + [0] * (table_size - successes)
    random.Random(_candidate_seed(seed, candidate_id)).shuffle(stream)
    return tuple(stream)


def generate_exact_count_fixture(
    candidate_means: Mapping[str, float],
    *,
    table_size: int,
    seed: int,
    graph: FrozenCandidateGraph | None = None,
) -> FrozenSourceFixture:
    """Build a reproducible exact-count frozen empirical source fixture.

    ``candidate_means`` is runner-only ground truth.  If ``graph`` is supplied,
    its candidate order controls tensor alignment and its candidate IDs must
    match the mapping exactly.  Without a graph, insertion order determines the
    public candidate order; individual streams remain order-invariant because
    each uses its own derived seed.
    """

    if not isinstance(candidate_means, Mapping) or not candidate_means:
        raise TypeError("candidate_means must be a non-empty mapping")
    table_size = _integer(table_size, "table_size", minimum=1)
    seed = _integer(seed, "seed", minimum=0)

    for candidate_id in candidate_means:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TypeError("candidate IDs must be non-empty strings")
    if graph is None:
        graph = FrozenCandidateGraph.from_ids(tuple(candidate_means))
    elif not isinstance(graph, FrozenCandidateGraph):
        raise TypeError("graph must be a FrozenCandidateGraph or None")
    if set(candidate_means) != set(graph.candidate_ids):
        missing = sorted(set(graph.candidate_ids) - set(candidate_means))
        extra = sorted(set(candidate_means) - set(graph.candidate_ids))
        raise ValueError(
            "candidate_means keys must exactly match graph candidate IDs; "
            f"missing={missing}, extra={extra}"
        )

    normalized_means = {
        candidate_id: _mean(candidate_means[candidate_id], candidate_id)
        for candidate_id in graph.candidate_ids
    }
    reward_streams = {
        candidate_id: _exact_count_stream(
            normalized_means[candidate_id],
            table_size=table_size,
            seed=seed,
            candidate_id=candidate_id,
        )
        for candidate_id in graph.candidate_ids
    }
    cost_streams = {
        candidate_id: (1.0,) * table_size for candidate_id in graph.candidate_ids
    }
    return freeze_source_streams(
        graph,
        reward_streams,
        cost_streams,
        configured_means=normalized_means,
        metadata={
            "generator": GENERATOR,
            "table_size": str(table_size),
            "seed": str(seed),
            "success_count_rule": "round(table_size * configured_mean)",
        },
    )


__all__ = ["GENERATOR", "generate_exact_count_fixture"]
