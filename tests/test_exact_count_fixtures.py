from __future__ import annotations

import math

import pytest

from qgapselect.attack_oracles import (
    CandidateEdge,
    FrozenCandidateGraph,
    SourceCandidate,
)
from qgapselect.exact_count_fixtures import (
    GENERATOR,
    generate_exact_count_fixture,
)


def _stream_by_candidate(fixture) -> dict[str, tuple[int, ...]]:
    return {
        candidate_id: fixture.tensor.reward_streams[index]
        for index, candidate_id in enumerate(fixture.tensor.graph.candidate_ids)
    }


def _cost_by_candidate(fixture) -> dict[str, tuple[float, ...]]:
    return {
        candidate_id: fixture.tensor.cost_streams[index]
        for index, candidate_id in enumerate(fixture.tensor.graph.candidate_ids)
    }


def test_exact_success_counts_and_unit_costs() -> None:
    means = {"zero": 0.0, "low": 0.26, "half": 0.5, "one": 1.0}
    table_size = 10
    fixture = generate_exact_count_fixture(means, table_size=table_size, seed=7)

    streams = _stream_by_candidate(fixture)
    costs = _cost_by_candidate(fixture)
    for candidate_id, mean in means.items():
        assert len(streams[candidate_id]) == table_size
        assert sum(streams[candidate_id]) == round(table_size * mean)
        assert costs[candidate_id] == (1.0,) * table_size


def test_runner_evaluator_reports_configured_and_exact_empirical_means() -> None:
    fixture = generate_exact_count_fixture(
        {"a": 0.26, "b": 0.74},
        table_size=10,
        seed=11,
    )

    assert fixture.evaluator.expected_mean_by_candidate == {"a": 0.26, "b": 0.74}
    assert fixture.evaluator.frozen_mean_by_candidate == {"a": 0.3, "b": 0.7}
    assert fixture.evaluator.top_k(1) == ("b",)


def test_same_inputs_reproduce_tensor_and_manifest() -> None:
    arguments = {
        "candidate_means": {"a": 0.2, "b": 0.55, "c": 0.8},
        "table_size": 40,
        "seed": 123,
    }

    first = generate_exact_count_fixture(**arguments)
    second = generate_exact_count_fixture(**arguments)

    assert first.tensor.reward_streams == second.tensor.reward_streams
    assert first.tensor.cost_streams == second.tensor.cost_streams
    assert first.manifest_hash == second.manifest_hash
    assert dict(first.tensor.metadata)["generator"] == GENERATOR
    assert dict(first.tensor.metadata)["generator"] == (
        "exact_count_then_seeded_shuffle"
    )


def test_candidate_stream_is_invariant_to_mapping_order() -> None:
    first = generate_exact_count_fixture(
        {"a": 0.35, "b": 0.65, "c": 0.5},
        table_size=20,
        seed=91,
    )
    reordered = generate_exact_count_fixture(
        {"c": 0.5, "a": 0.35, "b": 0.65},
        table_size=20,
        seed=91,
    )

    assert _stream_by_candidate(first) == _stream_by_candidate(reordered)


def test_fixed_graph_makes_manifest_invariant_to_means_mapping_order() -> None:
    graph = FrozenCandidateGraph(
        candidates=(
            SourceCandidate("a", payload_hash="sha256:a", family="synthetic"),
            SourceCandidate("b", payload_hash="sha256:b", family="synthetic"),
        ),
        edges=(CandidateEdge("a", "b", "neighbor"),),
    )
    first = generate_exact_count_fixture(
        {"a": 0.3, "b": 0.7}, table_size=20, seed=4, graph=graph
    )
    reordered = generate_exact_count_fixture(
        {"b": 0.7, "a": 0.3}, table_size=20, seed=4, graph=graph
    )

    assert first.tensor.graph is graph
    assert first.tensor.reward_streams == reordered.tensor.reward_streams
    assert first.manifest_hash == reordered.manifest_hash


def test_seed_changes_positions_but_not_exact_count() -> None:
    first = generate_exact_count_fixture({"a": 0.4}, table_size=40, seed=1)
    second = generate_exact_count_fixture({"a": 0.4}, table_size=40, seed=2)
    first_stream = first.tensor.reward_streams[0]
    second_stream = second.tensor.reward_streams[0]

    assert sum(first_stream) == sum(second_stream) == 16
    assert first_stream != second_stream
    assert first.manifest_hash != second.manifest_hash


@pytest.mark.parametrize(
    "means",
    [
        {},
        {"a": -0.01},
        {"a": 1.01},
        {"a": math.nan},
        {"a": math.inf},
        {"a": True},
        {"": 0.5},
    ],
)
def test_invalid_candidate_means_are_rejected(means: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_exact_count_fixture(means, table_size=10, seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("table_size", "seed"),
    [(0, 0), (-1, 0), (10, -1), (True, 0), (10, True), (1.5, 0)],
)
def test_invalid_table_size_or_seed_is_rejected(table_size: object, seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_exact_count_fixture(
            {"a": 0.5},
            table_size=table_size,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )


def test_graph_candidate_ids_must_match_means_exactly() -> None:
    graph = FrozenCandidateGraph.from_ids(("a", "b"))

    with pytest.raises(ValueError, match="exactly match"):
        generate_exact_count_fixture(
            {"a": 0.5, "extra": 0.1},
            table_size=8,
            seed=2,
            graph=graph,
        )
    with pytest.raises(TypeError, match="FrozenCandidateGraph"):
        generate_exact_count_fixture(
            {"a": 0.5},
            table_size=8,
            seed=2,
            graph=object(),  # type: ignore[arg-type]
        )
