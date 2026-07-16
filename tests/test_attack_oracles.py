from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qgapselect.attack_oracles import (
    BlindSourceRewardOracle,
    CandidateEdge,
    FrozenCandidateGraph,
    SourceBudgetExhaustedError,
    SourceCandidate,
    SourceOracleBudget,
    SourceQueryStatus,
    SourceStreamExhaustedError,
    freeze_source_streams,
    generate_bernoulli_landscape,
)


def _fixture():
    graph = FrozenCandidateGraph(
        candidates=(
            SourceCandidate("a", payload_hash="sha256:a", family="lexical"),
            SourceCandidate("b", payload_hash="sha256:b", family="semantic"),
        ),
        edges=(CandidateEdge("a", "b", "rewrite"),),
    )
    return freeze_source_streams(
        graph,
        reward_streams={"a": [1, 0, 1], "b": [0, 1, 1]},
        cost_streams={"a": [0.1, 0.2, 0.3], "b": [1.0, 1.5, 2.0]},
        configured_means={"a": 0.75, "b": 0.25},
        metadata={"split": "source-only"},
    )


def test_graph_and_stream_tensor_are_frozen_copies() -> None:
    candidates = [SourceCandidate("a"), SourceCandidate("b")]
    rewards = {"a": [1, 0], "b": [0, 1]}
    costs = {"a": [1.0, 1.0], "b": [2.0, 2.0]}
    graph = FrozenCandidateGraph(candidates, [CandidateEdge("a", "b")])
    fixture = freeze_source_streams(graph, rewards, costs)

    candidates.append(SourceCandidate("c"))
    rewards["a"][0] = 0
    costs["b"][0] = 99.0

    oracle = fixture.open_oracle(SourceOracleBudget(max_queries=2, max_cost=10.0))
    assert oracle.candidate_ids == ("a", "b")
    assert oracle.graph.successors("a") == ("b",)
    assert oracle.query("a").reward == 1
    assert oracle.query("b").cost == 2.0
    with pytest.raises(FrozenInstanceError):
        fixture.tensor.graph = FrozenCandidateGraph.from_ids(("x",))


def test_selector_api_hides_means_and_unconsumed_streams() -> None:
    fixture = _fixture()
    oracle = fixture.open_oracle(SourceOracleBudget(max_queries=4, max_cost=10.0))

    assert isinstance(oracle, BlindSourceRewardOracle)
    assert not hasattr(oracle, "means")
    assert not hasattr(oracle, "configured_means")
    assert not hasattr(oracle, "frozen_means")
    assert not hasattr(oracle, "reward_streams")
    assert not hasattr(oracle, "evaluator")
    assert set(oracle.snapshot().cursors) == {"a", "b"}


def test_candidate_cursors_are_independent_and_records_are_global() -> None:
    oracle = _fixture().open_oracle(
        SourceOracleBudget(max_queries=4, max_cost=10.0)
    )

    first_a = oracle.query("a", tag="round-1")
    first_b = oracle.query("b", tag="round-1")
    second_a = oracle.query("a", tag="round-2")
    snapshot = oracle.snapshot()

    assert (first_a.reward, first_b.reward, second_a.reward) == (1, 0, 0)
    assert (first_a.candidate_query_index, second_a.candidate_query_index) == (0, 1)
    assert [item.global_query_index for item in (first_a, first_b, second_a)] == [1, 2, 3]
    assert dict(snapshot.cursors) == {"a": 2, "b": 1}
    assert snapshot.queries_used == 3
    assert snapshot.cost_used == pytest.approx(1.3)
    assert [record.tag for record in snapshot.records] == [
        "round-1",
        "round-1",
        "round-2",
    ]
    assert all(record.status is SourceQueryStatus.ACCEPTED for record in snapshot.records)


def test_query_budget_rejection_is_recorded_without_consumption() -> None:
    oracle = _fixture().open_oracle(
        SourceOracleBudget(max_queries=1, max_cost=100.0)
    )
    oracle.query("a")

    with pytest.raises(SourceBudgetExhaustedError) as error:
        oracle.query("b", tag="denied")

    assert error.value.status is SourceQueryStatus.QUERY_BUDGET_EXHAUSTED
    snapshot = oracle.snapshot()
    assert snapshot.queries_used == 1
    assert snapshot.cost_used == 0.1
    assert dict(snapshot.cursors) == {"a": 1, "b": 0}
    assert snapshot.attempts == 2
    assert snapshot.records[-1].status is SourceQueryStatus.QUERY_BUDGET_EXHAUSTED
    assert snapshot.records[-1].reward is None
    assert snapshot.records[-1].cost is None


def test_decimal_cost_budget_is_strict_and_rejection_does_not_advance_cursor() -> None:
    fixture = _fixture()
    oracle = fixture.open_oracle(SourceOracleBudget(max_queries=10, max_cost=0.3))

    assert oracle.query("a").cost == 0.1
    assert oracle.query("a").cost == 0.2
    with pytest.raises(SourceBudgetExhaustedError) as error:
        oracle.query("a")
    assert error.value.status is SourceQueryStatus.COST_BUDGET_EXHAUSTED
    snapshot = oracle.snapshot()
    assert snapshot.queries_used == 2
    assert snapshot.cost_used == 0.3
    assert snapshot.cursors["a"] == 2
    assert snapshot.records[-1].cost is None

    fresh = fixture.open_oracle(SourceOracleBudget(max_queries=1, max_cost=0.3))
    assert fresh.query("a").candidate_query_index == 0


def test_stream_exhaustion_is_distinct_from_budget_exhaustion() -> None:
    graph = FrozenCandidateGraph.from_ids(("only",))
    fixture = freeze_source_streams(
        graph,
        reward_streams={"only": [1]},
        cost_streams={"only": [1.0]},
    )
    oracle = fixture.open_oracle(SourceOracleBudget(max_queries=2, max_cost=2.0))
    oracle.query("only")

    with pytest.raises(SourceStreamExhaustedError):
        oracle.query("only")

    snapshot = oracle.snapshot()
    assert snapshot.queries_used == 1
    assert snapshot.cursors["only"] == 1
    assert snapshot.records[-1].status is SourceQueryStatus.STREAM_EXHAUSTED


def test_snapshot_and_manifest_commit_to_the_exact_fixture() -> None:
    fixture = _fixture()
    oracle = fixture.open_oracle(SourceOracleBudget(max_queries=3, max_cost=4.0))
    snapshot = oracle.snapshot()

    assert len(fixture.manifest_hash) == 64
    assert snapshot.manifest_hash == fixture.tensor.manifest_hash
    assert snapshot.manifest_hash == fixture.evaluator.manifest_hash
    with pytest.raises(TypeError):
        snapshot.cursors["a"] = 3  # type: ignore[index]

    changed = freeze_source_streams(
        fixture.tensor.graph,
        reward_streams={"a": [0, 0, 1], "b": [0, 1, 1]},
        cost_streams={"a": [0.1, 0.2, 0.3], "b": [1.0, 1.5, 2.0]},
        configured_means={"a": 0.75, "b": 0.25},
        metadata={"split": "source-only"},
    )
    assert changed.manifest_hash != fixture.manifest_hash


def test_evaluator_is_runner_only_and_scores_top_k_outputs() -> None:
    evaluator = _fixture().evaluator

    assert evaluator.top_k(1) == ("a",)
    report = evaluator.evaluate(("a",), k=1)
    assert report.exact_match
    assert report.precision_at_k == 1.0
    assert report.recall_at_k == 1.0
    assert report.top_k_regret == 0.0

    wrong = evaluator.evaluate(("b",), k=1)
    assert not wrong.exact_match
    assert wrong.hits == 0
    assert wrong.top_k_regret == 0.5
    with pytest.raises(ValueError, match="more than k"):
        evaluator.evaluate(("a", "b"), k=1)
    with pytest.raises(ValueError, match="unknown"):
        evaluator.evaluate(("missing",), k=1)


def test_bernoulli_generator_is_deterministic_and_candidate_order_independent() -> None:
    first = generate_bernoulli_landscape(
        {"a": 0.2, "b": 0.8},
        samples_per_candidate=32,
        seed=17,
        candidate_costs={"a": 1.0, "b": 3.0},
        cost_jitter=0.2,
        edges=(CandidateEdge("a", "b"),),
    )
    second = generate_bernoulli_landscape(
        {"a": 0.2, "b": 0.8},
        samples_per_candidate=32,
        seed=17,
        candidate_costs={"a": 1.0, "b": 3.0},
        cost_jitter=0.2,
        edges=(CandidateEdge("a", "b"),),
    )
    reversed_order = generate_bernoulli_landscape(
        {"b": 0.8, "a": 0.2},
        samples_per_candidate=32,
        seed=17,
        candidate_costs={"a": 1.0, "b": 3.0},
        cost_jitter=0.2,
    )

    assert first.manifest_hash == second.manifest_hash
    first_oracle = first.open_oracle(SourceOracleBudget(2, 10.0))
    reversed_oracle = reversed_order.open_oracle(SourceOracleBudget(2, 10.0))
    assert first_oracle.query("a").reward == reversed_oracle.query("a").reward
    assert first_oracle.query("b").reward == reversed_oracle.query("b").reward
    assert first.evaluator.expected_mean_by_candidate == {"a": 0.2, "b": 0.8}


@pytest.mark.parametrize(
    ("rewards", "costs", "message"),
    [
        ({"a": [0.5], "b": [1]}, {"a": [1.0], "b": [1.0]}, "Bernoulli"),
        ({"a": [1], "b": [0]}, {"a": [0.0], "b": [1.0]}, "positive"),
        ({"a": [1], "b": [0]}, {"a": [1.0], "b": [1.0, 2.0]}, "lengths"),
    ],
)
def test_invalid_frozen_streams_are_rejected(rewards, costs, message: str) -> None:
    graph = FrozenCandidateGraph.from_ids(("a", "b"))
    with pytest.raises(ValueError, match=message):
        freeze_source_streams(graph, rewards, costs)


def test_generator_validates_landscape_and_heterogeneous_cost_parameters() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        generate_bernoulli_landscape(
            {"a": 1.1}, samples_per_candidate=2, seed=1
        )
    with pytest.raises(ValueError, match="exactly match"):
        generate_bernoulli_landscape(
            {"a": 0.5},
            samples_per_candidate=2,
            seed=1,
            candidate_costs={"b": 1.0},
        )
    with pytest.raises(ValueError, match="cost_jitter"):
        generate_bernoulli_landscape(
            {"a": 0.5}, samples_per_candidate=2, seed=1, cost_jitter=1.0
        )
