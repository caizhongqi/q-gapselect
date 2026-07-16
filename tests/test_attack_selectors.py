from __future__ import annotations

from collections import defaultdict

import pytest

from qgapselect.attack_selectors import (
    CostAwareRacingSelector,
    GCGSGreedyGraphSelector,
    InternalCLUCBStyleSelector,
    RandomSelector,
    SuccessiveHalvingSelector,
    UniformTopKSelector,
)


class _ReplaySampler:
    def __init__(
        self,
        rewards: dict[str, tuple[float, ...]],
        costs: dict[str, float] | None = None,
    ) -> None:
        self._rewards = rewards
        self._costs = costs or {candidate: 1.0 for candidate in rewards}
        self.calls: defaultdict[str, int] = defaultdict(int)

    @property
    def means(self) -> object:
        raise AssertionError("a selector read hidden means")

    def sample(self, candidate_id: str) -> tuple[float, float]:
        offset = self.calls[candidate_id]
        sequence = self._rewards[candidate_id]
        self.calls[candidate_id] += 1
        return sequence[offset % len(sequence)], self._costs[candidate_id]


def _constant_sampler(costs: dict[str, float] | None = None) -> _ReplaySampler:
    return _ReplaySampler(
        {
            "a": (0.1,),
            "b": (0.9,),
            "c": (0.6,),
            "d": (0.2,),
        },
        costs,
    )


@pytest.mark.parametrize(
    "selector,adjacency",
    [
        (UniformTopKSelector(seed=7), None),
        (SuccessiveHalvingSelector(seed=7), None),
        (CostAwareRacingSelector(seed=7), None),
        (
            GCGSGreedyGraphSelector(seed=7),
            {"a": ("b",), "b": ("a", "c"), "c": ("b", "d"), "d": ("c",)},
        ),
    ],
)
def test_selectors_find_empirical_top_two_without_hidden_means(
    selector: object,
    adjacency: dict[str, tuple[str, ...]] | None,
) -> None:
    sampler = _constant_sampler()
    result = selector.select(  # type: ignore[attr-defined]
        candidates=("a", "b", "c", "d"),
        sampler=sampler,
        k=2,
        query_budget=48,
        cost_budget=48.0,
        sample_cost_upper_bound=1.0,
        adjacency=adjacency,
    )

    assert result.selected == ("b", "c")
    assert result.complete
    assert result.resources.within_budget
    assert result.resources.queries_used == len(result.trace)


def test_random_selector_is_seeded_and_never_queries_sampler() -> None:
    first_sampler = _constant_sampler()
    second_sampler = _constant_sampler()
    arguments = {
        "candidates": ("a", "b", "c", "d"),
        "k": 2,
        "query_budget": 9,
        "cost_budget": 9.0,
    }
    first = RandomSelector(seed=19).select(sampler=first_sampler, **arguments)
    second = RandomSelector(seed=19).select(sampler=second_sampler, **arguments)

    assert first.ranking == second.ranking
    assert first.selected == second.selected
    assert first.resources.queries_used == 0
    assert sum(first_sampler.calls.values()) == 0


def test_uniform_selector_enforces_query_and_reserved_cost_budgets() -> None:
    sampler = _constant_sampler({candidate: 0.6 for candidate in "abcd"})
    result = UniformTopKSelector().select(
        candidates=("a", "b", "c", "d"),
        sampler=sampler,
        k=2,
        query_budget=100,
        cost_budget=3.0,
        sample_cost_upper_bound=0.75,
    )

    assert result.resources.queries_used == 4
    assert result.resources.cost_used == pytest.approx(2.4)
    assert result.resources.remaining_cost == pytest.approx(0.6)
    assert not result.resources.can_afford_another_sample
    assert result.stop_reason == "cost_budget_exhausted"


def test_cost_bound_is_a_checked_sampler_contract() -> None:
    sampler = _constant_sampler({candidate: 1.1 for candidate in "abcd"})
    with pytest.raises(ValueError, match="sample_cost_upper_bound"):
        UniformTopKSelector().select(
            candidates=("a", "b", "c", "d"),
            sampler=sampler,
            k=1,
            query_budget=4,
            cost_budget=4.0,
            sample_cost_upper_bound=1.0,
        )


def test_boolean_attack_event_is_a_valid_reward() -> None:
    sampler = _ReplaySampler({"failed": (False,), "succeeded": (True,)})
    result = UniformTopKSelector().select(
        candidates=("failed", "succeeded"),
        sampler=sampler,
        k=1,
        query_budget=2,
        cost_budget=2.0,
    )

    assert result.selected == ("succeeded",)
    assert result.estimates == {"failed": 0.0, "succeeded": 1.0}


def test_ties_break_by_frozen_candidate_order() -> None:
    sampler = _ReplaySampler({candidate: (0.5,) for candidate in "wxyz"})
    result = UniformTopKSelector(seed=999).select(
        candidates=("z", "x", "w", "y"),
        sampler=sampler,
        k=3,
        query_budget=8,
        cost_budget=8.0,
    )

    assert result.ranking == ("z", "x", "w", "y")
    assert result.selected == ("z", "x", "w")


def test_uniform_partial_pass_never_exceeds_query_budget() -> None:
    result = UniformTopKSelector().select(
        candidates=("a", "b", "c", "d"),
        sampler=_constant_sampler(),
        k=2,
        query_budget=6,
        cost_budget=20.0,
    )

    assert result.resources.queries_used == 6
    assert dict(result.sample_counts) == {"a": 2, "b": 2, "c": 1, "d": 1}
    assert result.resources.cost_used == 6.0


def test_successive_halving_eliminates_low_empirical_arms() -> None:
    result = SuccessiveHalvingSelector().select(
        candidates=("a", "b", "c", "d"),
        sampler=_constant_sampler(),
        k=1,
        query_budget=24,
        cost_budget=24.0,
    )

    assert result.selected == ("b",)
    assert set(result.eliminated_candidates) == {"a", "c", "d"}
    assert result.rounds_completed == 2


def test_cost_aware_racing_allocates_more_queries_to_a_cheap_uncertain_arm() -> None:
    sampler = _ReplaySampler(
        {"cheap": (0.4, 0.6), "expensive": (0.4, 0.6)},
        {"cheap": 0.1, "expensive": 1.0},
    )
    result = CostAwareRacingSelector(confidence=0.05).select(
        candidates=("cheap", "expensive"),
        sampler=sampler,
        k=1,
        query_budget=12,
        cost_budget=12.0,
        sample_cost_upper_bound=1.0,
    )

    assert result.sample_counts["cheap"] > result.sample_counts["expensive"]
    assert result.resources.within_budget


def test_cost_aware_racing_partial_initialization_does_not_starve_unseen_arms() -> None:
    result = CostAwareRacingSelector().select(
        candidates=("a", "b", "c", "d"),
        sampler=_constant_sampler(),
        k=1,
        query_budget=3,
        cost_budget=3.0,
    )

    assert dict(result.sample_counts) == {"a": 1, "b": 1, "c": 1, "d": 0}


def test_internal_clucb_style_finds_top_k_without_hidden_means() -> None:
    result = InternalCLUCBStyleSelector(confidence=0.05, seed=9).select(
        candidates=("a", "b", "c", "d"),
        sampler=_constant_sampler(),
        k=2,
        query_budget=80,
        cost_budget=80.0,
    )

    assert result.selector_id == "internal_clucb_style_topk_v1"
    assert result.selected == ("b", "c")
    assert result.resources.within_budget
    assert result.sample_counts["c"] > result.sample_counts["a"]
    assert result.sample_counts["c"] > result.sample_counts["b"]


def test_internal_clucb_style_obeys_strict_query_and_cost_budgets() -> None:
    sampler = _constant_sampler({candidate: 0.4 for candidate in "abcd"})
    result = InternalCLUCBStyleSelector().select(
        candidates=("a", "b", "c", "d"),
        sampler=sampler,
        k=1,
        query_budget=7,
        cost_budget=3.0,
        sample_cost_upper_bound=0.5,
    )

    assert result.resources.queries_used == 7
    assert result.resources.cost_used == pytest.approx(2.8)
    assert result.resources.within_budget
    assert len(result.trace) == 7


def test_internal_clucb_style_uses_deterministic_boundary_ties() -> None:
    first = InternalCLUCBStyleSelector(seed=1).select(
        candidates=("z", "x", "w"),
        sampler=_ReplaySampler({candidate: (0.5,) for candidate in "wxz"}),
        k=1,
        query_budget=9,
        cost_budget=9.0,
    )
    second = InternalCLUCBStyleSelector(seed=999).select(
        candidates=("z", "x", "w"),
        sampler=_ReplaySampler({candidate: (0.5,) for candidate in "wxz"}),
        k=1,
        query_budget=9,
        cost_budget=9.0,
    )

    assert first.trace == second.trace
    assert first.selected == second.selected == ("z",)


def test_internal_clucb_style_all_candidates_is_query_free() -> None:
    sampler = _ReplaySampler({"a": (0.0,), "b": (1.0,)})
    result = InternalCLUCBStyleSelector().select(
        candidates=("a", "b"),
        sampler=sampler,
        k=2,
        query_budget=10,
        cost_budget=10.0,
    )

    assert result.selected == ("a", "b")
    assert result.resources.queries_used == 0
    assert result.stop_reason == "selection_converged"


def test_gcgs_uses_seeded_restarts_to_cover_disconnected_components() -> None:
    adjacency = {"a": ("b",), "b": ("a",), "c": ("d",), "d": ("c",)}
    first = GCGSGreedyGraphSelector(seed=11).select(
        candidates=("a", "b", "c", "d"),
        sampler=_constant_sampler(),
        k=2,
        query_budget=12,
        cost_budget=12.0,
        adjacency=adjacency,
    )
    second = GCGSGreedyGraphSelector(seed=11).select(
        candidates=("a", "b", "c", "d"),
        sampler=_constant_sampler(),
        k=2,
        query_budget=12,
        cost_budget=12.0,
        adjacency=adjacency,
    )

    assert first.trace == second.trace
    assert all(first.sample_counts[candidate] > 0 for candidate in "abcd")
    assert first.selected == ("b", "c")


@pytest.mark.parametrize(
    "adjacency,match",
    [
        (None, "adjacency is required"),
        ({"outside": ()}, "key outside"),
        ({"a": ("outside",)}, "neighbor outside"),
    ],
)
def test_gcgs_rejects_invalid_graphs(
    adjacency: dict[str, tuple[str, ...]] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        GCGSGreedyGraphSelector().select(
            candidates=("a", "b"),
            sampler=_ReplaySampler({"a": (0.0,), "b": (1.0,)}),
            k=1,
            query_budget=2,
            cost_budget=2.0,
            adjacency=adjacency,
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"candidates": (), "k": 1}, ValueError),
        ({"candidates": ("a", "a"), "k": 1}, ValueError),
        ({"candidates": ("a",), "k": 0}, ValueError),
        ({"candidates": ("a",), "k": 2}, ValueError),
        ({"candidates": ("a",), "k": 1, "query_budget": -1}, ValueError),
        ({"candidates": ("a",), "k": 1, "cost_budget": -1.0}, ValueError),
    ],
)
def test_common_input_validation(kwargs: dict[str, object], error: type[Exception]) -> None:
    arguments: dict[str, object] = {
        "candidates": ("a",),
        "sampler": _ReplaySampler({"a": (0.5,)}),
        "k": 1,
        "query_budget": 1,
        "cost_budget": 1.0,
    }
    arguments.update(kwargs)
    with pytest.raises(error):
        UniformTopKSelector().select(**arguments)  # type: ignore[arg-type]


def test_zero_budget_returns_deterministic_unsampled_selection() -> None:
    result = UniformTopKSelector().select(
        candidates=("c", "a", "b"),
        sampler=_ReplaySampler({candidate: (0.5,) for candidate in "abc"}),
        k=2,
        query_budget=0,
        cost_budget=0.0,
    )

    assert result.selected == ("c", "a")
    assert result.estimates == {"c": None, "a": None, "b": None}
    assert result.resources.queries_used == 0
    assert result.stop_reason == "query_budget_exhausted"
