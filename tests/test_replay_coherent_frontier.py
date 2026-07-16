from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qgapselect.replay_coherent_frontier import (
    CLAIM_SCOPE,
    SCHEDULE_LOADING_MODEL,
    ReplayFrontierSchedule,
    ReplayPreservingCoherentFrontier,
)


def _schedule() -> ReplayFrontierSchedule:
    return ReplayFrontierSchedule(
        n_arms=4,
        active_indices_by_level=(
            (0, 1, 2, 3),
            (1, 2, 3),
            (2, 3),
            (2,),
        ),
        output_births_by_level=((0,), (1,), ()),
    )


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_schedule_enforces_nested_frontier_and_unique_output_births() -> None:
    schedule = _schedule()

    assert schedule.level_count == 3
    assert schedule.active_indices_by_level[0] == (0, 1, 2, 3)
    assert len(schedule.fingerprint) == 64

    with pytest.raises(ValueError, match="nested"):
        ReplayFrontierSchedule(
            3,
            ((0, 1, 2), (0, 1), (0, 2)),
            ((2,), ()),
        )
    with pytest.raises(ValueError, match="exclude output births"):
        ReplayFrontierSchedule(
            3,
            ((0, 1, 2), (0, 1), (0,)),
            ((1,), ()),
        )


def test_prefix_transcripts_are_replayable_and_encode_stop_events() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule(), prefix_levels=3)

    selected_early = frontier.transcript(0)
    selected_late = frontier.transcript(1)
    unresolved = frontier.transcript(2)
    rejected = frontier.transcript(3)

    assert selected_early.events == ("select", "inactive", "inactive")
    assert selected_early.history_mask == 0b001
    assert selected_early.stop_code == 1
    assert selected_early.output_bit == 1
    assert selected_late.history_mask == 0b011
    assert selected_late.stop_code == 2
    assert unresolved.events == ("continue", "continue", "continue")
    assert unresolved.stop_code == 0
    assert unresolved.work_code == 0b001
    assert rejected.events == ("continue", "continue", "reject")
    assert rejected.stop_code == 3
    assert rejected.output_bit == 0
    assert all(frontier.verify_transcript(item) for item in frontier.transcripts())

    tampered = replace(selected_early, stop_code=2)
    assert frontier.verify_transcript(tampered) is False


def test_scheduled_prefix_leaves_later_decisions_unresolved() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule(), prefix_levels=1)

    assert frontier.transcript(0).events == ("select",)
    assert frontier.transcript(1).events == ("continue",)
    assert frontier.transcript(1).stop_code == 0
    assert frontier.transcript(1).work_code == 0b001
    assert frontier.history_dimension == 2
    assert frontier.stop_dimension == 2


def test_compute_relation_is_an_involution_on_arbitrary_workspace() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule())
    state = _random_state(frontier.statevector_dimension, 71)

    restored = frontier.apply_compute(frontier.apply_compute(state))

    assert np.allclose(restored, state, atol=1e-12)


def test_durable_output_survives_strict_transient_uncompute() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule())
    state = frontier.uniform_index_state()

    result = frontier.apply(state)
    view = result.state.reshape(frontier.shape)

    assert view[0, 0, 0, 1, 0] == pytest.approx(0.5)
    assert view[1, 0, 0, 1, 0] == pytest.approx(0.5)
    assert view[2, 0, 0, 0, 0] == pytest.approx(0.5)
    assert view[3, 0, 0, 0, 0] == pytest.approx(0.5)
    assert np.count_nonzero(view[:, 1:, :, :, :]) == 0
    assert np.count_nonzero(view[:, :, 1:, :, :]) == 0
    assert np.count_nonzero(view[:, :, :, :, 1:]) == 0
    assert result.durable_output_probability == pytest.approx(0.5)
    assert result.resources.cleanup.passed
    assert result.resources.cleanup.expected_durable_output_residual_l2 == pytest.approx(0.0)
    assert result.resources.cleanup.transient_nonzero_probability == pytest.approx(0.0)
    assert result.invariants.passed
    assert result.invariants.stop_is_first_terminal_event
    assert result.invariants.stopped_branches_remain_inactive
    assert result.status == "code_sanity_passed_theorem_blocked"
    assert result.quantum_advantage_claimable is False


def test_durable_copy_is_xor_and_preserves_an_initial_one() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule())
    state = frontier.uniform_index_state(output_bit=1)

    result = frontier.apply(state)
    view = result.state.reshape(frontier.shape)

    assert view[0, 0, 0, 0, 0] == pytest.approx(0.5)
    assert view[1, 0, 0, 0, 0] == pytest.approx(0.5)
    assert view[2, 0, 0, 1, 0] == pytest.approx(0.5)
    assert view[3, 0, 0, 1, 0] == pytest.approx(0.5)
    assert result.resources.cleanup.passed


def test_trace_is_replayable_and_tampering_fails_closed() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule())
    trace = frontier.execution_trace()

    assert [event.operation for event in trace] == [
        "compute_scheduled_prefix",
        "copy_durable_output_from_work_selected_bit",
        "uncompute_scheduled_prefix_by_replay",
    ]
    assert frontier.verify_execution_trace(trace)
    tampered = (replace(trace[0], charged_schedule_cells=0), *trace[1:])
    assert frontier.verify_execution_trace(tampered) is False


def test_resource_ledger_charges_schedule_and_never_assumes_qram() -> None:
    frontier = ReplayPreservingCoherentFrontier(_schedule())
    result = frontier.apply(frontier.uniform_index_state())
    resources = result.resources

    expected_cells = 2 * 3 * _schedule().n_arms * _schedule().level_count
    assert resources.query_counts["compiled_schedule_cell_accesses"] == expected_cells
    assert resources.query_counts["qram_queries"] == 0
    assert resources.query_counts["reward_oracle_queries"] == 0
    assert resources.qram_assumed is False
    assert resources.schedule_loading_model == SCHEDULE_LOADING_MODEL
    assert resources.query_count_semantics == "logical_charges_not_physical_oracle_equivalence"
    assert resources.depth_semantics.endswith("upper_bound")
    assert resources.gate_counts["charged_schedule_cell_selects"] == expected_cells
    assert resources.gate_counts["durable_output_cnot"] == 1
    assert resources.depth == sum(resources.gate_counts.values())
    assert resources.qubits == sum(
        dimension.bit_length() - 1
        for dimension in resources.register_dimensions.values()
    )
    assert resources.claim_scope == CLAIM_SCOPE
    assert "frontier_schedule_is_a_supplied_public_fixture_not_coherently_discovered" in (
        result.blockers
    )


def test_inputs_and_clean_workspace_are_strict() -> None:
    schedule = _schedule()
    with pytest.raises(ValueError, match="exceeds"):
        ReplayPreservingCoherentFrontier(schedule, prefix_levels=4)
    with pytest.raises(ValueError, match="statevector"):
        ReplayPreservingCoherentFrontier(schedule, max_statevector_dimension=1)

    frontier = ReplayPreservingCoherentFrontier(schedule)
    dirty = np.zeros(frontier.shape, dtype=np.complex128)
    dirty[0, 1, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="must start clean"):
        frontier.apply(dirty.reshape(-1))
