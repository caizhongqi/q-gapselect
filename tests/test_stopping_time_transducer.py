from __future__ import annotations

import numpy as np
import pytest

from qgapselect.stopping_time_transducer import (
    CLAIM_STATUS,
    VariableTimeStoppingTransducer,
    precision_bits_from_history_record,
)
from qgapselect.unknown_boundary_history import unknown_boundary_history_record


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_stopping_compute_is_reversible_and_charged() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3, 4],
        boundary_phase=0.5,
    )
    state = _random_state(transducer.statevector_dimension, 101)

    computed = transducer.apply_compute(state)
    restored = transducer.apply_compute(computed)

    assert np.allclose(restored, state, atol=1e-12)
    snapshot = transducer.query_snapshot()
    assert snapshot["stopping_history:compute"] == 2
    assert snapshot["stopping:serial_full_history_cost_per_branch"] == 2 * ((2**3 - 1) + (2**4 - 1))
    assert transducer.resource_snapshot().claim_status == CLAIM_STATUS


def test_stopping_profiles_record_first_output_and_sentinel() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3],
        boundary_phase=0.5,
    )

    profiles = transducer.profiles()
    assert profiles[0].stop_code == transducer.stop_code_sentinel
    assert profiles[1].ever_active is True
    assert profiles[1].output is False
    assert profiles[2].first_output_level == 0
    assert profiles[2].stop_code == 0
    assert profiles[2].stopping_cost == 2**3 - 1

    summary = transducer.summary()
    assert summary.output_count == 1
    assert summary.active_count == 2
    assert summary.unresolved_count == 3
    assert summary.variable_over_serial_per_branch == pytest.approx(1.0)


def test_stopping_phase_oracle_restores_work_registers() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3],
        boundary_phase=0.5,
    )
    state = transducer.uniform_index_state()

    result = transducer.apply_phase(state, phase_on="output")
    view = result.state.reshape(transducer.shape)
    expected = state.reshape(transducer.shape).copy()
    expected[2, 0, 0, 0] *= -1

    assert np.allclose(view, expected, atol=1e-12)
    assert np.count_nonzero(view[:, 1:, :, :]) == 0
    assert np.count_nonzero(view[:, :, 1, :]) == 0
    assert np.count_nonzero(view[:, :, :, 1]) == 0
    assert result.output_probability == pytest.approx(1 / 4)
    assert result.active_probability == pytest.approx(2 / 4)
    assert result.resources.query_counts["stopping_history:phase:output"] == 1


def test_stopping_summary_can_show_variable_time_savings() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.63, 0.86],
        [3, 8],
        boundary_phase=0.5,
    )

    summary = transducer.summary()

    assert summary.serial_full_history_cost_per_branch == (2**3 - 1) + (2**8 - 1)
    assert summary.output_count >= 2
    assert summary.variable_over_serial_per_branch < 1.0
    assert summary.all_branch_rms_search_proxy > summary.coherent_branch_rms_cost


def test_precision_bits_from_history_record_matches_levels() -> None:
    record = unknown_boundary_history_record(8)
    bits = precision_bits_from_history_record(record)

    assert len(bits) == record.level_count
    assert all(bit >= 1 for bit in bits)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (([0.2], [3]), ValueError),
        (([0.2, 0.3], []), ValueError),
        (([0.2, 0.3], [0]), ValueError),
    ],
)
def test_stopping_inputs_are_strict(
    args: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        VariableTimeStoppingTransducer(*args)  # type: ignore[arg-type]
