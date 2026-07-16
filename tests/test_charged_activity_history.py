from __future__ import annotations

import numpy as np
import pytest

from qgapselect.charged_activity_history import (
    CLAIM_STATUS,
    ChargedPhaseHistoryTransducer,
    deterministic_boundary_phases,
    logarithmic_precision_schedule,
)


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_charged_compute_is_reversible_and_charges_qpe_units() -> None:
    transducer = ChargedPhaseHistoryTransducer(
        [0.10, 0.49, 0.52, 0.86],
        [3, 4],
        boundary_phase=0.5,
        activity_window_multipliers=[2.0, 2.0],
    )
    state = _random_state(transducer.statevector_dimension, 17)

    computed = transducer.apply_compute(state)
    restored = transducer.apply_compute(computed)

    assert np.allclose(restored, state, atol=1e-12)
    snapshot = transducer.query_snapshot()
    assert snapshot["charged_history:compute"] == 2
    assert snapshot["charged:coherent_classifier_calls"] == 2
    assert snapshot["charged:qpe_query_units_serial_levels"] == 2 * ((2**3 - 1) + (2**4 - 1))
    assert transducer.resource_snapshot().claim_status == CLAIM_STATUS


def test_charged_phase_oracle_restores_flags_and_matches_generated_rows() -> None:
    transducer = ChargedPhaseHistoryTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3],
        boundary_phase=0.5,
        activity_window_multipliers=[2.0],
    )
    active_rows, output_rows = transducer.predicate_rows()
    assert active_rows == ((1, 2),)
    assert output_rows == ((2,),)

    state = transducer.uniform_history_state()
    result = transducer.apply_phase(state, phase_on="output")
    view = result.state.reshape(transducer.shape)
    expected = state.reshape(transducer.shape).copy()
    expected[0, 2, 0, 0] *= -1

    assert np.allclose(view, expected, atol=1e-12)
    assert np.count_nonzero(view[:, :, :, 1]) == 0
    assert np.count_nonzero(view[:, :, 1, :]) == 0
    assert result.active_probability == pytest.approx(2 / 4)
    assert result.output_probability == pytest.approx(1 / 4)
    assert result.resources.query_counts["charged_history:phase:output"] == 1


def test_charged_summary_is_generated_from_phase_windows() -> None:
    transducer = ChargedPhaseHistoryTransducer(
        deterministic_boundary_phases(16, near_boundary_fraction=0.5),
        logarithmic_precision_schedule(4, base_bits=3, growth_period=1),
        boundary_phase=0.5,
    )

    summary = transducer.summarize_predicates()

    assert summary.predicate_source == "finite_qpe_phase_windows"
    assert summary.output_subset_active is True
    assert summary.total_active_pairs > 0
    assert summary.total_output_pairs > 0
    assert summary.serial_qpe_query_units_per_compute == sum(2**bits - 1 for bits in [3, 4, 5, 6])
    assert summary.max_level_qpe_query_units_per_compute == 2**6 - 1


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        (([0.2], [3]), {}, ValueError),
        (([0.2, 1.1], [3]), {}, ValueError),
        (([0.2, 0.3], []), {}, ValueError),
        (([0.2, 0.3], [0]), {}, ValueError),
        (([0.2, 0.3], [3]), {"activity_window_multipliers": [1.0, 1.0]}, ValueError),
        (([0.2, 0.3], [3]), {"boundary_phase": -0.1}, ValueError),
    ],
)
def test_charged_inputs_are_strict(
    args: tuple[object, ...], kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ChargedPhaseHistoryTransducer(*args, **kwargs)  # type: ignore[arg-type]
