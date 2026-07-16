from __future__ import annotations

import numpy as np
import pytest

from qgapselect.activity_history_transducer import (
    CLAIM_STATUS,
    ToyActivityHistoryTransducer,
    toy_transducer_from_history_record,
)
from qgapselect.unknown_boundary_history import unknown_boundary_history_record


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_activity_history_compute_is_reversible_and_charged() -> None:
    transducer = ToyActivityHistoryTransducer(
        5,
        active_indices_by_level=((0, 2), (1, 3, 4)),
        output_indices_by_level=((2,), (3,)),
    )
    state = _random_state(transducer.statevector_dimension, 11)

    computed = transducer.apply_compute(state)
    restored = transducer.apply_compute(computed)

    assert np.allclose(restored, state, atol=1e-12)
    assert transducer.query_snapshot()["history:compute"] == 2
    assert transducer.resource_snapshot().claim_status == CLAIM_STATUS
    assert transducer.resource_snapshot().statevector_dimension == (
        transducer.level_dimension * transducer.index_dimension * 4
    )


def test_activity_history_phase_oracle_restores_flags() -> None:
    transducer = ToyActivityHistoryTransducer(
        4,
        active_indices_by_level=((0, 1), (2, 3)),
        output_indices_by_level=((1,), (2,)),
    )
    state = transducer.uniform_history_state()

    result = transducer.apply_phase(state, phase_on="output")
    view = result.state.reshape(transducer.shape)
    expected = state.reshape(transducer.shape).copy()
    expected[0, 1, 0, 0] *= -1
    expected[1, 2, 0, 0] *= -1

    assert np.allclose(view, expected, atol=1e-12)
    assert np.count_nonzero(view[:, :, :, 1]) == 0
    assert np.count_nonzero(view[:, :, 1, :]) == 0
    assert result.output_probability == pytest.approx(2 / 8)
    assert result.active_probability == pytest.approx(4 / 8)
    assert result.resources.query_counts["history:compute"] == 2
    assert result.resources.query_counts["history:phase:output"] == 1


def test_activity_history_phase_rejects_unknown_predicate() -> None:
    transducer = ToyActivityHistoryTransducer(
        3,
        active_indices_by_level=((0,),),
        output_indices_by_level=((0,),),
    )
    with pytest.raises(ValueError):
        transducer.apply_phase(transducer.zero_state(), phase_on="missing")


def test_toy_transducer_from_history_record_has_promised_counts() -> None:
    record = unknown_boundary_history_record(8)
    transducer = toy_transducer_from_history_record(record)

    assert transducer.n_arms == record.n
    assert transducer.level_count == record.level_count
    assert all(
        len(row) == layer.active_count
        for row, layer in zip(
            transducer.active_indices_by_level, record.layers, strict=True
        )
    )
    assert all(
        len(row) == layer.output_births
        for row, layer in zip(
            transducer.output_indices_by_level, record.layers, strict=True
        )
    )


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((1, ((0,),), ((0,),)), ValueError),
        ((4, (), ()), ValueError),
        ((4, ((0,),), ((0,), (1,))), ValueError),
        ((4, ((0, 0),), ((0,),)), ValueError),
        ((4, ((0,),), ((1,),)), ValueError),
        ((4, ((4,),), ((4,),)), IndexError),
    ],
)
def test_activity_history_inputs_are_strict(
    args: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ToyActivityHistoryTransducer(*args)  # type: ignore[arg-type]
