import numpy as np

from qgapselect.qcollide.topology_persistence import (
    collision_basin_persistence,
    collision_filtration_profile,
)


def _pair_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    control = np.asarray([[0.1, 0.8], [0.9, 0.2]], dtype=float)
    payload = np.ones((2, 2), dtype=float)
    behavior = np.ones((2, 2), dtype=float)
    return control, payload, behavior


def test_basin_persistence_tracks_birth_merge_and_survival() -> None:
    control, payload, behavior = _pair_arrays()
    persistence = collision_basin_persistence(
        control,
        payload,
        behavior,
        epsilon_max=1.0,
        payload_delta=0.5,
        behavior_gamma=0.5,
    )
    assert persistence.interval_count == 2
    assert persistence.finite_interval_count == 1
    assert persistence.essential_interval_count == 1
    assert np.isclose(persistence.total_lifetime, 1.5)
    assert np.isclose(persistence.normalized_total_lifetime, 0.75)
    assert persistence.half_window_persistent_basin_count == 2


def test_filtration_summary_is_monotone_and_detects_cycle_onset() -> None:
    control, payload, behavior = _pair_arrays()
    profile = collision_filtration_profile(
        control,
        payload,
        behavior,
        (0.1, 0.2, 0.8, 0.9, 1.0),
        nominal_epsilon=0.8,
        payload_delta=0.5,
        behavior_gamma=0.5,
    )
    summary = profile.summary
    assert summary.edge_count_monotone
    assert summary.capacity_monotone
    assert summary.cycle_rank_monotone
    assert summary.capacity_onset_epsilon == 0.1
    assert summary.half_max_capacity_epsilon == 0.1
    assert summary.cycle_onset_epsilon == 0.9
    assert summary.peak_beta0 == 2
    assert summary.nominal_capacity_fraction == 1.0


def test_empty_eligible_relation_has_zero_persistence() -> None:
    control, payload, behavior = _pair_arrays()
    persistence = collision_basin_persistence(
        control,
        payload,
        behavior,
        epsilon_max=1.0,
        payload_delta=2.0,
        behavior_gamma=0.5,
    )
    assert persistence.interval_count == 0
    assert persistence.total_lifetime == 0.0
    assert persistence.normalized_total_lifetime == 0.0
