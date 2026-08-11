from __future__ import annotations

import pytest

from qgapselect.qcollide.cifar_single_run_selected import (
    select_checkpoint_from_calibration_trajectory,
)


def test_single_run_selector_chooses_nearest_observed_checkpoint():
    result = select_checkpoint_from_calibration_trajectory(
        {0: 0.01, 5: 0.24, 6: 0.29, 7: 0.305, 8: 0.34},
        target_accuracy=0.30,
        maximum_accuracy_mismatch=0.05,
    )
    assert result["checkpoint_epoch"] == 7
    assert result["calibration_accuracy"] == 0.305
    assert abs(result["absolute_calibration_mismatch"] - 0.005) < 1e-12
    assert result["within_tolerance"] is True


def test_single_run_selector_breaks_exact_tie_toward_earlier_epoch():
    result = select_checkpoint_from_calibration_trajectory(
        {4: 0.28, 5: 0.32},
        target_accuracy=0.30,
        maximum_accuracy_mismatch=0.05,
    )
    assert result["checkpoint_epoch"] == 4


def test_single_run_selector_fails_closed_outside_fixed_tolerance():
    with pytest.raises(RuntimeError, match="no checkpoint satisfies"):
        select_checkpoint_from_calibration_trajectory(
            {0: 0.01, 5: 0.10, 10: 0.18},
            target_accuracy=0.30,
            maximum_accuracy_mismatch=0.05,
        )
