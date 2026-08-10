import numpy as np
import pytest

from qgapselect.qcollide.capacity_meta import (
    build_capacity_meta_summary,
    extract_capacity_cells,
)


def _document(
    dataset: str,
    offset: float,
    *,
    alternate_schema: bool = False,
) -> dict[str, object]:
    rows = [
        {
            "architecture": architecture,
            "visible_rank": rank,
            "baseline_openness": openness,
            "baseline_packing": min(
                1.0,
                max(0.0, 0.7 * openness + offset + architecture_offset),
            ),
            "baseline_se": 0.01,
        }
        for architecture, architecture_offset in (("cnn", 0.0), ("vit", 0.1))
        for rank, openness in ((1, 0.9), (2, 0.6), (4, 0.3))
    ]
    key = "baseline_and_rank6_closure" if alternate_schema else "baseline_and_closure"
    return {"claim_scope": {"dataset": dataset}, key: rows}


def test_capacity_meta_recovers_dataset_and_architecture_offsets() -> None:
    result = build_capacity_meta_summary(
        [
            _document("dataset-a", 0.0),
            _document("dataset-b", 0.2, alternate_schema=True),
        ]
    )
    model = result["linear_models"]["openness_dataset_architecture"]
    coefficients = model["coefficients"]
    assert result["cell_count"] == 12
    assert np.isclose(coefficients["openness"], 0.7)
    assert np.isclose(coefficients["dataset:dataset-b"], 0.2)
    assert np.isclose(coefficients["architecture:vit"], 0.1)
    assert np.isclose(model["r_squared"], 1.0)


def test_capacity_meta_records_monotone_rank_curves() -> None:
    result = build_capacity_meta_summary(
        [_document("dataset-a", 0.0), _document("dataset-b", 0.2)]
    )
    assert all(
        row["packing_nonincreasing_fraction"] == 1.0
        and row["openness_nonincreasing_fraction"] == 1.0
        for row in result["monotonicity"]
    )
    assert set(result["leave_one_dataset_out"]) == {"dataset-a", "dataset-b"}


def test_extract_capacity_cells_rejects_duplicate_architecture_rank() -> None:
    document = _document("dataset-a", 0.0)
    document["baseline_and_closure"].append(document["baseline_and_closure"][0])
    with pytest.raises(ValueError, match="duplicate baseline cell"):
        extract_capacity_cells(document)


def test_capacity_meta_requires_multiple_unique_datasets() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_capacity_meta_summary([_document("dataset-a", 0.0)])
    with pytest.raises(ValueError, match="unique dataset"):
        build_capacity_meta_summary(
            [_document("dataset-a", 0.0), _document("dataset-a", 0.1)]
        )
