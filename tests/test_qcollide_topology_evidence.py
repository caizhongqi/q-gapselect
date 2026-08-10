import numpy as np

from qgapselect.qcollide.topology_evidence import build_topology_evidence_audit


def _summary(dataset: str, offset: float) -> dict[str, object]:
    return {
        "claim_scope": {"dataset": dataset},
        "baseline_and_closure": [
            {
                "architecture": "cnn",
                "visible_rank": 1,
                "baseline_openness": 0.9 - offset,
                "baseline_packing": 0.8 - offset,
            },
            {
                "architecture": "cnn",
                "visible_rank": 2,
                "baseline_openness": 0.5 - offset,
                "baseline_packing": 0.4 - offset,
            },
        ],
        "static_spectra": [
            {
                "architecture": "cnn",
                "visible_rank": 1,
                "disp_rank95": 4.0,
                "grad_rank95": 5.0,
            },
            {
                "architecture": "cnn",
                "visible_rank": 2,
                "disp_rank95": 2.0,
                "grad_rank95": 5.0,
            },
        ],
    }


def test_legacy_audit_is_explicitly_partial() -> None:
    audit = build_topology_evidence_audit(
        (_summary("dataset-a", 0.0), _summary("dataset-b", 0.1))
    )
    assert audit["cell_count"] == 4
    assert audit["claim_scope"]["supports_full_collision_topology"] is False
    assert audit["claim_scope"]["supports_partial_capacity_dimension_profile"] is True
    status = audit["metric_identifiability"]
    assert status["K_C_matching_capacity"] == "identified"
    assert status["H_C_component_entropy"] == "not_identifiable_without_adjacency"
    assert np.isclose(audit["associations"]["openness_vs_capacity"], 1.0)


def test_duplicate_dataset_documents_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        build_topology_evidence_audit((_summary("same", 0.0), _summary("same", 0.1)))
