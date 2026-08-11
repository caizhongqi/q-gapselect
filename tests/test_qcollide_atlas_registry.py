import json
from pathlib import Path

from qgapselect.qcollide.atlas_registry import validate_atlas_registry


def test_frozen_neural_atlas_registry_meets_coverage_gates():
    registry = json.loads(
        Path("configs/qcollide_neural_atlas_registry.json").read_text(encoding="utf-8")
    )
    audit = validate_atlas_registry(registry)
    assert audit["domain_count"] == 8
    assert audit["gates"]["all_required_domains_present"] is True
    assert audit["gates"]["unique_system_ids"] is True
    assert audit["gates"]["at_least_40_system_cells"] is True
    assert audit["gates"]["at_least_20_model_names"] is True
    assert audit["gates"]["at_least_12_datasets"] is True
    assert audit["gates"]["at_least_12_architecture_families"] is True


def test_registry_has_query_discovery_and_real_graph_quantum_tracks():
    registry = json.loads(
        Path("configs/qcollide_neural_atlas_registry.json").read_text(encoding="utf-8")
    )
    audit = validate_atlas_registry(registry)
    experiments = set(audit["cross_cutting_experiments"])
    assert "collision_discovery_query_budget" in experiments
    assert "cross_model_transfer" in experiments
    assert "real_graph_quantum_capacity" in experiments
    assert "causal_collision_closure" in experiments
