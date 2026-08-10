from qgapselect.qcollide.cifar_performance_results import (
    merge_cifar_performance_components,
)


def _component(architecture: str, seed: int) -> dict[str, object]:
    rows = [
        {
            "architecture": architecture,
            "model_seed": seed,
            "checkpoint_epoch": epoch,
            "calibration_accuracy": 0.1 + 0.1 * epoch,
            "evaluation_accuracy": 0.1 + 0.08 * epoch,
        }
        for epoch in (0, 1, 3)
    ]
    return {
        "artifact_type": "qcollide_cifar_checkpoint_performance_component",
        "schema_version": 1,
        "master_seed": 17,
        "architecture": architecture,
        "model_seed": seed,
        "training_seed": seed + 100,
        "claim_scope": {},
        "training": {
            "architecture": architecture,
            "model_seed": seed,
            "evaluation_accuracy": rows[-1]["evaluation_accuracy"],
        },
        "rows": rows,
        "final_accuracy_consistency_error": 0.0,
    }


def test_performance_merge_requires_complete_grid_and_preserves_checkpoints() -> None:
    architectures = ["resnet18", "vit_tiny"]
    seeds = [0, 1]
    merged = merge_cifar_performance_components(
        [
            _component(architecture, seed)
            for architecture in architectures
            for seed in seeds
        ],
        configured_architectures=architectures,
        configured_seeds=seeds,
    )
    assert merged["component_count"] == 4
    assert len(merged["rows"]) == 12
    assert len(merged["checkpoint_summary"]) == 6
    assert merged["gates"]["complete_component_grid"] is True
    assert merged["gates"]["maximum_final_accuracy_consistency_error"] == 0.0
