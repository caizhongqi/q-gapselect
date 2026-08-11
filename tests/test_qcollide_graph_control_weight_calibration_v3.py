from qgapselect.qcollide.graph_control_weight_calibration import select_graph_control_weights


def _artifact(architecture, seed, weight, control_r2, accuracy):
    return {
        "artifact_type": "qcollide_controlled_graph_functional_collision_component",
        "architecture": architecture,
        "model_seed": seed,
        "calibration": {"control_loss_weight": weight},
        "model_diagnostics": {
            "control_head_calibration_r2": control_r2,
            "evaluation_accuracy": accuracy,
        },
        "gates": {
            "control_r2_pass": control_r2 >= 0.2,
            "evaluation_accuracy_pass": accuracy >= 0.7,
        },
    }


def test_selects_smallest_weight_passing_both_seeds():
    artifacts = []
    for architecture in ("gcn", "gat"):
        for seed in (100, 101):
            artifacts.append(_artifact(architecture, seed, 0.5, 0.15, 0.8))
            artifacts.append(_artifact(architecture, seed, 1.0, 0.22, 0.78))
            artifacts.append(_artifact(architecture, seed, 2.0, 0.30, 0.74))
    result = select_graph_control_weights(
        artifacts,
        architectures=("gcn", "gat"),
        calibration_seeds=(100, 101),
        control_loss_weights=(0.5, 1.0, 2.0),
    )
    assert result["selected_control_loss_weight"] == {"gcn": 1.0, "gat": 1.0}
    assert result["gates"]["complete_calibration_grid"] is True
    assert result["gates"]["all_architectures_have_jointly_passing_weight"] is True


def test_fails_closed_when_no_weight_passes():
    artifacts = []
    for seed in (100, 101):
        artifacts.append(_artifact("gcn", seed, 0.5, 0.10, 0.80))
        artifacts.append(_artifact("gcn", seed, 1.0, 0.18, 0.78))
    result = select_graph_control_weights(
        artifacts,
        architectures=("gcn",),
        calibration_seeds=(100, 101),
        control_loss_weights=(0.5, 1.0),
    )
    assert result["selected_control_loss_weight"]["gcn"] is None
    assert result["gates"]["all_architectures_have_jointly_passing_weight"] is False
