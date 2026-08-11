"""Fail-closed selection for independent Cora auxiliary-control weight calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def select_graph_control_weights(
    artifacts: Sequence[Mapping[str, object]],
    *,
    architectures: Sequence[str],
    calibration_seeds: Sequence[int],
    control_loss_weights: Sequence[float],
) -> dict[str, object]:
    """Select the smallest weight whose every calibration seed passes both gates."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    architecture_values = tuple(str(value) for value in architectures)
    seed_values = tuple(int(value) for value in calibration_seeds)
    weight_values = tuple(float(value) for value in control_loss_weights)
    expected = {
        (architecture, seed, weight)
        for architecture in architecture_values
        for seed in seed_values
        for weight in weight_values
    }
    indexed: dict[tuple[str, int, float], Mapping[str, object]] = {}
    for artifact in artifacts:
        if str(artifact.get("artifact_type")) != (
            "qcollide_controlled_graph_functional_collision_component"
        ):
            raise ValueError("unexpected graph calibration artifact type")
        architecture = str(artifact["architecture"])
        seed = int(artifact["model_seed"])
        weight = float(artifact["calibration"]["control_loss_weight"])
        key = (architecture, seed, weight)
        if key in indexed:
            raise ValueError(f"duplicate calibration cell {key}")
        indexed[key] = artifact
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)

    selected: dict[str, float | None] = {}
    architecture_rows: list[dict[str, object]] = []
    for architecture in architecture_values:
        candidates: list[dict[str, object]] = []
        chosen: float | None = None
        for weight in weight_values:
            rows = [indexed.get((architecture, seed, weight)) for seed in seed_values]
            complete = all(row is not None for row in rows)
            control_values = [
                float(row["model_diagnostics"]["control_head_calibration_r2"])
                for row in rows
                if row is not None
            ]
            accuracy_values = [
                float(row["model_diagnostics"]["evaluation_accuracy"])
                for row in rows
                if row is not None
            ]
            control_pass = complete and all(
                bool(row["gates"]["control_r2_pass"])
                for row in rows
                if row is not None
            )
            accuracy_pass = complete and all(
                bool(row["gates"]["evaluation_accuracy_pass"])
                for row in rows
                if row is not None
            )
            joint_pass = bool(control_pass and accuracy_pass)
            candidates.append(
                {
                    "control_loss_weight": weight,
                    "complete_seed_grid": complete,
                    "minimum_control_r2": min(control_values) if control_values else None,
                    "mean_control_r2": (
                        sum(control_values) / len(control_values)
                        if control_values
                        else None
                    ),
                    "minimum_evaluation_accuracy": (
                        min(accuracy_values) if accuracy_values else None
                    ),
                    "mean_evaluation_accuracy": (
                        sum(accuracy_values) / len(accuracy_values)
                        if accuracy_values
                        else None
                    ),
                    "all_control_r2_pass": control_pass,
                    "all_accuracy_pass": accuracy_pass,
                    "joint_gate_pass": joint_pass,
                }
            )
            if chosen is None and joint_pass:
                chosen = weight
        selected[architecture] = chosen
        architecture_rows.append(
            {
                "architecture": architecture,
                "selected_control_loss_weight": chosen,
                "selection_rule": "smallest_jointly_passing_weight",
                "candidates": candidates,
            }
        )

    all_selected = all(value is not None for value in selected.values())
    return {
        "artifact_type": "qcollide_graph_control_weight_calibration_v3",
        "schema_version": 1,
        "architectures": list(architecture_values),
        "calibration_seeds": list(seed_values),
        "control_loss_weights": list(weight_values),
        "selected_control_loss_weight": selected,
        "architecture_rows": architecture_rows,
        "gates": {
            "complete_calibration_grid": not missing and not extra,
            "missing_cell_count": len(missing),
            "extra_cell_count": len(extra),
            "all_architectures_have_jointly_passing_weight": all_selected,
        },
        "claim_boundary": {
            "main_model_seeds_used_for_hyperparameter_selection": False,
            "minimum_control_r2_changed_after_v2": False,
            "minimum_evaluation_accuracy_changed_after_v2": False,
            "selection_uses_smallest_passing_weight": True,
            "confirmatory_main_seed_results_included": False,
        },
    }


__all__ = ["select_graph_control_weights"]
