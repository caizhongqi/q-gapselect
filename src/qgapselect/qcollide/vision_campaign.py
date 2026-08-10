"""End-to-end scalable real-image CNN/Transformer Q-COLLIDE campaign."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .vision_attack import evaluate_packing, generate_attacks
from .vision_common import (
    adaptive_rank_summary,
    architectures,
    integer_values,
    mean_geometry,
    number,
    packed_costs,
    positive_int,
    sequence,
    summarize_rows,
)
from .vision_interventions import (
    dangerous_displacements,
    dangerous_hidden_gradients,
    dangerous_spectrum,
    intervention_bases,
    residual_dangerous_energy,
)
from .vision_linear import combine_projection, row_basis, visible_control_projection
from .vision_models import derived_seed, load_real_vision_dataset, train_vision_model
from .vision_panels import (
    calibrate_projection,
    make_benign_panel,
    prepare_anchor_panel,
    stratified_correct_indices,
)


def _optional_positive_int(config: Mapping[str, object], name: str) -> int | None:
    value = config.get(name)
    if value is None:
        return None
    return positive_int(value, name)


def run_real_vision_campaign(config: Mapping[str, object]) -> dict[str, object]:
    schema_version = positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("only schema_version=1 is supported")
    master_seed = positive_int(config.get("master_seed"), "master_seed")
    dataset_seed = positive_int(config.get("dataset_seed"), "dataset_seed")
    dataset_name = str(config.get("dataset", "digits"))
    data_root = Path(str(config.get("data_root", ".cache/qcollide-data")))
    architecture_values = architectures(config.get("architectures"))
    model_seeds = integer_values(config.get("model_seeds"), "model_seeds", allow_zero=True)
    visible_ranks = integer_values(config.get("visible_ranks"), "visible_ranks")
    closure_ranks = integer_values(config.get("closure_ranks"), "closure_ranks")
    configured_closure_ranks = tuple(
        int(value) for value in sequence(config.get("closure_ranks"), "closure_ranks")
    )
    if 0 not in configured_closure_ranks:
        closure_ranks = (0, *closure_ranks)

    integer_names = (
        "hidden_dimension",
        "control_dimension",
        "epochs",
        "batch_size",
        "anchors_per_class",
        "support_samples",
        "benign_repetitions",
        "line_search_steps",
    )
    integers = {name: positive_int(config.get(name), name) for name in integer_names}
    jacobian_batch_size = positive_int(
        config.get("jacobian_batch_size", 64),
        "jacobian_batch_size",
    )
    numeric_names = (
        "learning_rate",
        "control_loss_weight",
        "benign_radius",
        "control_quantile",
        "payload_quantile",
        "maximum_attack_radius",
        "target_margin",
        "packing_target",
    )
    numbers = {name: number(config.get(name), name) for name in numeric_names}
    for name in ("control_quantile", "payload_quantile", "packing_target"):
        if not 0.0 < numbers[name] < 1.0:
            raise ValueError(f"{name} must lie in (0,1)")
    if max(visible_ranks) > integers["control_dimension"]:
        raise ValueError("visible_ranks cannot exceed control_dimension")

    data = load_real_vision_dataset(
        dataset=dataset_name,
        dataset_seed=dataset_seed,
        data_root=data_root,
        train_samples=_optional_positive_int(config, "train_samples"),
        calibration_samples=_optional_positive_int(config, "calibration_samples"),
        evaluation_samples=_optional_positive_int(config, "evaluation_samples"),
    )
    rows: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []

    for architecture in architecture_values:
        for model_seed in model_seeds:
            seed_parts = (architecture, model_seed)
            if data.dataset_name != "scikit-learn digits":
                seed_parts = (data.dataset_name, *seed_parts)
            seed = derived_seed(master_seed, *seed_parts)
            model, diagnostics = train_vision_model(
                architecture,
                data,
                seed=seed,
                hidden_dimension=integers["hidden_dimension"],
                control_dimension=integers["control_dimension"],
                epochs=integers["epochs"],
                batch_size=integers["batch_size"],
                learning_rate=numbers["learning_rate"],
                control_loss_weight=numbers["control_loss_weight"],
            )
            training.append({"model_seed": model_seed, **diagnostics})
            calibration_indices = stratified_correct_indices(
                model,
                data.calibration_images,
                data.calibration_labels,
                per_class=integers["anchors_per_class"],
            )
            evaluation_indices = stratified_correct_indices(
                model,
                data.evaluation_images,
                data.evaluation_labels,
                per_class=integers["anchors_per_class"],
            )
            calibration_panel = prepare_anchor_panel(
                model,
                data.calibration_images,
                data.calibration_labels,
                calibration_indices,
                jacobian_batch_size=jacobian_batch_size,
            )
            evaluation_panel = prepare_anchor_panel(
                model,
                data.evaluation_images,
                data.evaluation_labels,
                evaluation_indices,
                jacobian_batch_size=jacobian_batch_size,
            )
            support_count = min(integers["support_samples"], len(data.train_images))
            hidden_support_parts: list[torch.Tensor] = []
            with torch.no_grad():
                support_tensor = torch.from_numpy(
                    data.train_images[:support_count, None].astype(np.float32)
                )
                for start in range(0, len(support_tensor), 2048):
                    _, _, hidden_support_tensor = model(support_tensor[start : start + 2048])
                    hidden_support_parts.append(hidden_support_tensor.cpu())
            hidden_support = torch.cat(hidden_support_parts, dim=0).numpy()
            calibration_benign = make_benign_panel(
                model,
                calibration_panel,
                seed=derived_seed(seed, "calibration-benign"),
                repetitions=integers["benign_repetitions"],
                radius=numbers["benign_radius"],
            )

            for visible_rank in visible_ranks:
                base_projection = visible_control_projection(model, visible_rank)
                base_projection_rank = row_basis(base_projection).shape[0]
                base_calibration = calibrate_projection(
                    base_projection,
                    hidden_support,
                    calibration_benign,
                    control_quantile=numbers["control_quantile"],
                    payload_quantile=numbers["payload_quantile"],
                )
                calibration_attacks = generate_attacks(
                    model,
                    calibration_panel,
                    base_calibration,
                    maximum_radius=numbers["maximum_attack_radius"],
                    line_search_steps=integers["line_search_steps"],
                    target_margin=numbers["target_margin"],
                )
                dangerous_displacement = dangerous_displacements(
                    calibration_attacks,
                    calibration_panel,
                    base_projection,
                )
                displacement_spectrum = dangerous_spectrum(dangerous_displacement)
                dangerous_gradient = dangerous_hidden_gradients(
                    model,
                    calibration_panel,
                    base_projection,
                )
                spectrum = dangerous_spectrum(dangerous_gradient)
                bases = intervention_bases(
                    base_projection,
                    spectrum["basis"],
                    hidden_support,
                    maximum_rank=max(closure_ranks),
                    seed=derived_seed(seed, "interventions", visible_rank),
                )
                spectra.append(
                    {
                        "dataset": data.dataset_name,
                        "architecture": architecture,
                        "model_seed": model_seed,
                        "visible_rank": visible_rank,
                        "rank_90": spectrum["rank_90"],
                        "rank_95": spectrum["rank_95"],
                        "rank_99": spectrum["rank_99"],
                        "total_energy": spectrum["total_energy"],
                        "singular_values": spectrum["singular_values"],
                        "displacement_rank_95": displacement_spectrum["rank_95"],
                        "displacement_total_energy": displacement_spectrum["total_energy"],
                    }
                )
                methods = {
                    "baseline": np.zeros((integers["hidden_dimension"], 0)),
                    **bases,
                }
                for intervention, basis in methods.items():
                    for closure_rank in closure_ranks:
                        if intervention == "baseline" and closure_rank != 0:
                            continue
                        if intervention != "baseline" and closure_rank == 0:
                            continue
                        selected = basis[:, : min(closure_rank, basis.shape[1])]
                        projection = combine_projection(base_projection, selected)
                        effective_closure_rank = projection.shape[0] - base_projection_rank
                        if (
                            intervention == "targeted"
                            and closure_rank > 0
                            and effective_closure_rank == 0
                        ):
                            continue
                        projection_calibration = calibrate_projection(
                            projection,
                            hidden_support,
                            calibration_benign,
                            control_quantile=numbers["control_quantile"],
                            payload_quantile=numbers["payload_quantile"],
                        )
                        attacks = generate_attacks(
                            model,
                            evaluation_panel,
                            projection_calibration,
                            maximum_radius=numbers["maximum_attack_radius"],
                            line_search_steps=integers["line_search_steps"],
                            target_margin=numbers["target_margin"],
                        )
                        packing = evaluate_packing(
                            attacks,
                            evaluation_panel,
                            projection_calibration,
                        )
                        if packing.topology is None:
                            raise RuntimeError("packing evaluation did not emit topology metrics")
                        topology = packing.topology
                        mean_openness, mean_tunnel_dimension = mean_geometry(
                            model,
                            evaluation_panel,
                            projection_calibration,
                        )
                        classical_cost, quantum_proxy = packed_costs(
                            evaluation_panel.size,
                            packing.matching_size,
                        )
                        residual_energy = residual_dangerous_energy(
                            dangerous_gradient,
                            basis,
                            closure_rank,
                        )
                        total_energy = max(float(spectrum["total_energy"]), 1e-12)
                        rows.append(
                            {
                                "dataset": data.dataset_name,
                                "architecture": architecture,
                                "model_seed": model_seed,
                                "visible_rank": visible_rank,
                                "intervention": intervention,
                                "closure_rank": closure_rank,
                                "effective_closure_rank": effective_closure_rank,
                                "projection_rank": int(projection.shape[0]),
                                "evaluation_accuracy": diagnostics["evaluation_accuracy"],
                                "control_mse": diagnostics["evaluation_control_mse"],
                                "control_epsilon": projection_calibration.control_epsilon,
                                "payload_delta": projection_calibration.payload_delta,
                                "benign_control_acceptance": (
                                    projection_calibration.control_acceptance
                                ),
                                "benign_payload_exceedance": (
                                    projection_calibration.payload_exceedance
                                ),
                                "mean_openness": mean_openness,
                                "mean_tunnel_dimension": mean_tunnel_dimension,
                                "edge_count": packing.edge_count,
                                "matching_size": packing.matching_size,
                                "packing_fraction": packing.packing_fraction,
                                "candidate_fraction": packing.candidate_fraction,
                                "mean_input_l2": packing.mean_input_l2,
                                "classical_packed_cost": classical_cost,
                                "quantum_query_proxy": quantum_proxy,
                                "residual_energy": residual_energy,
                                "residual_energy_fraction": residual_energy / total_energy,
                                "static_rank_95": spectrum["rank_95"],
                                "collision_active_vertex_fraction": (
                                    topology.active_vertex_fraction
                                ),
                                "collision_beta0_active": topology.beta0_active,
                                "collision_beta1_active": topology.beta1_active,
                                "collision_component_entropy": (
                                    topology.component_edge_entropy
                                ),
                                "collision_normalized_component_entropy": (
                                    topology.normalized_component_edge_entropy
                                ),
                                "collision_effective_component_count": (
                                    topology.effective_component_count
                                ),
                                "collision_largest_component_edge_fraction": (
                                    topology.largest_component_edge_fraction
                                ),
                                "collision_independence_ratio": (
                                    topology.independence_ratio
                                ),
                                "collision_displacement_rank_95": (
                                    topology.displacement_rank_95
                                ),
                                "collision_displacement_entropy_rank": (
                                    topology.displacement_entropy_rank
                                ),
                                "collision_displacement_stable_rank": (
                                    topology.displacement_stable_rank
                                ),
                            }
                        )

    summary = summarize_rows(rows)
    slug = data.dataset_name.lower().replace("-", "_").replace(" ", "_")
    return {
        "artifact_type": f"qcollide_real_{slug}_cnn_transformer_causal_diagnostic",
        "schema_version": schema_version,
        "master_seed": master_seed,
        "claim_scope": {
            "real_image_dataset": True,
            "dataset": data.dataset_name,
            "image_size": data.image_size,
            "class_count": data.class_count,
            "architectures": list(architecture_values),
            "train_samples": len(data.train_images),
            "calibration_samples": len(data.calibration_images),
            "evaluation_samples": len(data.evaluation_images),
            "production_scale_claimed": False,
            "real_world_attack_claimed": False,
            "quantum_values_are": "analytic endpoint-query proxies",
            "coherent_quantum_execution": False,
            "new_lower_bound_claimed": False,
            "full_collision_topology_emitted": True,
            "training_phase_transition_claimed": False,
        },
        "config": dict(config),
        "training": training,
        "spectra": spectra,
        "rows": rows,
        "summary": summary,
        "adaptive_rank_summary": adaptive_rank_summary(
            summary,
            spectra,
            packing_target=numbers["packing_target"],
        ),
    }


__all__ = ["run_real_vision_campaign"]
