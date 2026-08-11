"""Efficient Speech Commands experiment runner using dataset manifests instead of full scans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from .audio_atlas import (
    _extract_features,
    _fix_waveform,
    _nearest_same_label_indices,
    _optional_dependencies,
    _select_correct_per_class,
    audio_control_descriptors,
)
from .text_atlas import (
    control_output,
    control_projection_from_head,
    pairwise_distances,
    residual_hidden,
    standardized_projection,
)
from .topology_persistence import collision_filtration_profile


def _balanced_from_manifest(
    dataset,
    *,
    commands: Sequence[str],
    per_class: int,
    seed: int,
    target_samples: int,
    torch,
) -> tuple[np.ndarray, np.ndarray]:
    walker = getattr(dataset, "_walker", None)
    if walker is None:
        raise RuntimeError("current torchaudio SPEECHCOMMANDS dataset lacks the expected file manifest")
    by_label: dict[str, list[int]] = {label: [] for label in commands}
    for index, filename in enumerate(walker):
        label = Path(str(filename)).parent.name
        if label in by_label:
            by_label[label].append(index)
    rng = np.random.default_rng(seed)
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for label_index, label in enumerate(commands):
        candidates = np.asarray(by_label[label], dtype=int)
        if len(candidates) < per_class:
            raise RuntimeError(
                f"Speech Commands manifest has {len(candidates)} {label!r} files; "
                f"{per_class} required"
            )
        chosen = rng.choice(candidates, size=per_class, replace=False)
        for index in chosen:
            waveform, sample_rate, _, _, _ = dataset[int(index)]
            waveforms.append(
                _fix_waveform(
                    waveform,
                    sample_rate=int(sample_rate),
                    target_samples=target_samples,
                    torch=torch,
                )
            )
            labels.append(label_index)
    return np.stack(waveforms), np.asarray(labels, dtype=np.int64)


def run_audio_manifest_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    fixture_seed: int,
) -> dict[str, object]:
    """Run one fixed Speech Commands fixture with a common linear behavior probe."""

    (
        torch,
        torchaudio,
        LogisticRegression,
        Ridge,
        StandardScaler,
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForAudioClassification,
    ) = _optional_dependencies()
    if architecture not in tuple(str(value) for value in config["architectures"]):
        raise ValueError(f"architecture {architecture!r} is not configured")
    if fixture_seed not in tuple(int(value) for value in config["fixture_seeds"]):
        raise ValueError(f"fixture_seed={fixture_seed} is not configured")
    torch.manual_seed(fixture_seed)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    np.random.seed(fixture_seed % (2**32))

    root = str(config.get("data_root", ".cache/qcollide-audio"))
    dataset_args = {"root": root, "url": "speech_commands_v0.02", "download": True}
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(subset="training", **dataset_args)
    calibration_dataset = torchaudio.datasets.SPEECHCOMMANDS(subset="validation", **dataset_args)
    evaluation_dataset = torchaudio.datasets.SPEECHCOMMANDS(subset="testing", **dataset_args)
    commands = tuple(str(value) for value in config["commands"])
    target_samples = int(config.get("target_samples", 16000))
    train_waveforms, train_labels = _balanced_from_manifest(
        train_dataset,
        commands=commands,
        per_class=int(config["train_per_class"]),
        seed=fixture_seed + 1,
        target_samples=target_samples,
        torch=torch,
    )
    calibration_waveforms, calibration_labels = _balanced_from_manifest(
        calibration_dataset,
        commands=commands,
        per_class=int(config["calibration_per_class"]),
        seed=fixture_seed + 2,
        target_samples=target_samples,
        torch=torch,
    )
    evaluation_waveforms, evaluation_labels = _balanced_from_manifest(
        evaluation_dataset,
        commands=commands,
        per_class=int(config["evaluation_per_class"]),
        seed=fixture_seed + 3,
        target_samples=target_samples,
        torch=torch,
    )

    (
        train_hidden_raw,
        calibration_hidden_raw,
        evaluation_hidden_raw,
        parameter_count,
        source,
    ) = _extract_features(
        architecture=architecture,
        torch=torch,
        train_waveforms=train_waveforms,
        calibration_waveforms=calibration_waveforms,
        evaluation_waveforms=evaluation_waveforms,
        train_labels=train_labels,
        config=config,
        model_seed=fixture_seed,
        AutoFeatureExtractor=AutoFeatureExtractor,
        AutoModel=AutoModel,
        AutoModelForAudioClassification=AutoModelForAudioClassification,
    )
    hidden_scaler = StandardScaler().fit(train_hidden_raw)
    train_hidden = hidden_scaler.transform(train_hidden_raw)
    calibration_hidden = hidden_scaler.transform(calibration_hidden_raw)
    evaluation_hidden = hidden_scaler.transform(evaluation_hidden_raw)

    classifier = LogisticRegression(
        C=float(config.get("classifier_c", 1.0)),
        max_iter=int(config.get("classifier_max_iter", 800)),
        solver="lbfgs",
    ).fit(train_hidden, train_labels)
    calibration_logits = classifier.decision_function(calibration_hidden)
    evaluation_logits = classifier.decision_function(evaluation_hidden)
    calibration_predictions = calibration_logits.argmax(axis=1)
    evaluation_predictions = evaluation_logits.argmax(axis=1)
    calibration_accuracy = float(np.mean(calibration_predictions == calibration_labels))
    evaluation_accuracy = float(np.mean(evaluation_predictions == evaluation_labels))

    descriptor_scaler = StandardScaler().fit(audio_control_descriptors(train_waveforms))
    train_controls = descriptor_scaler.transform(audio_control_descriptors(train_waveforms))
    calibration_controls = descriptor_scaler.transform(
        audio_control_descriptors(calibration_waveforms)
    )
    control_head = Ridge(alpha=float(config.get("control_ridge_alpha", 1.0))).fit(
        train_hidden,
        train_controls,
    )
    control_r2 = float(control_head.score(calibration_hidden, calibration_controls))
    coefficients = np.asarray(control_head.coef_, dtype=float)

    correct_calibration = np.flatnonzero(calibration_predictions == calibration_labels)
    benign_hidden = calibration_hidden[correct_calibration]
    benign_logits = calibration_logits[correct_calibration]
    benign_labels = calibration_labels[correct_calibration]
    anchors_per_class = int(config["anchors_per_class"])
    anchor_indices = _select_correct_per_class(
        calibration_logits,
        calibration_labels,
        per_class=anchors_per_class,
    )
    candidate_indices = _select_correct_per_class(
        evaluation_logits,
        evaluation_labels,
        per_class=anchors_per_class,
    )
    anchor_hidden = calibration_hidden[anchor_indices]
    candidate_hidden = evaluation_hidden[candidate_indices]
    anchor_logits = calibration_logits[anchor_indices]
    candidate_logits = evaluation_logits[candidate_indices]
    anchor_labels = calibration_labels[anchor_indices]
    candidate_labels = evaluation_labels[candidate_indices]

    rows: list[dict[str, Any]] = []
    for visible_rank in (int(value) for value in config["visible_ranks"]):
        projection = control_projection_from_head(coefficients, visible_rank)
        standardized = standardized_projection(projection, train_hidden)
        benign_control = control_output(benign_hidden, standardized)
        benign_residual = residual_hidden(benign_hidden, projection)
        nearest = _nearest_same_label_indices(benign_control, benign_labels)
        nominal_epsilon = max(
            float(
                np.quantile(
                    np.linalg.norm(benign_control - benign_control[nearest], axis=1),
                    float(config["control_quantile"]),
                )
            ),
            1e-8,
        )
        payload_delta = float(
            np.quantile(
                np.linalg.norm(benign_residual - benign_residual[nearest], axis=1),
                float(config["payload_quantile"]),
            )
        )
        behavior_gamma = float(
            np.quantile(
                np.linalg.norm(benign_logits - benign_logits[nearest], axis=1)
                / sqrt(benign_logits.shape[1]),
                float(config["behavior_quantile"]),
            )
        )
        thresholds = tuple(
            sorted(
                {
                    max(1e-10, nominal_epsilon * float(multiplier))
                    for multiplier in config["epsilon_multipliers"]
                }
            )
        )
        anchor_control = control_output(anchor_hidden, standardized)
        candidate_control = control_output(candidate_hidden, standardized)
        anchor_residual = residual_hidden(anchor_hidden, projection)
        candidate_residual = residual_hidden(candidate_hidden, projection)
        control_distances = pairwise_distances(candidate_control, anchor_control)
        payload_distances = pairwise_distances(candidate_residual, anchor_residual)
        behavior_distances = pairwise_distances(candidate_logits, anchor_logits) / sqrt(
            anchor_logits.shape[1]
        )
        valid_pairs = candidate_labels[:, None] != anchor_labels[None, :]
        displacements = candidate_residual[:, None, :] - anchor_residual[None, :, :]
        profile = collision_filtration_profile(
            control_distances,
            payload_distances,
            behavior_distances,
            thresholds,
            nominal_epsilon=nominal_epsilon,
            payload_delta=payload_delta,
            behavior_gamma=behavior_gamma,
            valid_pairs=valid_pairs,
            edge_displacements=displacements,
        )
        rows.append(
            {
                "visible_rank": visible_rank,
                "nominal_epsilon": nominal_epsilon,
                "payload_delta": payload_delta,
                "behavior_gamma": behavior_gamma,
                "filtration_summary": asdict(profile.summary),
                "basin_persistence": asdict(profile.basin_persistence),
                "filtration_points": [
                    {"control_epsilon": point.control_epsilon, **asdict(point.metrics)}
                    for point in profile.points
                ],
            }
        )

    minimum_control_r2 = float(config["minimum_control_r2"])
    minimum_accuracy = float(config["minimum_evaluation_accuracy"])
    return {
        "artifact_type": "qcollide_audio_functional_collision_atlas_component",
        "schema_version": 1,
        "domain": "audio",
        "dataset": "speech_commands_v0.02",
        "command_subset": list(commands),
        "architecture": architecture,
        "fixture_seed": fixture_seed,
        "feature_source": source,
        "model_diagnostics": {
            "hidden_dimension": int(train_hidden.shape[1]),
            "parameter_count": parameter_count,
            "calibration_accuracy": calibration_accuracy,
            "evaluation_accuracy": evaluation_accuracy,
            "control_head_calibration_r2": control_r2,
        },
        "gates": {
            "minimum_control_r2": minimum_control_r2,
            "control_r2_pass": control_r2 >= minimum_control_r2,
            "minimum_evaluation_accuracy": minimum_accuracy,
            "evaluation_accuracy_pass": evaluation_accuracy >= minimum_accuracy,
        },
        "selection": {
            "anchors_per_class": anchors_per_class,
            "anchor_count": int(len(anchor_indices)),
            "candidate_count": int(len(candidate_indices)),
        },
        "rows": rows,
        "claim_boundary": {
            "manifest_selection_avoids_full_audio_scan": True,
            "commands_are_fixed_before_results": True,
            "common_linear_task_probe": True,
            "control_descriptors_are_label_agnostic": True,
            "cnn_is_supervised_trained_but_pretrained_models_are_frozen": True,
            "architecture_causality_claimed": False,
            "quantum_execution": False,
        },
    }


__all__ = ["run_audio_manifest_component"]
