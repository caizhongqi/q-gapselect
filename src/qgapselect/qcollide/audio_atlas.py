"""Speech Commands functional-collision atlas across CNN, Wav2Vec2, and AST."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from .text_atlas import (
    control_output,
    control_projection_from_head,
    pairwise_distances,
    residual_hidden,
    standardized_projection,
)
from .topology_persistence import collision_filtration_profile


def audio_control_descriptors(waveforms: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Eight label-free acoustic descriptors for fixed-length mono waveforms."""

    values = np.asarray(waveforms, dtype=float)
    if values.ndim != 2 or values.shape[1] < 256:
        raise ValueError("waveforms must have shape (batch, samples>=256)")
    rms = np.sqrt(np.mean(values**2, axis=1) + 1e-12)
    zero_crossing = np.mean(values[:, 1:] * values[:, :-1] < 0.0, axis=1)
    windowed = values * np.hanning(values.shape[1])[None, :]
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / sample_rate)
    power = spectrum.sum(axis=1)
    centroid = (spectrum @ frequencies) / np.maximum(power, 1e-12)
    centered = frequencies[None, :] - centroid[:, None]
    bandwidth = np.sqrt(
        np.sum(spectrum * centered**2, axis=1) / np.maximum(power, 1e-12)
    )
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_threshold = 0.85 * power
    rolloff_indices = np.argmax(cumulative >= rolloff_threshold[:, None], axis=1)
    rolloff = frequencies[rolloff_indices]
    low = spectrum[:, frequencies <= 1000.0].sum(axis=1) / np.maximum(power, 1e-12)
    high = spectrum[:, frequencies >= 4000.0].sum(axis=1) / np.maximum(power, 1e-12)
    envelope_bins = values.reshape(values.shape[0], 100, values.shape[1] // 100)
    envelope = np.sqrt(np.mean(envelope_bins**2, axis=2) + 1e-12)
    envelope_variability = envelope.std(axis=1) / np.maximum(envelope.mean(axis=1), 1e-12)
    return np.column_stack(
        [
            np.log1p(rms),
            zero_crossing,
            centroid / sample_rate,
            bandwidth / sample_rate,
            rolloff / sample_rate,
            low,
            high,
            envelope_variability,
        ]
    )


def _optional_dependencies():
    try:
        import torch
        import torchaudio
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler
        from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError(
            "install audio atlas dependencies: torch torchaudio transformers scikit-learn"
        ) from exc
    return (
        torch,
        torchaudio,
        LogisticRegression,
        Ridge,
        StandardScaler,
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForAudioClassification,
    )


def _fix_waveform(waveform, *, sample_rate: int, target_samples: int, torch) -> np.ndarray:
    tensor = waveform.float()
    if tensor.ndim != 2:
        raise ValueError("Speech Commands waveform must have shape (channels, samples)")
    tensor = tensor.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz Speech Commands audio, got {sample_rate}")
    if tensor.shape[1] < target_samples:
        tensor = torch.nn.functional.pad(tensor, (0, target_samples - tensor.shape[1]))
    elif tensor.shape[1] > target_samples:
        tensor = tensor[:, :target_samples]
    return tensor.squeeze(0).cpu().numpy().astype(np.float32)


def _load_balanced_split(
    dataset,
    *,
    commands: Sequence[str],
    per_class: int,
    seed: int,
    target_samples: int,
    torch,
) -> tuple[np.ndarray, np.ndarray]:
    by_label: dict[str, list[int]] = {label: [] for label in commands}
    for index in range(len(dataset)):
        _, _, label, _, _ = dataset[index]
        if label in by_label:
            by_label[label].append(index)
    rng = np.random.default_rng(seed)
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for label_index, label in enumerate(commands):
        candidates = np.asarray(by_label[label], dtype=int)
        if len(candidates) < per_class:
            raise RuntimeError(
                f"Speech Commands split has {len(candidates)} examples of {label}; "
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


def _build_audio_cnn(torch, classes: int, hidden_dimension: int):
    nn = torch.nn

    class AudioCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, 11, stride=4, padding=5),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 128, 5, stride=2, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.project = nn.Linear(128, hidden_dimension)
            self.head = nn.Linear(hidden_dimension, classes)

        def encode(self, x):
            hidden = self.encoder(x[:, None]).squeeze(-1)
            return torch.nn.functional.gelu(self.project(hidden))

        def forward(self, x):
            hidden = self.encode(x)
            return self.head(hidden), hidden

    return AudioCNN()


def _train_audio_cnn(
    torch,
    waveforms: np.ndarray,
    labels: np.ndarray,
    *,
    classes: int,
    hidden_dimension: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    torch.manual_seed(seed)
    model = _build_audio_cnn(torch, classes, hidden_dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 17)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(waveforms))
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            x = torch.from_numpy(waveforms[chosen])
            y = torch.from_numpy(labels[chosen])
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    return model


def _cnn_features(torch, model, waveforms: np.ndarray, *, batch_size: int) -> np.ndarray:
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            batch = torch.from_numpy(waveforms[start : start + batch_size])
            _, hidden = model(batch)
            parts.append(hidden.cpu().float().numpy())
    return np.concatenate(parts)


def _wav2vec_features(
    torch,
    waveforms: np.ndarray,
    *,
    model_name: str,
    batch_size: int,
    AutoFeatureExtractor,
    AutoModel,
) -> tuple[np.ndarray, int]:
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            batch = [row for row in waveforms[start : start + batch_size]]
            encoded = extractor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
            output = model(**encoded, return_dict=True)
            parts.append(output.last_hidden_state.mean(dim=1).cpu().float().numpy())
    count = int(sum(parameter.numel() for parameter in model.parameters()))
    return np.concatenate(parts), count


def _ast_features(
    torch,
    waveforms: np.ndarray,
    *,
    model_name: str,
    batch_size: int,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
) -> tuple[np.ndarray, int]:
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            batch = [row for row in waveforms[start : start + batch_size]]
            encoded = extractor(batch, sampling_rate=16000, return_tensors="pt")
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states[-1]
            parts.append(hidden.mean(dim=1).cpu().float().numpy())
    count = int(sum(parameter.numel() for parameter in model.parameters()))
    return np.concatenate(parts), count


def _extract_features(
    *,
    architecture: str,
    torch,
    train_waveforms: np.ndarray,
    calibration_waveforms: np.ndarray,
    evaluation_waveforms: np.ndarray,
    train_labels: np.ndarray,
    config: Mapping[str, object],
    model_seed: int,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    batch_size = int(config["feature_batch_size"])
    if architecture == "audio_cnn":
        model = _train_audio_cnn(
            torch,
            train_waveforms,
            train_labels,
            classes=len(config["commands"]),
            hidden_dimension=int(config["cnn_hidden_dimension"]),
            seed=model_seed,
            epochs=int(config["cnn_epochs"]),
            batch_size=int(config["cnn_batch_size"]),
            learning_rate=float(config["cnn_learning_rate"]),
        )
        return (
            _cnn_features(torch, model, train_waveforms, batch_size=batch_size),
            _cnn_features(torch, model, calibration_waveforms, batch_size=batch_size),
            _cnn_features(torch, model, evaluation_waveforms, batch_size=batch_size),
            int(sum(parameter.numel() for parameter in model.parameters())),
            "trained_audio_cnn",
        )
    if architecture == "wav2vec2":
        model_name = str(config["wav2vec2_model"])
        train, count = _wav2vec_features(
            torch,
            train_waveforms,
            model_name=model_name,
            batch_size=batch_size,
            AutoFeatureExtractor=AutoFeatureExtractor,
            AutoModel=AutoModel,
        )
        calibration, _ = _wav2vec_features(
            torch,
            calibration_waveforms,
            model_name=model_name,
            batch_size=batch_size,
            AutoFeatureExtractor=AutoFeatureExtractor,
            AutoModel=AutoModel,
        )
        evaluation, _ = _wav2vec_features(
            torch,
            evaluation_waveforms,
            model_name=model_name,
            batch_size=batch_size,
            AutoFeatureExtractor=AutoFeatureExtractor,
            AutoModel=AutoModel,
        )
        return train, calibration, evaluation, count, model_name
    if architecture == "ast":
        model_name = str(config["ast_model"])
        train, count = _ast_features(
            torch,
            train_waveforms,
            model_name=model_name,
            batch_size=batch_size,
            AutoFeatureExtractor=AutoFeatureExtractor,
            AutoModelForAudioClassification=AutoModelForAudioClassification,
        )
        calibration, _ = _ast_features(
            torch,
            calibration_waveforms,
            model_name=model_name,
            batch_size=batch_size,
            AutoFeatureExtractor=AutoFeatureExtractor,
            AutoModelForAudioClassification=AutoModelForAudioClassification,
        )
        evaluation, _ = _ast_features(
            torch,
            evaluation_waveforms,
            model_name=model_name,
            batch_size=batch_size,
            AutoFeatureExtractor=AutoFeatureExtractor,
            AutoModelForAudioClassification=AutoModelForAudioClassification,
        )
        return train, calibration, evaluation, count, model_name
    raise ValueError(f"unknown audio architecture {architecture!r}")


def _select_correct_per_class(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
) -> np.ndarray:
    predictions = logits.argmax(axis=1)
    selected: list[int] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) < per_class:
            raise RuntimeError(
                f"audio class {label} has {len(candidates)} correct examples; {per_class} required"
            )
        selected.extend(int(value) for value in candidates[:per_class])
    return np.asarray(selected, dtype=np.int64)


def _nearest_same_label_indices(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(values, values)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    distances[~same] = np.inf
    if np.any(~np.isfinite(np.min(distances, axis=1))):
        raise RuntimeError("each audio calibration class needs at least two correct examples")
    return np.argmin(distances, axis=1)


def run_audio_atlas_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    fixture_seed: int,
) -> dict[str, object]:
    """Run one Speech Commands v2 collision-topology component."""

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
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root, url="speech_commands_v0.02", subset="training", download=True)
    calibration_dataset = torchaudio.datasets.SPEECHCOMMANDS(root, url="speech_commands_v0.02", subset="validation", download=True)
    evaluation_dataset = torchaudio.datasets.SPEECHCOMMANDS(root, url="speech_commands_v0.02", subset="testing", download=True)
    commands = tuple(str(value) for value in config["commands"])
    target_samples = int(config.get("target_samples", 16000))
    train_waveforms, train_labels = _load_balanced_split(
        train_dataset,
        commands=commands,
        per_class=int(config["train_per_class"]),
        seed=fixture_seed + 1,
        target_samples=target_samples,
        torch=torch,
    )
    calibration_waveforms, calibration_labels = _load_balanced_split(
        calibration_dataset,
        commands=commands,
        per_class=int(config["calibration_per_class"]),
        seed=fixture_seed + 2,
        target_samples=target_samples,
        torch=torch,
    )
    evaluation_waveforms, evaluation_labels = _load_balanced_split(
        evaluation_dataset,
        commands=commands,
        per_class=int(config["evaluation_per_class"]),
        seed=fixture_seed + 3,
        target_samples=target_samples,
        torch=torch,
    )

    train_hidden_raw, calibration_hidden_raw, evaluation_hidden_raw, parameter_count, source = _extract_features(
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
    calibration_controls = descriptor_scaler.transform(audio_control_descriptors(calibration_waveforms))
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
            "commands_are_fixed_before_results": True,
            "common_linear_task_probe": True,
            "control_descriptors_are_label_agnostic": True,
            "cnn_is_supervised_trained_but_pretrained_models_are_frozen": True,
            "architecture_causality_claimed": False,
            "quantum_execution": False,
        },
    }


__all__ = ["audio_control_descriptors", "run_audio_atlas_component"]
