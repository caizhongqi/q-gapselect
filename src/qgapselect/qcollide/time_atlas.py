"""ETTm1 functional-collision atlas across recurrent, convolutional, and temporal Transformers."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping
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

_ETTM1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"


def time_control_descriptors(windows: np.ndarray, *, target_index: int = -1) -> np.ndarray:
    """Eight label-free trend/seasonality descriptors for multivariate input windows."""

    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[1] < 17:
        raise ValueError("windows must have shape (batch, time>=17, channels)")
    target = values[:, :, target_index]
    differences = np.diff(values, axis=1)
    t = np.linspace(-1.0, 1.0, values.shape[1])
    denominator = float(np.sum(t * t))
    centered_target = target - target.mean(axis=1, keepdims=True)
    slope = centered_target @ t / max(denominator, 1e-12)
    spectrum = np.abs(np.fft.rfft(centered_target, axis=1)) ** 2
    low_frequency = spectrum[:, 1 : min(5, spectrum.shape[1])].sum(axis=1)
    total_frequency = spectrum[:, 1:].sum(axis=1)

    def lag_correlation(lag: int) -> np.ndarray:
        left = centered_target[:, :-lag]
        right = centered_target[:, lag:]
        numerator = np.sum(left * right, axis=1)
        scale = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return numerator / np.maximum(scale, 1e-12)

    channel_means = values.mean(axis=1)
    return np.column_stack(
        [
            values.mean(axis=(1, 2)),
            values.std(axis=(1, 2)),
            np.mean(np.abs(differences), axis=(1, 2)),
            slope,
            low_frequency / np.maximum(total_frequency, 1e-12),
            lag_correlation(4),
            lag_correlation(16),
            channel_means.std(axis=1),
        ]
    )


def _optional_dependencies():
    try:
        import pandas as pd
        import torch
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError(
            "install time atlas extras with: pip install -e '.[atlas_time]'"
        ) from exc
    return torch, pd, Ridge, StandardScaler


def _build_model(
    torch,
    *,
    architecture: str,
    channels: int,
    seq_len: int,
    pred_len: int,
    hidden: int,
):
    nn = torch.nn

    class LSTMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(channels, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden, pred_len)

        def encode(self, x):
            output, _ = self.encoder(x)
            return output[:, -1]

        def forward(self, x):
            hidden_values = self.encode(x)
            return self.head(hidden_values), hidden_values

    class TCNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = channels
            for dilation in (1, 2, 4):
                padding = dilation
                layers.extend(
                    [
                        nn.Conv1d(
                            in_channels,
                            hidden,
                            kernel_size=3,
                            padding=padding,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                    ]
                )
                in_channels = hidden
            self.network = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, pred_len)

        def encode(self, x):
            hidden_values = self.network(x.transpose(1, 2))
            return hidden_values[:, :, -1]

        def forward(self, x):
            hidden_values = self.encode(x)
            return self.head(hidden_values), hidden_values

    class PatchTSTModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_len = 16
            self.stride = 8
            self.patch = nn.Linear(self.patch_len, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, pred_len)

        def encode(self, x):
            patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
            batch, channel_count, patch_count, patch_len = patches.shape
            tokens = self.patch(patches.reshape(batch * channel_count, patch_count, patch_len))
            tokens = self.transformer(tokens)
            channel_hidden = self.norm(tokens.mean(dim=1)).reshape(batch, channel_count, hidden)
            return channel_hidden.mean(dim=1)

        def forward(self, x):
            hidden_values = self.encode(x)
            return self.head(hidden_values), hidden_values

    class PeriodBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(hidden, hidden * 2, kernel_size=(1, 3), padding=(0, 1))
            self.conv3 = nn.Conv2d(hidden, hidden * 2, kernel_size=(3, 3), padding=1)
            self.project = nn.Conv2d(hidden * 2, hidden, kernel_size=1)
            self.norm = nn.LayerNorm(hidden)

        def forward(self, x):
            batch, length, dimension = x.shape
            spectrum = torch.fft.rfft(x, dim=1)
            amplitude = spectrum.abs().mean(dim=(0, 2))
            amplitude[0] = 0
            top_count = min(3, max(1, amplitude.numel() - 1))
            indices = torch.topk(amplitude, top_count).indices
            outputs = []
            weights = []
            for index in indices:
                frequency_index = max(int(index.item()), 1)
                period = max(1, length // frequency_index)
                padded_length = ((length + period - 1) // period) * period
                if padded_length > length:
                    padding = torch.zeros(
                        batch,
                        padded_length - length,
                        dimension,
                        dtype=x.dtype,
                        device=x.device,
                    )
                    padded = torch.cat([x, padding], dim=1)
                else:
                    padded = x
                grid = padded.reshape(batch, padded_length // period, period, dimension)
                grid = grid.permute(0, 3, 1, 2)
                transformed = torch.nn.functional.gelu(self.conv1(grid))
                transformed = transformed + torch.nn.functional.gelu(self.conv3(grid))
                transformed = self.project(transformed).permute(0, 2, 3, 1)
                transformed = transformed.reshape(batch, padded_length, dimension)[:, :length]
                outputs.append(transformed)
                weights.append(amplitude[index])
            weight = torch.softmax(torch.stack(weights), dim=0)
            combined = sum(weight[i] * outputs[i] for i in range(len(outputs)))
            return self.norm(x + combined)

    class TimesNetModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(channels, hidden)
            self.blocks = nn.ModuleList([PeriodBlock(), PeriodBlock()])
            self.head = nn.Linear(hidden, pred_len)

        def encode(self, x):
            hidden_values = self.embed(x)
            for block in self.blocks:
                hidden_values = block(hidden_values)
            return hidden_values.mean(dim=1)

        def forward(self, x):
            hidden_values = self.encode(x)
            return self.head(hidden_values), hidden_values

    builders = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "timesnet": TimesNetModel,
        "patchtst": PatchTSTModel,
    }
    if architecture not in builders:
        raise ValueError(f"unknown time-series architecture {architecture!r}")
    return builders[architecture]()


def _download_etm1(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(_ETTM1_URL, headers={"User-Agent": "q-gapselect-atlas/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())


def _window_indices(start: int, end: int, seq_len: int, pred_len: int) -> np.ndarray:
    first = start
    last = end - seq_len - pred_len
    if last < first:
        raise ValueError("split is too short for configured sequence and prediction lengths")
    return np.arange(first, last + 1, dtype=np.int64)


def _take_indices(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=count, replace=False))


def _make_windows(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([values[index : index + seq_len] for index in indices]).astype(np.float32)
    y = np.stack(
        [
            values[index + seq_len : index + seq_len + pred_len, target_index]
            for index in indices
        ]
    ).astype(np.float32)
    return x, y


def _encode_numpy(torch, model, windows: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    hidden_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size])
            forecast, hidden_values = model(batch)
            predictions.append(forecast.cpu().float().numpy())
            hidden_rows.append(hidden_values.cpu().float().numpy())
    return np.concatenate(predictions), np.concatenate(hidden_rows)


def _nearest_indices(values: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(values, values)
    np.fill_diagonal(distances, np.inf)
    return np.argmin(distances, axis=1)


def run_time_atlas_component(
    config: Mapping[str, object],
    *,
    architecture: str,
    model_seed: int,
) -> dict[str, object]:
    """Train one ETTm1 forecaster and measure functional-collision topology."""

    torch, pd, Ridge, StandardScaler = _optional_dependencies()
    if architecture not in tuple(str(value) for value in config["architectures"]):
        raise ValueError(f"architecture {architecture!r} is not configured")
    if model_seed not in tuple(int(value) for value in config["model_seeds"]):
        raise ValueError(f"model_seed={model_seed} is not configured")
    torch.manual_seed(model_seed)
    np.random.seed(model_seed % (2**32))
    torch.set_num_threads(int(config.get("torch_threads", 2)))

    data_path = Path(str(config.get("data_path", ".cache/qcollide-time/ETTm1.csv")))
    _download_etm1(data_path)
    frame = pd.read_csv(data_path)
    columns = [column for column in frame.columns if column != "date"]
    raw_values = frame[columns].to_numpy(dtype=np.float64)
    target_index = columns.index(str(config.get("target", "OT")))
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    train_end = 12 * 30 * 24 * 4
    val_end = train_end + 4 * 30 * 24 * 4
    test_end = train_end + 8 * 30 * 24 * 4
    if len(raw_values) < test_end:
        raise RuntimeError("ETTm1 file is shorter than the frozen official split")

    scaler = StandardScaler().fit(raw_values[:train_end])
    values = scaler.transform(raw_values).astype(np.float32)
    train_indices_all = _window_indices(0, train_end, seq_len, pred_len)
    val_indices_all = _window_indices(train_end - seq_len, val_end, seq_len, pred_len)
    test_indices_all = _window_indices(val_end - seq_len, test_end, seq_len, pred_len)
    train_indices = _take_indices(train_indices_all, int(config["train_windows"]), model_seed + 11)
    train_x, train_y = _make_windows(
        values,
        train_indices,
        seq_len=seq_len,
        pred_len=pred_len,
        target_index=target_index,
    )

    model = _build_model(
        torch,
        architecture=architecture,
        channels=values.shape[1],
        seq_len=seq_len,
        pred_len=pred_len,
        hidden=int(config["hidden_dimension"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    batch_size = int(config["batch_size"])
    rng = np.random.default_rng(model_seed + 23)
    model.train()
    for _ in range(int(config["epochs"])):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            x = torch.from_numpy(train_x[chosen])
            y = torch.from_numpy(train_y[chosen])
            optimizer.zero_grad(set_to_none=True)
            forecast, _ = model(x)
            loss = torch.nn.functional.mse_loss(forecast, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    support_indices = _take_indices(
        train_indices_all,
        int(config["support_windows"]),
        model_seed + 31,
    )
    calibration_indices = _take_indices(
        val_indices_all,
        int(config["calibration_windows"]),
        model_seed + 37,
    )
    evaluation_indices = _take_indices(
        test_indices_all,
        int(config["evaluation_windows"]),
        model_seed + 41,
    )
    support_x, _ = _make_windows(
        values,
        support_indices,
        seq_len=seq_len,
        pred_len=pred_len,
        target_index=target_index,
    )
    calibration_x, calibration_y = _make_windows(
        values,
        calibration_indices,
        seq_len=seq_len,
        pred_len=pred_len,
        target_index=target_index,
    )
    evaluation_x, evaluation_y = _make_windows(
        values,
        evaluation_indices,
        seq_len=seq_len,
        pred_len=pred_len,
        target_index=target_index,
    )
    support_forecast, support_hidden_raw = _encode_numpy(
        torch, model, support_x, batch_size=batch_size
    )
    calibration_forecast, calibration_hidden_raw = _encode_numpy(
        torch, model, calibration_x, batch_size=batch_size
    )
    evaluation_forecast, evaluation_hidden_raw = _encode_numpy(
        torch, model, evaluation_x, batch_size=batch_size
    )
    hidden_scaler = StandardScaler().fit(support_hidden_raw)
    support_hidden = hidden_scaler.transform(support_hidden_raw)
    calibration_hidden = hidden_scaler.transform(calibration_hidden_raw)
    evaluation_hidden = hidden_scaler.transform(evaluation_hidden_raw)

    descriptor_scaler = StandardScaler().fit(time_control_descriptors(support_x))
    support_controls = descriptor_scaler.transform(time_control_descriptors(support_x))
    calibration_controls = descriptor_scaler.transform(time_control_descriptors(calibration_x))
    control_head = Ridge(alpha=float(config.get("control_ridge_alpha", 1.0))).fit(
        support_hidden,
        support_controls,
    )
    control_r2 = float(control_head.score(calibration_hidden, calibration_controls))
    coefficients = np.asarray(control_head.coef_, dtype=float)

    anchor_count = min(int(config["anchors"]), len(calibration_x))
    candidate_count = min(int(config["candidates"]), len(evaluation_x))
    anchor_indices_local = np.linspace(0, len(calibration_x) - 1, anchor_count, dtype=int)
    candidate_indices_local = np.linspace(0, len(evaluation_x) - 1, candidate_count, dtype=int)
    anchor_hidden = calibration_hidden[anchor_indices_local]
    candidate_hidden = evaluation_hidden[candidate_indices_local]
    anchor_behavior = calibration_forecast[anchor_indices_local]
    candidate_behavior = evaluation_forecast[candidate_indices_local]

    rows: list[dict[str, Any]] = []
    for visible_rank in (int(value) for value in config["visible_ranks"]):
        projection = control_projection_from_head(coefficients, visible_rank)
        standardized = standardized_projection(projection, support_hidden)
        benign_control = control_output(calibration_hidden, standardized)
        benign_residual = residual_hidden(calibration_hidden, projection)
        nearest = _nearest_indices(benign_control)
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
                np.linalg.norm(calibration_forecast - calibration_forecast[nearest], axis=1)
                / sqrt(pred_len),
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
        behavior_distances = pairwise_distances(candidate_behavior, anchor_behavior) / sqrt(pred_len)
        valid_pairs = np.ones_like(control_distances, dtype=bool)
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

    evaluation_error = evaluation_forecast - evaluation_y
    return {
        "artifact_type": "qcollide_time_functional_collision_atlas_component",
        "schema_version": 1,
        "domain": "time_series",
        "dataset": "ETTm1",
        "architecture": architecture,
        "model_seed": model_seed,
        "model_diagnostics": {
            "forecast_mse": float(np.mean(evaluation_error**2)),
            "forecast_mae": float(np.mean(np.abs(evaluation_error))),
            "control_head_calibration_r2": control_r2,
            "hidden_dimension": int(support_hidden.shape[1]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "sample_counts": {
            "train_windows": int(len(train_x)),
            "support_windows": int(len(support_x)),
            "calibration_windows": int(len(calibration_x)),
            "evaluation_windows": int(len(evaluation_x)),
            "anchors": int(anchor_count),
            "candidates": int(candidate_count),
        },
        "rows": rows,
        "claim_boundary": {
            "standard_ett_minute_split": True,
            "compact_reimplementation_not_official_checkpoint": True,
            "control_descriptors_use_input_window_only": True,
            "thresholds_calibrated_on_validation_windows_only": True,
            "quantum_execution": False,
        },
    }


__all__ = ["run_time_atlas_component", "time_control_descriptors"]
