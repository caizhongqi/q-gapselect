"""Trainable synthetic dual-head model used by the causal experiment."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class DualHeadModel:
    """One-hidden-layer tanh network with control and behaviour heads."""

    representation_weight: np.ndarray
    representation_bias: np.ndarray
    control_head: np.ndarray
    main_head: np.ndarray
    latent_scales: np.ndarray
    main_coefficients: np.ndarray

    @property
    def latent_dimension(self) -> int:
        return int(self.representation_weight.shape[0])

    @property
    def hidden_dimension(self) -> int:
        return int(self.representation_weight.shape[1])

    def hidden(self, latent: np.ndarray) -> np.ndarray:
        values = np.asarray(latent, dtype=float)
        inputs = values * self.latent_scales
        return np.tanh(inputs @ self.representation_weight + self.representation_bias)

    def full_control(self, latent: np.ndarray) -> np.ndarray:
        return self.hidden(latent) @ self.control_head

    def main(self, latent: np.ndarray) -> np.ndarray:
        return self.hidden(latent) @ self.main_head

    def hidden_jacobian(self, latent: np.ndarray) -> np.ndarray:
        point = np.asarray(latent, dtype=float)
        if point.ndim != 1 or point.shape[0] != self.latent_dimension:
            raise ValueError("latent must be one point with the configured dimension")
        hidden = self.hidden(point)
        return (
            (1.0 - hidden * hidden)[:, None]
            * self.representation_weight.T
            * self.latent_scales[None, :]
        )


def _seed(master_seed: int, *parts: object) -> int:
    payload = "\0".join(("qcollide-dual-head-v1", str(master_seed), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _model_seeds(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("model_seeds must be a sequence")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"model_seeds[{index}] must be a non-negative integer")
        result.append(item)
    if not result:
        raise ValueError("model_seeds must be non-empty")
    return tuple(result)


def train_dual_head_model(
    *,
    seed: int,
    latent_dimension: int,
    hidden_dimension: int,
    training_samples: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    minimum_latent_scale: float,
) -> tuple[DualHeadModel, dict[str, float | int]]:
    """Train the synthetic network with deterministic minibatch Adam."""

    if hidden_dimension < latent_dimension:
        raise ValueError("hidden_dimension must be at least latent_dimension")
    if batch_size > training_samples:
        raise ValueError("batch_size must not exceed training_samples")
    if not 0.0 < minimum_latent_scale <= 1.0:
        raise ValueError("minimum_latent_scale must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    scales = np.geomspace(1.0, minimum_latent_scale, latent_dimension)
    latent = rng.uniform(-1.0, 1.0, size=(training_samples, latent_dimension))
    inputs = latent * scales
    target_control = inputs
    main_coefficients = 1.0 / scales
    main_coefficients /= np.linalg.norm(main_coefficients)
    target_main = target_control @ main_coefficients

    representation_weight = rng.normal(
        scale=1.0 / sqrt(latent_dimension),
        size=(latent_dimension, hidden_dimension),
    )
    representation_bias = np.zeros(hidden_dimension, dtype=float)
    control_head = rng.normal(scale=0.1, size=(hidden_dimension, latent_dimension))
    main_head = rng.normal(scale=0.1, size=hidden_dimension)
    parameters = [
        representation_weight,
        representation_bias,
        control_head,
        main_head,
    ]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    beta1 = 0.9
    beta2 = 0.999

    for step in range(1, training_steps + 1):
        indices = rng.integers(0, training_samples, size=batch_size)
        batch_inputs = inputs[indices]
        batch_control = target_control[indices]
        batch_main = target_main[indices]
        hidden = np.tanh(batch_inputs @ representation_weight + representation_bias)
        predicted_control = hidden @ control_head
        predicted_main = hidden @ main_head
        control_gradient = 2.0 * (predicted_control - batch_control) / batch_size
        main_gradient = 2.0 * (predicted_main - batch_main) / batch_size
        control_head_gradient = hidden.T @ control_gradient
        main_head_gradient = hidden.T @ main_gradient
        hidden_gradient = (
            control_gradient @ control_head.T
            + main_gradient[:, None] * main_head[None, :]
        )
        activation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        gradients = [
            batch_inputs.T @ activation_gradient,
            activation_gradient.sum(axis=0),
            control_head_gradient,
            main_head_gradient,
        ]
        for index, (parameter, gradient) in enumerate(
            zip(parameters, gradients, strict=True)
        ):
            first_moments[index] = (
                beta1 * first_moments[index] + (1.0 - beta1) * gradient
            )
            second_moments[index] = (
                beta2 * second_moments[index] + (1.0 - beta2) * gradient * gradient
            )
            corrected_first = first_moments[index] / (1.0 - beta1**step)
            corrected_second = second_moments[index] / (1.0 - beta2**step)
            parameter -= (
                learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + 1e-8)
            )

    model = DualHeadModel(
        representation_weight,
        representation_bias,
        control_head,
        main_head,
        scales,
        main_coefficients,
    )
    validation = rng.uniform(-1.0, 1.0, size=(2048, latent_dimension))
    validation_control = validation * scales
    validation_main = validation_control @ main_coefficients
    diagnostics: dict[str, float | int] = {
        "control_mse": float(
            np.mean((model.full_control(validation) - validation_control) ** 2)
        ),
        "main_mse": float(np.mean((model.main(validation) - validation_main) ** 2)),
        "training_steps": training_steps,
        "training_samples": training_samples,
    }
    return model, diagnostics


__all__ = ["DualHeadModel", "train_dual_head_model"]
