"""Shared validation, deterministic seeding, and scaling regression utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from math import isfinite, log

import numpy as np


def derived_seed(master_seed: int, *parts: object) -> int:
    payload = "\0".join(("qcollide-scaling-v1", str(master_seed), *map(str, parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def int_values(value: object, name: str) -> tuple[int, ...]:
    result = tuple(
        positive_int(item, f"{name}[{index}]")
        for index, item in enumerate(sequence(value, name))
    )
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def float_values(value: object, name: str) -> tuple[float, ...]:
    result = tuple(
        number(item, f"{name}[{index}]")
        for index, item in enumerate(sequence(value, name))
    )
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def fit_log_linear(
    rows: Sequence[Mapping[str, object]],
    *,
    response: str,
    predictors: tuple[str, ...],
) -> dict[str, object]:
    """Fit log(response) on an intercept and log predictors by least squares."""

    if not rows:
        raise ValueError("rows must be non-empty")
    design: list[list[float]] = []
    target: list[float] = []
    for row_index, row in enumerate(rows):
        y = number(row.get(response), f"rows[{row_index}].{response}")
        if y <= 0.0:
            raise ValueError(f"rows[{row_index}].{response} must be positive")
        values = [1.0]
        for predictor in predictors:
            x = number(row.get(predictor), f"rows[{row_index}].{predictor}")
            if x <= 0.0:
                raise ValueError(f"rows[{row_index}].{predictor} must be positive")
            values.append(log(x))
        design.append(values)
        target.append(log(y))

    matrix = np.asarray(design, dtype=float)
    vector = np.asarray(target, dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    fitted = matrix @ coefficients
    residual = vector - fitted
    total = vector - vector.mean()
    residual_sum = float(residual @ residual)
    total_sum = float(total @ total)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    return {
        "response": response,
        "predictors": list(predictors),
        "intercept": float(coefficients[0]),
        "coefficients": {
            predictor: float(coefficients[index + 1])
            for index, predictor in enumerate(predictors)
        },
        "r_squared": float(r_squared),
        "sample_count": len(rows),
        "design_rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
    }


def power_count(n: int, exponent: float, *, maximum: int) -> int:
    return max(1, min(maximum, int(round(n**exponent))))
