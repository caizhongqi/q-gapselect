"""Frozen multi-scale calibration campaign for Q-COLLIDE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .scaling_common import fit_log_linear, mapping, positive_int, sequence
from .scaling_geometry import run_f4
from .scaling_pair import run_f0, run_f1, run_f2
from .scaling_prefix import run_f3


def run_scaling_campaign(config: Mapping[str, object]) -> dict[str, object]:
    """Run F0--F4 while preserving the analytic-only claim boundary."""

    schema_version = positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("only schema_version=1 is supported")
    master_seed = positive_int(config.get("master_seed"), "master_seed")
    panels = {
        "F0": run_f0(mapping(config.get("f0"), "f0"), master_seed),
        "F1": run_f1(mapping(config.get("f1"), "f1"), master_seed),
        "F2": run_f2(mapping(config.get("f2"), "f2"), master_seed),
        "F3": run_f3(mapping(config.get("f3"), "f3"), master_seed),
        "F4": run_f4(mapping(config.get("f4"), "f4"), master_seed),
    }
    return {
        "artifact_type": "qcollide_f0_f4_scaling_diagnostic",
        "schema_version": schema_version,
        "master_seed": master_seed,
        "claim_scope": {
            "quantum_values_are": "analytic endpoint-query proxies",
            "statevector_execution": False,
            "compiled_circuit_execution": False,
            "hardware_execution": False,
            "free_qram_assumed": False,
            "new_lower_bound_claimed": False,
            "neural_causal_claimed": False,
        },
        "panels": panels,
        "record_counts": {
            name: len(panel["rows"])
            for name, panel in panels.items()
        },
    }


def compact_summary(artifact: Mapping[str, object]) -> dict[str, Any]:
    panels = mapping(artifact.get("panels"), "artifact.panels")
    f1 = mapping(panels["F1"], "artifact.panels.F1")
    f3 = mapping(panels["F3"], "artifact.panels.F3")
    f4 = mapping(panels["F4"], "artifact.panels.F4")
    f4_rows = sequence(f4["rows"], "artifact.panels.F4.rows")
    return {
        "record_counts": dict(mapping(artifact["record_counts"], "record_counts")),
        "f1_joint_fits": dict(mapping(f1["joint_fits"], "F1.joint_fits")),
        "f3_schedule_summaries": list(
            sequence(f3["schedule_summaries"], "F3.schedule_summaries")
        ),
        "f4_mean_packing_fraction": float(
            np.mean(
                [
                    float(mapping(row, "F4.row")["packing_fraction"])
                    for row in f4_rows
                ]
            )
        ),
    }


__all__ = ["compact_summary", "fit_log_linear", "run_scaling_campaign"]
