"""F3 weighted prefix-claw scaling panel."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .costs import (
    prefix_rms_cost,
    prefix_survival_rates,
    product_johnson_cost,
    weighted_qcollide_cost,
)
from .fixtures import weighted_prefix_claw
from .graph import packing_statistics
from .scaling_common import (
    derived_seed,
    float_values,
    int_values,
    mapping,
    positive_int,
    power_count,
    sequence,
)


def run_f3(config: Mapping[str, object], master_seed: int) -> dict[str, object]:
    n_values = int_values(config.get("n_values"), "f3.n_values")
    exponents = float_values(config.get("matching_exponents"), "f3.matching_exponents")
    repetitions = positive_int(config.get("repetitions"), "f3.repetitions")
    schedules = sequence(config.get("schedules"), "f3.schedules")
    rows: list[dict[str, object]] = []
    for schedule_index, raw_schedule in enumerate(schedules):
        schedule = mapping(raw_schedule, f"f3.schedules[{schedule_index}]")
        name = str(schedule.get("name"))
        cumulative_bits = int_values(
            schedule.get("cumulative_bits"),
            f"f3.schedules[{schedule_index}].cumulative_bits",
        )
        stage_costs = float_values(
            schedule.get("stage_costs"),
            f"f3.schedules[{schedule_index}].stage_costs",
        )
        if len(cumulative_bits) != len(stage_costs):
            raise ValueError(f"F3 schedule {name!r} has misaligned levels and costs")
        for n in n_values:
            if (1 << cumulative_bits[-1]) < 4 * n:
                raise ValueError(f"F3 final signature range is too small for n={n}")
            for exponent in exponents:
                matching_size = power_count(n, exponent, maximum=n)
                for repetition in range(repetitions):
                    instance = weighted_prefix_claw(
                        n=n,
                        matching_size=matching_size,
                        cumulative_bits=cumulative_bits,
                        stage_costs=stage_costs,
                        seed=derived_seed(
                            master_seed,
                            "F3",
                            name,
                            n,
                            exponent,
                            repetition,
                        ),
                    )
                    stats = packing_statistics(instance)
                    survival = prefix_survival_rates(
                        (record.prefixes for record in instance.left),
                        (record.prefixes for record in instance.right),
                    )
                    rms = prefix_rms_cost(stage_costs, survival)
                    max_cost = sum(stage_costs)
                    rows.append(
                        {
                            "schedule": name,
                            "n": n,
                            "matching_exponent": exponent,
                            "matching_size": stats.matching_size,
                            "survival_rates": list(survival),
                            "rms_endpoint_cost": rms,
                            "max_endpoint_cost": max_cost,
                            "max_to_rms_ratio": max_cost / rms,
                            "weighted_qcollide_cost": weighted_qcollide_cost(
                                n,
                                stats.matching_size,
                                rms,
                            ),
                            "uniform_cost_walk": max_cost
                            * product_johnson_cost(n, stats.matching_size).total_cost,
                            "fingerprint": instance.metadata["fingerprint"],
                        }
                    )
    schedule_summaries: list[dict[str, object]] = []
    for raw_schedule in schedules:
        schedule = mapping(raw_schedule, "f3.schedule")
        name = str(schedule.get("name"))
        cell = [row for row in rows if row["schedule"] == name]
        ratios = np.asarray([row["max_to_rms_ratio"] for row in cell], dtype=float)
        schedule_summaries.append(
            {
                "schedule": name,
                "mean_max_to_rms_ratio": float(ratios.mean()),
                "median_max_to_rms_ratio": float(np.median(ratios)),
                "minimum_max_to_rms_ratio": float(ratios.min()),
                "maximum_max_to_rms_ratio": float(ratios.max()),
            }
        )
    return {
        "fixture": "F3",
        "semantics": "weighted prefix survival and RMS endpoint cost",
        "rows": rows,
        "schedule_summaries": schedule_summaries,
    }
