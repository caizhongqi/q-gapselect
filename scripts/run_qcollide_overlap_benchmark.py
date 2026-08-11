#!/usr/bin/env python3
"""Benchmark collision-capacity sketches from disjoint to dense overlap."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from qgapselect.qcollide.overlap_bounds import (
    bipartite_capacity_envelope,
    survival_capacity_confidence_interval,
    survival_probability_envelope,
)
from qgapselect.qcollide.overlap_capacity import (
    critical_retention_probability,
    estimate_overlap_capacity,
    make_dense_graph,
    make_hub_graph,
    make_matching_graph,
    make_sparse_overlap_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qcollide_overlap_benchmark.json"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--matching-size", type=int, default=16)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--failure-probability", type=float, default=1e-3)
    args = parser.parse_args()

    graphs = [
        make_matching_graph(args.n, args.matching_size, args.seed),
        make_sparse_overlap_graph(args.n, args.matching_size, 4, args.seed + 1),
        make_hub_graph(args.n, min(args.n, 4 * args.matching_size), args.seed + 2),
        make_dense_graph(args.n, 0.08, args.seed + 3),
    ]
    records = []
    for graph in graphs:
        q = critical_retention_probability(max(1, graph.matching_size))
        deterministic = bipartite_capacity_envelope(graph)
        population = survival_probability_envelope(
            graph.matching_size,
            graph.maximum_degree,
            q,
        )
        for degree_aware in (False, True):
            estimate = estimate_overlap_capacity(
                graph,
                target_q=q,
                trials=args.trials,
                seed=args.seed + 101 * len(records),
                degree_aware=degree_aware,
            )
            record = {
                **asdict(estimate),
                "degree_aware": degree_aware,
                "target_q": q,
                "deterministic_capacity_envelope": {
                    "lower": deterministic.lower_bound,
                    "upper": deterministic.upper_bound,
                },
                "uniform_survival_population_envelope": {
                    "lower": population.lower_survival,
                    "upper": population.upper_survival,
                },
            }
            if not degree_aware:
                successes = round(estimate.survival_probability * args.trials)
                certificate = survival_capacity_confidence_interval(
                    successes,
                    args.trials,
                    retention_probability=q,
                    maximum_degree=graph.maximum_degree,
                    failure_probability=args.failure_probability,
                    domain_cap=min(graph.n_left, graph.n_right),
                )
                record["uniform_survival_capacity_certificate"] = asdict(certificate)
            records.append(record)

    payload = {
        "artifact_type": "qcollide_overlap_capacity_benchmark",
        "schema_version": 2,
        "n": args.n,
        "requested_matching_size": args.matching_size,
        "trials": args.trials,
        "failure_probability": args.failure_probability,
        "records": records,
        "claim_boundary": {
            "packed_matching_inversion_exact": True,
            "general_overlap_point_estimator_proved": False,
            "uniform_survival_capacity_interval_proved": True,
            "arbitrary_overlap_deterministic_envelope_proved": True,
            "degree_aware_scaled_matching_is_empirical": True,
            "edge_count_is_capacity": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
