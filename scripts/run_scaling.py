#!/usr/bin/env python3
"""Emit the preregistered analytic scaling suite with machine provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from qgapselect.experiments import canonical_scaling_suite, fit_loglog_slopes

WARNING = (
    "Values are computed analytic complexity proxies, not observed quantum "
    "query counts, wall-clock speedups, or activated theorem results."
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY / "configs" / "scaling.json"
DEFAULTS: dict[str, object] = {
    "seed": 1729,
    "sizes": [8, 16, 32, 64, 128],
    "mean_gap": 0.125,
    "spread": 4.0,
    "regimes": [
        "best_arm",
        "equal_gap_topk",
        "heterogeneous_topk",
        "dense_output",
        "partition_direct_sum",
    ],
}
ALLOWED_CONFIG_FIELDS = {
    "seed",
    "sizes",
    "mean_gap",
    "spread",
    "regimes",
    # Retained in provenance but deliberately unused by an analytic-only run.
    "trials",
    "delta",
    "notes",
}
ALLOWED_REGIMES = {
    "best_arm",
    "equal_gap_topk",
    "heterogeneous_topk",
    "dense_output",
    "partition_direct_sum",
}


@dataclass(frozen=True, slots=True)
class ResolvedRunConfig:
    sizes: tuple[int, ...]
    mean_gap: float
    spread: float
    seed: int
    regimes: tuple[str, ...]
    source_path: str
    source_sha256: str
    source_document: dict[str, object]

    def executable_dict(self) -> dict[str, object]:
        return {
            "sizes": list(self.sizes),
            "mean_gap": self.mean_gap,
            "spread": self.spread,
            "seed": self.seed,
            "regimes": list(self.regimes),
        }


def _sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one size is required")
    if any(item < 2 or item % 2 for item in result):
        raise argparse.ArgumentTypeError("sizes must be even integers >= 2")
    return result


def _regimes(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(result) - ALLOWED_REGIMES
    if not result:
        raise argparse.ArgumentTypeError("at least one regime is required")
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown regimes: " + ", ".join(sorted(unknown))
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON experiment config (default: configs/scaling.json)",
    )
    # None is essential: only explicitly supplied CLI values override JSON.
    parser.add_argument("--sizes", type=_sizes)
    parser.add_argument(
        "--mean-gap",
        type=float,
        help="boundary gap in Bernoulli mean (angular gaps are derived and recorded)",
    )
    parser.add_argument("--spread", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--regimes", type=_regimes)
    parser.add_argument("--format", choices=("json", "jsonl", "csv"), default="json")
    parser.add_argument("--output", type=Path, help="write here instead of stdout")
    parser.add_argument(
        "--omit-slopes",
        action="store_true",
        help="omit descriptive log-log fits from JSON output",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> ResolvedRunConfig:
    raw = args.config.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON config {args.config}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("the scaling config must be a JSON object")
    unknown = set(document) - ALLOWED_CONFIG_FIELDS
    if unknown:
        raise ValueError("unknown config fields: " + ", ".join(sorted(unknown)))

    merged = dict(DEFAULTS)
    merged.update(document)
    if args.sizes is not None:
        merged["sizes"] = list(args.sizes)
    if args.mean_gap is not None:
        merged["mean_gap"] = args.mean_gap
    if args.spread is not None:
        merged["spread"] = args.spread
    if args.seed is not None:
        merged["seed"] = args.seed
    if args.regimes is not None:
        merged["regimes"] = list(args.regimes)

    sizes = _sizes(",".join(str(item) for item in merged["sizes"]))
    regimes = _regimes(",".join(str(item) for item in merged["regimes"]))
    mean_gap = float(merged["mean_gap"])
    spread = float(merged["spread"])
    if not 0.0 < mean_gap <= 1.0:
        raise ValueError("resolved mean_gap must lie in (0, 1]")
    if spread < 1.0:
        raise ValueError("resolved spread must be at least one")
    resolved_source = args.config.resolve()
    try:
        portable_source = resolved_source.relative_to(REPOSITORY).as_posix()
    except ValueError:
        portable_source = str(resolved_source)
    return ResolvedRunConfig(
        sizes=sizes,
        mean_gap=mean_gap,
        spread=spread,
        seed=int(merged["seed"]),
        regimes=regimes,
        source_path=portable_source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_document=document,
    )


def _provenance(config: ResolvedRunConfig) -> dict[str, object]:
    unused = {
        key: config.source_document[key]
        for key in ("trials", "delta")
        if key in config.source_document
    }
    return {
        "generator": "scripts/run_scaling.py",
        "schema_version": 1,
        "seed": config.seed,
        "config_source": config.source_path,
        "config_sha256": config.source_sha256,
        "source_config": config.source_document,
        "resolved_executable_config": config.executable_dict(),
        "declared_but_unused_by_analytic_run": unused,
        "claim_boundary": WARNING,
    }


def _include_scenario(scenario: str, regimes: tuple[str, ...]) -> bool:
    allowed: set[str] = set()
    if "best_arm" in regimes:
        allowed.update(("equal_gap_k1", "heterogeneous_k1"))
    if "equal_gap_topk" in regimes or "dense_output" in regimes:
        allowed.add("equal_gap_half")
    if "heterogeneous_topk" in regimes or "dense_output" in regimes:
        allowed.add("heterogeneous_half")
    if "partition_direct_sum" in regimes and scenario.startswith("partition_"):
        return True
    return scenario in allowed


def _records(config: ResolvedRunConfig):
    return tuple(
        record
        for record in canonical_scaling_suite(
            config.sizes,
            gap=config.mean_gap,
            spread=config.spread,
            seed=config.seed,
        )
        if _include_scenario(record.scenario, config.regimes)
    )


def _json_document(
    args: argparse.Namespace, config: ResolvedRunConfig
) -> str:
    records = _records(config)
    document: dict[str, object] = {
        "artifact_type": "analytic_complexity_proxy_table",
        "provenance": _provenance(config),
        "records": [record.as_dict() for record in records],
    }
    if not args.omit_slopes:
        document["descriptive_slopes"] = [
            {
                "scenario": slope.scenario,
                "method": slope.method,
                "observations": slope.observations,
                "slope": slope.slope,
                "intercept": slope.intercept,
                "claim_status": slope.claim_status,
            }
            for slope in fit_loglog_slopes(records)
        ]
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _json_lines(args: argparse.Namespace, config: ResolvedRunConfig) -> str:
    provenance = _provenance(config)
    records = _records(config)
    lines = [
        json.dumps(
            {
                "record_type": "provenance",
                **provenance,
            },
            sort_keys=True,
        )
    ]
    lines.extend(
        json.dumps(
            {
                "record_type": "complexity_proxy",
                "provenance_seed": config.seed,
                **record.as_dict(),
            },
            sort_keys=True,
        )
        for record in records
    )
    if not args.omit_slopes:
        lines.extend(
            json.dumps(
                {
                    "record_type": "descriptive_slope",
                    "provenance_seed": config.seed,
                    "scenario": slope.scenario,
                    "method": slope.method,
                    "observations": slope.observations,
                    "slope": slope.slope,
                    "intercept": slope.intercept,
                    "claim_status": slope.claim_status,
                },
                sort_keys=True,
            )
            for slope in fit_loglog_slopes(records)
        )
    return "\n".join(lines) + "\n"


def _csv_document(args: argparse.Namespace, config: ResolvedRunConfig) -> str:
    records = _records(config)
    resolved_config = json.dumps(config.executable_dict(), sort_keys=True)
    fieldnames = (
        "scenario",
        "method",
        "n",
        "k",
        "output_size",
        "min_mean_gap",
        "max_mean_gap",
        "min_angular_gap",
        "max_angular_gap",
        "value",
        "unit",
        "claim_status",
        "data_source",
        "metadata",
        "provenance_seed",
        "provenance_config",
        "claim_boundary",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        row = record.as_dict()
        row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
        row["provenance_seed"] = config.seed
        row["provenance_config"] = resolved_config
        row["claim_boundary"] = WARNING
        writer.writerow(row)
    return buffer.getvalue()


def render(args: argparse.Namespace, config: ResolvedRunConfig) -> str:
    if args.format == "json":
        return _json_document(args, config)
    if args.format == "jsonl":
        return _json_lines(args, config)
    return _csv_document(args, config)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = resolve_config(args)
        payload = render(args, config)
    except (argparse.ArgumentTypeError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"configuration error: {error}") from error
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
