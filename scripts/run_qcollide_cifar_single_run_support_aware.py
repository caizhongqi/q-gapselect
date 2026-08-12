#!/usr/bin/env python3
"""Run CIFAR-100 single-run matching with support-aware topology sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar100_prepared import install_prepared_cifar100_adapter
from qgapselect.qcollide import cifar_single_run_selected as selected_module
from qgapselect.qcollide.cifar_support_sampling import support_aware_correct_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    target_count = int(config["topology_samples_per_side"])
    minimum_supported = int(config["minimum_supported_classes"])
    support_metadata: list[dict[str, object]] = []

    def support_sampler(logits, labels, *, class_count, per_class):
        expected_count = int(class_count) * int(per_class)
        if expected_count != target_count:
            raise RuntimeError(
                "anchors_per_class * class_count must equal topology_samples_per_side"
            )
        indices, metadata = support_aware_correct_indices(
            logits,
            labels,
            class_count=int(class_count),
            target_count=target_count,
            minimum_supported_classes=minimum_supported,
        )
        support_metadata.append(metadata)
        return indices

    install_prepared_cifar100_adapter()
    selected_module._stratified_correct_indices = support_sampler
    artifact = selected_module.run_cifar_single_run_selected_component(
        config,
        architecture=args.architecture,
        model_seed=args.model_seed,
    )
    if len(support_metadata) != 2:
        raise RuntimeError("support-aware sampler must be called for both topology sides")
    artifact["experiment_config"] = str(args.config)
    artifact["dataset_backend"] = "prepared:huggingface:uoft-cs/cifar100"
    artifact["selection"]["topology_sampling_mode"] = "coverage_first_correct_support"
    artifact["selection"]["calibration_support"] = support_metadata[0]
    artifact["selection"]["evaluation_support"] = support_metadata[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "architecture": args.architecture,
                "model_seed": args.model_seed,
                "selected_checkpoint_epoch": artifact["selected_checkpoint_epoch"],
                "selected_performance": artifact["selected_performance"],
                "calibration_supported_classes": support_metadata[0][
                    "supported_class_count"
                ],
                "evaluation_supported_classes": support_metadata[1][
                    "supported_class_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
