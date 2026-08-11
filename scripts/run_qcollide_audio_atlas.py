#!/usr/bin/env python3
"""Run one manifest-efficient Speech Commands collision-atlas component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.audio_experiment import run_audio_manifest_component
from qgapselect.qcollide.audio_pcm import patch_torchaudio_pcm_loader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--fixture-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    patch_torchaudio_pcm_loader()
    artifact = run_audio_manifest_component(
        config,
        architecture=args.architecture,
        fixture_seed=args.fixture_seed,
    )
    artifact["experiment_config"] = str(args.config)
    artifact["claim_boundary"]["audio_decode_backend"] = "stdlib_wave_pcm16"
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
                "fixture_seed": args.fixture_seed,
                "gates": artifact["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
