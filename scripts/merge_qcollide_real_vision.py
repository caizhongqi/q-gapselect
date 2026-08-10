#!/usr/bin/env python3
"""Merge independently executed Q-COLLIDE vision components."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qgapselect.qcollide.vision import (
    compact_real_vision_summary,
    merge_real_vision_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_real_vision_digits.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = sorted(args.input_directory.glob("*.json"))
    if not paths:
        raise RuntimeError("no component JSON files were found")
    components = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    artifact = merge_real_vision_artifacts(
        components,
        packing_target=float(config["packing_target"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = compact_real_vision_summary(artifact)
    summary["full_artifact_sha256"] = digest
    summary["full_artifact_bytes"] = args.output.stat().st_size
    summary["component_files"] = [path.name for path in paths]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "component_count": len(paths),
                "full_artifact_sha256": digest,
                "output": str(args.output),
                "summary": str(args.summary),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
