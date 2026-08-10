#!/usr/bin/env python3
"""Run the trainable synthetic Q-COLLIDE dual-head causal campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.dual_head import (
    compact_dual_head_summary,
    run_dual_head_causal_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_dual_head.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_dual_head_diagnostic.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_dual_head_causal_campaign(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact_dual_head_summary(artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
