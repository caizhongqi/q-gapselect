#!/usr/bin/env python3
"""Run the Q-COLLIDE dangerous-direction closure-spectrum campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.closure_spectrum import (
    compact_closure_spectrum_summary,
    run_closure_spectrum_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_closure_spectrum.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_closure_spectrum_diagnostic.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_closure_spectrum_campaign(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact_closure_spectrum_summary(artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
