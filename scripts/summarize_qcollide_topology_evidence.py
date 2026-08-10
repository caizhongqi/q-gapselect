#!/usr/bin/env python3
"""Build the partial legacy collision-topology evidence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.topology_evidence import build_topology_evidence_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digits",
        type=Path,
        default=Path("artifacts/qcollide_real_vision_summary.json"),
    )
    parser.add_argument(
        "--fashion",
        type=Path,
        default=Path("artifacts/qcollide_fashion_mnist_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_topology_evidence_audit.json"),
    )
    args = parser.parse_args()
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.digits, args.fashion)
    ]
    audit = build_topology_evidence_audit(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["associations"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
