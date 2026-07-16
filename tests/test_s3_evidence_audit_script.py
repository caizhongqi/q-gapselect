from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

from test_s3_evidence_audit import _documents


def _environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_s3_evidence_cli_reads_json_and_gzip_and_writes_audit(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    documents = _documents()
    names = ("adaptive", "coherent", "frontier", "composition")
    paths: dict[str, Path] = {}
    for name, document in zip(names, documents, strict=True):
        suffix = ".json.gz" if name == "composition" else ".json"
        path = tmp_path / f"{name}{suffix}"
        payload = (json.dumps(document, sort_keys=True) + "\n").encode()
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(payload, mtime=0))
        else:
            path.write_bytes(payload)
        paths[name] = path
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_s3_evidence_audit.py"),
            "--adaptive",
            str(paths["adaptive"]),
            "--coherent",
            str(paths["coherent"]),
            "--frontier",
            str(paths["frontier"]),
            "--composition",
            str(paths["composition"]),
            "--output",
            str(output),
        ),
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "q_gapselect_s3_integrated_evidence_audit"
    assert artifact["report"]["satisfied_gate_count"] == 1
    assert artifact["report"]["open_gate_count"] == 6
    assert artifact["report"]["satisfied_checkpoint_count"] == 1
    assert artifact["report"]["claim_gate_count"] == 6
    assert artifact["report"]["satisfied_claim_gate_count"] == 0
    assert artifact["report"]["ccf_a_claimable"] is False
    assert artifact["claim_boundary"]["independently_proves_theorems"] is False
    assert artifact["claim_boundary"]["independent_theorem_verifier_available"] is False
    assert artifact["claim_boundary"]["theorem_claim_activation_locked"] is True
    assert artifact["source_artifacts"]["composition"]["path"].endswith(".json.gz")
    assert "ccf_a_claimable=False" in completed.stdout


def test_s3_evidence_cli_rejects_abbreviated_arguments(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_s3_evidence_audit.py"),
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ),
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
