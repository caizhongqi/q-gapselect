from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_uci_downloader_lists_only_official_cc_by_sources() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "download_uci_benchmarks.py"),
            "--list",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    assert set(document) == {"letter", "optdigits", "covertype"}
    assert all(row["license"] == "CC BY 4.0" for row in document.values())
    assert all(row["url"].startswith("https://archive.ics.uci.edu/") for row in document.values())


def test_uci_downloader_rejects_unknown_dataset_without_network_access() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "download_uci_benchmarks.py"),
            "not-a-dataset",
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "invalid choice: 'not-a-dataset'" in completed.stderr
