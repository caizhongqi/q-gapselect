#!/usr/bin/env python3
"""Download and safely extract the preregistered official UCI datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "data" / "uci"
DATASETS = {
    "letter": {
        "url": "https://archive.ics.uci.edu/static/public/59/letter%2Brecognition.zip",
        "landing_page": "https://archive.ics.uci.edu/dataset/59/letter%2Brecognition",
        "license": "CC BY 4.0",
        "required_files": ("letter-recognition.data",),
    },
    "optdigits": {
        "url": (
            "https://archive.ics.uci.edu/static/public/80/"
            "optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip"
        ),
        "landing_page": (
            "https://archive.ics.uci.edu/dataset/80/"
            "optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits"
        ),
        "license": "CC BY 4.0",
        "required_files": ("optdigits.tra", "optdigits.tes"),
    },
    "covertype": {
        "url": "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        "landing_page": "https://archive.ics.uci.edu/dataset/31/covertype",
        "license": "CC BY 4.0",
        "required_files": ("covtype.data.gz",),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            target = (destination / member.filename).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe zip member {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(str(target.relative_to(destination)))
    return tuple(sorted(extracted))


def download_dataset(dataset_id: str, output: Path) -> dict[str, object]:
    metadata = DATASETS[dataset_id]
    directory = output / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "official.zip"
    request = urllib.request.Request(
        str(metadata["url"]),
        headers={"User-Agent": "q-gapselect-research-artifact/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    extracted = _safe_extract(archive, directory)
    missing = [
        name
        for name in metadata["required_files"]
        if not any(Path(item).name == name for item in extracted)
    ]
    if missing:
        raise RuntimeError(f"official archive is missing required files: {missing}")
    manifest = {
        "dataset_id": dataset_id,
        "url": metadata["url"],
        "landing_page": metadata["landing_page"],
        "license": metadata["license"],
        "archive_sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "required_files": list(metadata["required_files"]),
        "extracted_files": list(extracted),
        "source": "official_uci_machine_download",
    }
    (directory / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _dataset_id(value: str) -> str:
    if value not in DATASETS:
        choices = ", ".join(map(repr, DATASETS))
        raise argparse.ArgumentTypeError(
            f"invalid choice: {value!r} (choose from {choices})"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Python 3.10 rejects an empty ``nargs='*'`` positional when ``choices``
    # is set, even though no dataset value needs validation.  Validate only
    # supplied values through ``type`` so ``--list`` behaves identically on
    # every supported Python version.
    parser.add_argument(
        "datasets",
        nargs="*",
        type=_dataset_id,
        metavar="{" + ",".join(DATASETS) + "}",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        sys.stdout.write(json.dumps(DATASETS, indent=2, sort_keys=True) + "\n")
        return 0
    selected = tuple(args.datasets) or ("letter", "optdigits")
    manifests = [download_dataset(dataset_id, args.output) for dataset_id in selected]
    sys.stdout.write(
        f"downloaded {len(manifests)} official UCI datasets to {args.output}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
