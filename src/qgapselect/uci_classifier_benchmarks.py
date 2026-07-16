"""Auditable UCI classifier-selection fixtures for external-validity studies.

This module turns predictions from a preregistered classifier catalogue into
frozen Bernoulli arms.  It is intentionally a *semi-synthetic* benchmark: the
features, labels, train/test split, and correctness bits are empirical, while
the arm-selection problem and its quantum query interface are experimental
constructions.  Consequently these fixtures can test external validity but do
not, by themselves, prove a quantum advantage.

The trusted harness owns test labels, correctness streams, empirical means,
the separating threshold, and Top-k truth.  An algorithm should receive only
the :class:`UCIAlgorithmFixture` produced by
:meth:`UCIClassifierBenchmark.open_algorithm_fixture`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .attack_oracles import (
    BlindSourceRewardOracle,
    FrozenCandidateGraph,
    FrozenSourceFixture,
    SourceCandidate,
    SourceOracleBudget,
    freeze_source_streams,
)
from .frozen_quantum_reference_benchmarking import FrozenQuantumReferenceInstance

CLAIM_SCOPE = "semi_synthetic_uci_external_validity_no_quantum_advantage_claim"
LICENSE_NAME = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
DEFAULT_SHARD_COUNT = 5
SKLEARN_DIGITS_OFFLINE_ID = "sklearn_digits_offline"


def _canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(document: object) -> str:
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest") from error
    return value.lower()


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _immutable_matrix(values: object, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if matrix.flags.c_contiguous and not matrix.flags.writeable:
        return matrix
    result = np.array(matrix, dtype=np.float64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _labels(values: Sequence[object], name: str) -> tuple[str, ...]:
    labels = tuple(str(value) for value in values)
    if not labels or any(not value for value in labels):
        raise ValueError(f"{name} must contain non-empty labels")
    return labels


@dataclass(frozen=True, slots=True)
class UCIDatasetSpec:
    """Preregistered identity and official split of one supported UCI dataset."""

    dataset_id: str
    display_name: str
    landing_url: str
    source_urls: tuple[str, ...]
    file_names: tuple[str, ...]
    feature_count: int
    train_rows: int
    test_rows: int
    split_strategy: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "landing_url": self.landing_url,
            "source_urls": list(self.source_urls),
            "file_names": list(self.file_names),
            "feature_count": self.feature_count,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "split_strategy": self.split_strategy,
        }


LETTER_SPEC = UCIDatasetSpec(
    dataset_id="letter_recognition",
    display_name="Letter Recognition",
    landing_url="https://archive.ics.uci.edu/dataset/59/letter+recognition",
    source_urls=(
        "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
    ),
    file_names=("letter-recognition.data",),
    feature_count=16,
    train_rows=16_000,
    test_rows=4_000,
    split_strategy="official_contiguous_first_16000_train_last_4000_test",
)

OPTDIGITS_SPEC = UCIDatasetSpec(
    dataset_id="optdigits",
    display_name="Optical Recognition of Handwritten Digits",
    landing_url=(
        "https://archive.ics.uci.edu/dataset/80/"
        "optical+recognition+of+handwritten+digits"
    ),
    source_urls=(
        "https://archive.ics.uci.edu/static/public/80/"
        "optical+recognition+of+handwritten+digits.zip",
        "https://archive.ics.uci.edu/static/public/80/"
        "optical+recognition+of+handwritten+digits.zip",
    ),
    file_names=("optdigits.tra", "optdigits.tes"),
    feature_count=64,
    train_rows=3_823,
    test_rows=1_797,
    split_strategy="official_predefined_train_and_test_files",
)

COVERTYPE_SPEC = UCIDatasetSpec(
    dataset_id="covertype",
    display_name="Covertype",
    landing_url="https://archive.ics.uci.edu/dataset/31/covertype",
    source_urls=("https://archive.ics.uci.edu/static/public/31/covertype.zip",),
    file_names=("covtype.data", "covtype.data.gz"),
    feature_count=54,
    train_rows=464_810,
    test_rows=116_202,
    split_strategy="non_official_fixed_stratified_hash_80_20_split_v1",
)

DATASET_SPECS: Mapping[str, UCIDatasetSpec] = MappingProxyType(
    {spec.dataset_id: spec for spec in (LETTER_SPEC, OPTDIGITS_SPEC, COVERTYPE_SPEC)}
)


@dataclass(frozen=True, slots=True)
class ParsedUCISource:
    """Strictly parsed local source file before split assignment."""

    path: str
    sha256: str
    byte_count: int
    features: NDArray[np.float64]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        path = _nonempty_string(self.path, "path")
        digest = _validate_sha256(self.sha256, "sha256")
        byte_count = _integer(self.byte_count, "byte_count", minimum=1)
        features = _immutable_matrix(self.features, "features")
        labels = _labels(self.labels, "labels")
        if len(labels) != features.shape[0]:
            raise ValueError("features and labels have different row counts")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True, slots=True)
class UCISourceRecord:
    """Source commitment included in every benchmark manifest."""

    role: str
    file_name: str
    sha256: str
    byte_count: int
    row_count: int
    source_url: str

    def __post_init__(self) -> None:
        _nonempty_string(self.role, "role")
        _nonempty_string(self.file_name, "file_name")
        _validate_sha256(self.sha256, "sha256")
        _integer(self.byte_count, "byte_count", minimum=1)
        _integer(self.row_count, "row_count", minimum=1)
        _nonempty_string(self.source_url, "source_url")

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "file_name": self.file_name,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "row_count": self.row_count,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class OfficialSplitMetadata:
    """Frozen official split description; no random resplitting is allowed."""

    strategy: str
    train_rows: int
    test_rows: int
    train_source_roles: tuple[str, ...]
    test_source_roles: tuple[str, ...]
    train_row_index_sha256: str | None = None
    test_row_index_sha256: str | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.strategy, "strategy")
        _integer(self.train_rows, "train_rows", minimum=1)
        _integer(self.test_rows, "test_rows", minimum=1)
        if self.train_row_index_sha256 is not None:
            _validate_sha256(self.train_row_index_sha256, "train_row_index_sha256")
        if self.test_row_index_sha256 is not None:
            _validate_sha256(self.test_row_index_sha256, "test_row_index_sha256")

    def as_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "strategy": self.strategy,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_source_roles": list(self.train_source_roles),
            "test_source_roles": list(self.test_source_roles),
        }
        if self.train_row_index_sha256 is not None:
            document["train_row_index_sha256"] = self.train_row_index_sha256
        if self.test_row_index_sha256 is not None:
            document["test_row_index_sha256"] = self.test_row_index_sha256
        return document


@dataclass(frozen=True, slots=True)
class LoadedUCIDataset:
    """Immutable train/test matrices plus auditable source and split metadata."""

    dataset_id: str
    train_features: NDArray[np.float64]
    train_labels: tuple[str, ...]
    test_features: NDArray[np.float64]
    test_labels: tuple[str, ...]
    sources: tuple[UCISourceRecord, ...]
    split: OfficialSplitMetadata
    landing_url: str
    license_name: str = LICENSE_NAME
    license_url: str = LICENSE_URL
    official_source: bool = True
    source_manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        dataset_id = _nonempty_string(self.dataset_id, "dataset_id")
        train_features = _immutable_matrix(self.train_features, "train_features")
        test_features = _immutable_matrix(self.test_features, "test_features")
        train_labels = _labels(self.train_labels, "train_labels")
        test_labels = _labels(self.test_labels, "test_labels")
        if train_features.shape[1] != test_features.shape[1]:
            raise ValueError("train and test feature dimensions differ")
        if len(train_labels) != train_features.shape[0]:
            raise ValueError("train features and labels have different row counts")
        if len(test_labels) != test_features.shape[0]:
            raise ValueError("test features and labels have different row counts")
        sources = tuple(self.sources)
        if not sources or any(not isinstance(item, UCISourceRecord) for item in sources):
            raise TypeError("sources must contain UCISourceRecord objects")
        if not isinstance(self.split, OfficialSplitMetadata):
            raise TypeError("split must be OfficialSplitMetadata")
        if self.split.train_rows != len(train_labels) or self.split.test_rows != len(test_labels):
            raise ValueError("split metadata row counts do not match matrices")
        _nonempty_string(self.landing_url, "landing_url")
        _nonempty_string(self.license_name, "license_name")
        _nonempty_string(self.license_url, "license_url")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "train_features", train_features)
        object.__setattr__(self, "test_features", test_features)
        object.__setattr__(self, "train_labels", train_labels)
        object.__setattr__(self, "test_labels", test_labels)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(
            self,
            "source_manifest_hash",
            _canonical_hash(self.source_document()),
        )

    def source_document(self) -> dict[str, object]:
        return {
            "schema": "qgapselect.uci-source-manifest.v1",
            "dataset_id": self.dataset_id,
            "official_source": self.official_source,
            "landing_url": self.landing_url,
            "license": {"name": self.license_name, "url": self.license_url},
            "sources": [source.as_dict() for source in self.sources],
            "split": self.split.as_dict(),
            "feature_count": int(self.train_features.shape[1]),
        }


def _read_ascii_rows(path: str | Path, *, expected_columns: int) -> tuple[bytes, list[list[str]]]:
    file_path = Path(path)
    payload = file_path.read_bytes()
    if not payload:
        raise ValueError(f"UCI source {file_path} is empty")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"UCI source {file_path} must be ASCII") from error
    if '"' in text:
        raise ValueError(f"quoted fields are not permitted in strict UCI source {file_path}")
    rows: list[list[str]] = []
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(field == "" for field in row):
                raise ValueError(f"blank row at line {line_number}")
            if len(row) != expected_columns:
                raise ValueError(
                    f"line {line_number} has {len(row)} columns; expected {expected_columns}"
                )
            if any(field != field.strip() or not field for field in row):
                raise ValueError(f"line {line_number} contains whitespace or an empty field")
            rows.append(row)
    except csv.Error as error:
        raise ValueError(f"malformed CSV in {file_path}: {error}") from error
    if not rows:
        raise ValueError(f"UCI source {file_path} contains no rows")
    return payload, rows


def _strict_integer(value: str, *, line_number: int, column_number: int) -> int:
    if not value.isdigit():
        raise ValueError(f"invalid integer at line {line_number}, column {column_number}")
    return int(value)


def parse_letter_source(path: str | Path) -> ParsedUCISource:
    """Strictly parse ``letter-recognition.data`` without assigning a split."""

    payload, rows = _read_ascii_rows(path, expected_columns=17)
    features = np.empty((len(rows), 16), dtype=np.float64)
    labels: list[str] = []
    for row_index, row in enumerate(rows):
        label = row[0]
        if len(label) != 1 or not "A" <= label <= "Z":
            raise ValueError(f"invalid Letter label at line {row_index + 1}")
        labels.append(label)
        for column, value in enumerate(row[1:], start=2):
            parsed = _strict_integer(
                value,
                line_number=row_index + 1,
                column_number=column,
            )
            if not 0 <= parsed <= 15:
                raise ValueError(
                    f"Letter feature outside [0, 15] at line {row_index + 1}, "
                    f"column {column}"
                )
            features[row_index, column - 2] = parsed
    return ParsedUCISource(
        path=str(Path(path)),
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
        features=features,
        labels=tuple(labels),
    )


def parse_optdigits_source(path: str | Path) -> ParsedUCISource:
    """Strictly parse one official Optdigits train or test file."""

    payload, rows = _read_ascii_rows(path, expected_columns=65)
    features = np.empty((len(rows), 64), dtype=np.float64)
    labels: list[str] = []
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row[:64], start=1):
            parsed = _strict_integer(
                value,
                line_number=row_index + 1,
                column_number=column,
            )
            if not 0 <= parsed <= 16:
                raise ValueError(
                    f"Optdigits feature outside [0, 16] at line {row_index + 1}, "
                    f"column {column}"
                )
            features[row_index, column - 1] = parsed
        label = _strict_integer(row[64], line_number=row_index + 1, column_number=65)
        if not 0 <= label <= 9:
            raise ValueError(f"invalid Optdigits label at line {row_index + 1}")
        labels.append(str(label))
    return ParsedUCISource(
        path=str(Path(path)),
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
        features=features,
        labels=tuple(labels),
    )


def _strict_signed_integer(value: str, *, line_number: int, column_number: int) -> int:
    digits = value[1:] if value.startswith("-") else value
    if not digits.isdigit() or value.startswith("+"):
        raise ValueError(f"invalid integer at line {line_number}, column {column_number}")
    return int(value)


def _file_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    if byte_count == 0:
        raise ValueError(f"UCI source {path} is empty")
    return digest.hexdigest(), byte_count


def parse_covertype_source(
    path: str | Path,
    *,
    expected_rows: int | None = None,
) -> ParsedUCISource:
    """Strictly and locally parse ``covtype.data`` or ``covtype.data.gz``.

    The parser streams decoded rows and never performs a network request.
    ``expected_rows`` enables preallocation and exact-count enforcement for the
    official 581,012-row loader, while allowing tiny parser fixtures in tests.
    """

    file_path = Path(path)
    if expected_rows is not None:
        expected_rows = _integer(expected_rows, "expected_rows", minimum=1)
    source_sha256, byte_count = _file_sha256_and_size(file_path)
    opener = gzip.open if file_path.suffix.lower() == ".gz" else open
    preallocated = (
        np.empty((expected_rows, COVERTYPE_SPEC.feature_count), dtype=np.float64)
        if expected_rows is not None
        else None
    )
    dynamic_rows: list[tuple[float, ...]] = []
    labels: list[str] = []
    try:
        with opener(file_path, mode="rt", encoding="ascii", newline="") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    raise ValueError(f"blank row at line {line_number}")
                if '"' in line:
                    raise ValueError(
                        f"quoted fields are not permitted at line {line_number}"
                    )
                fields = line.split(",")
                if len(fields) != 55:
                    raise ValueError(
                        f"line {line_number} has {len(fields)} columns; expected 55"
                    )
                if any(field != field.strip() or not field for field in fields):
                    raise ValueError(
                        f"line {line_number} contains whitespace or an empty field"
                    )
                values = tuple(
                    _strict_signed_integer(
                        value,
                        line_number=line_number,
                        column_number=column_number,
                    )
                    for column_number, value in enumerate(fields[:54], start=1)
                )
                binary = values[10:]
                if any(value not in {0, 1} for value in binary):
                    raise ValueError(
                        f"Covertype indicator outside {{0, 1}} at line {line_number}"
                    )
                if sum(values[10:14]) != 1:
                    raise ValueError(
                        f"Covertype wilderness indicators are not one-hot at line {line_number}"
                    )
                if sum(values[14:54]) != 1:
                    raise ValueError(
                        f"Covertype soil indicators are not one-hot at line {line_number}"
                    )
                label = _strict_integer(
                    fields[54],
                    line_number=line_number,
                    column_number=55,
                )
                if not 1 <= label <= 7:
                    raise ValueError(f"invalid Covertype label at line {line_number}")
                row_index = line_number - 1
                if preallocated is not None:
                    if row_index >= len(preallocated):
                        raise ValueError(
                            f"Covertype source has more than expected {expected_rows} rows"
                        )
                    preallocated[row_index] = values
                else:
                    dynamic_rows.append(tuple(float(value) for value in values))
                labels.append(str(label))
    except (UnicodeDecodeError, gzip.BadGzipFile, EOFError) as error:
        raise ValueError(f"invalid ASCII or gzip Covertype source {file_path}") from error
    if expected_rows is not None and len(labels) != expected_rows:
        raise ValueError(
            f"Covertype source has {len(labels)} rows; expected exactly {expected_rows}"
        )
    features = (
        preallocated
        if preallocated is not None
        else np.asarray(dynamic_rows, dtype=np.float64)
    )
    features.flags.writeable = False
    return ParsedUCISource(
        path=str(file_path),
        sha256=source_sha256,
        byte_count=byte_count,
        features=features,
        labels=tuple(labels),
    )


def _check_expected_digest(parsed: ParsedUCISource, expected: str | None, role: str) -> None:
    if expected is not None and parsed.sha256 != _validate_sha256(expected, f"{role}_sha256"):
        raise ValueError(
            f"{role} source SHA-256 mismatch: expected {expected}, observed {parsed.sha256}"
        )


def load_letter_recognition(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> LoadedUCIDataset:
    """Load the official Letter file and enforce its 16,000/4,000 split."""

    parsed = parse_letter_source(path)
    _check_expected_digest(parsed, expected_sha256, "letter")
    expected_rows = LETTER_SPEC.train_rows + LETTER_SPEC.test_rows
    if parsed.features.shape != (expected_rows, LETTER_SPEC.feature_count):
        raise ValueError(
            "official Letter source must contain exactly 20,000 rows and 16 features"
        )
    split_index = LETTER_SPEC.train_rows
    return LoadedUCIDataset(
        dataset_id=LETTER_SPEC.dataset_id,
        train_features=parsed.features[:split_index],
        train_labels=parsed.labels[:split_index],
        test_features=parsed.features[split_index:],
        test_labels=parsed.labels[split_index:],
        sources=(
            UCISourceRecord(
                role="official_combined_train_then_test",
                file_name=LETTER_SPEC.file_names[0],
                sha256=parsed.sha256,
                byte_count=parsed.byte_count,
                row_count=expected_rows,
                source_url=LETTER_SPEC.source_urls[0],
            ),
        ),
        split=OfficialSplitMetadata(
            strategy=LETTER_SPEC.split_strategy,
            train_rows=LETTER_SPEC.train_rows,
            test_rows=LETTER_SPEC.test_rows,
            train_source_roles=("official_combined_train_then_test:first_16000",),
            test_source_roles=("official_combined_train_then_test:last_4000",),
        ),
        landing_url=LETTER_SPEC.landing_url,
    )


def load_optdigits(
    train_path: str | Path,
    test_path: str | Path,
    *,
    expected_train_sha256: str | None = None,
    expected_test_sha256: str | None = None,
) -> LoadedUCIDataset:
    """Load and enforce the two official Optdigits source files and split."""

    train = parse_optdigits_source(train_path)
    test = parse_optdigits_source(test_path)
    _check_expected_digest(train, expected_train_sha256, "optdigits_train")
    _check_expected_digest(test, expected_test_sha256, "optdigits_test")
    if train.features.shape != (OPTDIGITS_SPEC.train_rows, OPTDIGITS_SPEC.feature_count):
        raise ValueError("official optdigits.tra must contain exactly 3,823 rows")
    if test.features.shape != (OPTDIGITS_SPEC.test_rows, OPTDIGITS_SPEC.feature_count):
        raise ValueError("official optdigits.tes must contain exactly 1,797 rows")
    sources = (
        UCISourceRecord(
            role="official_train",
            file_name=OPTDIGITS_SPEC.file_names[0],
            sha256=train.sha256,
            byte_count=train.byte_count,
            row_count=OPTDIGITS_SPEC.train_rows,
            source_url=OPTDIGITS_SPEC.source_urls[0],
        ),
        UCISourceRecord(
            role="official_test",
            file_name=OPTDIGITS_SPEC.file_names[1],
            sha256=test.sha256,
            byte_count=test.byte_count,
            row_count=OPTDIGITS_SPEC.test_rows,
            source_url=OPTDIGITS_SPEC.source_urls[1],
        ),
    )
    return LoadedUCIDataset(
        dataset_id=OPTDIGITS_SPEC.dataset_id,
        train_features=train.features,
        train_labels=train.labels,
        test_features=test.features,
        test_labels=test.labels,
        sources=sources,
        split=OfficialSplitMetadata(
            strategy=OPTDIGITS_SPEC.split_strategy,
            train_rows=OPTDIGITS_SPEC.train_rows,
            test_rows=OPTDIGITS_SPEC.test_rows,
            train_source_roles=("official_train",),
            test_source_roles=("official_test",),
        ),
        landing_url=OPTDIGITS_SPEC.landing_url,
    )


def validate_official_covertype_shape(row_count: int, feature_count: int) -> None:
    """Fail unless dimensions equal the official 581,012 by 54 source."""

    rows = _integer(row_count, "row_count", minimum=1)
    features = _integer(feature_count, "feature_count", minimum=1)
    if rows != 581_012 or features != COVERTYPE_SPEC.feature_count:
        raise ValueError("official Covertype source must have exactly 581,012 rows and 54 features")


def _row_index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"qgapselect.row-index-commitment.v1\0")
    digest.update(len(indices).to_bytes(8, "big", signed=False))
    for index in indices:
        digest.update(int(index).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StableStratifiedHashSplit:
    """Deterministic stratified split plus original-row commitments."""

    dataset_id: str
    test_fraction: float
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_row_index_sha256: str
    test_row_index_sha256: str
    per_label_counts: tuple[tuple[str, int, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "test_fraction": self.test_fraction,
            "train_rows": len(self.train_indices),
            "test_rows": len(self.test_indices),
            "train_row_index_sha256": self.train_row_index_sha256,
            "test_row_index_sha256": self.test_row_index_sha256,
            "per_label_counts": [
                {"label": label, "train": train, "test": test}
                for label, train, test in self.per_label_counts
            ],
        }


def stable_stratified_hash_split(
    dataset_id: str,
    labels: Sequence[object],
    *,
    test_fraction: float = 0.2,
) -> StableStratifiedHashSplit:
    """Split each label class by a stable hash of dataset, label, and row."""

    dataset_id = _nonempty_string(dataset_id, "dataset_id")
    normalized_labels = _labels(labels, "labels")
    if isinstance(test_fraction, bool):
        raise TypeError("test_fraction must be a real number, not bool")
    fraction = float(test_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 0.5:
        raise ValueError("test_fraction must lie in (0, 0.5)")
    by_label: dict[str, list[int]] = {}
    for row_index, label in enumerate(normalized_labels):
        by_label.setdefault(label, []).append(row_index)
    train_indices: list[int] = []
    test_indices: list[int] = []
    counts: list[tuple[str, int, int]] = []
    for label in sorted(by_label):
        if len(by_label[label]) < 2:
            raise ValueError(f"label {label!r} needs at least two rows for a fixed split")
        ordered = sorted(
            by_label[label],
            key=lambda row_index: (
                hashlib.sha256(
                    (
                        "qgapselect.stratified-hash-split.v1\0"
                        f"{dataset_id}\0{label}\0{row_index}"
                    ).encode()
                ).digest(),
                row_index,
            ),
        )
        test_count = max(1, min(len(ordered) - 1, round(fraction * len(ordered))))
        test_indices.extend(ordered[:test_count])
        train_indices.extend(ordered[test_count:])
        counts.append((label, len(ordered) - test_count, test_count))
    train_indices.sort()
    test_indices.sort()
    train_tuple = tuple(train_indices)
    test_tuple = tuple(test_indices)
    return StableStratifiedHashSplit(
        dataset_id=dataset_id,
        test_fraction=fraction,
        train_indices=train_tuple,
        test_indices=test_tuple,
        train_row_index_sha256=_row_index_sha256(train_tuple),
        test_row_index_sha256=_row_index_sha256(test_tuple),
        per_label_counts=tuple(counts),
    )


def load_covertype(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> LoadedUCIDataset:
    """Load official local Covertype rows and apply the fixed 80/20 split.

    The source is official; the split is a preregistered benchmark split, not
    an official UCI train/test partition.  No network fallback is attempted.
    """

    file_path = Path(path)
    if file_path.name not in set(COVERTYPE_SPEC.file_names):
        raise ValueError("Covertype path must be named covtype.data or covtype.data.gz")
    parsed = parse_covertype_source(file_path, expected_rows=581_012)
    _check_expected_digest(parsed, expected_sha256, "covertype")
    validate_official_covertype_shape(
        parsed.features.shape[0],
        parsed.features.shape[1],
    )
    split = stable_stratified_hash_split(
        COVERTYPE_SPEC.dataset_id,
        parsed.labels,
        test_fraction=0.2,
    )
    if (
        len(split.train_indices) != COVERTYPE_SPEC.train_rows
        or len(split.test_indices) != COVERTYPE_SPEC.test_rows
    ):
        raise ValueError(
            "Covertype label counts do not match the preregistered official-source split sizes"
        )
    train_indices = np.fromiter(split.train_indices, dtype=np.intp)
    test_indices = np.fromiter(split.test_indices, dtype=np.intp)
    train_features = parsed.features[train_indices]
    test_features = parsed.features[test_indices]
    train_features.flags.writeable = False
    test_features.flags.writeable = False
    source = UCISourceRecord(
        role="official_combined_source_before_preregistered_split",
        file_name=file_path.name,
        sha256=parsed.sha256,
        byte_count=parsed.byte_count,
        row_count=len(parsed.labels),
        source_url=COVERTYPE_SPEC.source_urls[0],
    )
    return LoadedUCIDataset(
        dataset_id=COVERTYPE_SPEC.dataset_id,
        train_features=train_features,
        train_labels=tuple(parsed.labels[index] for index in split.train_indices),
        test_features=test_features,
        test_labels=tuple(parsed.labels[index] for index in split.test_indices),
        sources=(source,),
        split=OfficialSplitMetadata(
            strategy=COVERTYPE_SPEC.split_strategy,
            train_rows=len(split.train_indices),
            test_rows=len(split.test_indices),
            train_source_roles=(
                "official_combined_source_before_preregistered_split:hash_remainder",
            ),
            test_source_roles=(
                "official_combined_source_before_preregistered_split:hash_prefix",
            ),
            train_row_index_sha256=split.train_row_index_sha256,
            test_row_index_sha256=split.test_row_index_sha256,
        ),
        landing_url=COVERTYPE_SPEC.landing_url,
        official_source=True,
    )


def make_in_memory_dataset(
    *,
    dataset_id: str,
    train_features: object,
    train_labels: Sequence[object],
    test_features: object,
    test_labels: Sequence[object],
) -> LoadedUCIDataset:
    """Create a clearly non-official dataset for unit tests and local audits."""

    x_train = _immutable_matrix(train_features, "train_features")
    x_test = _immutable_matrix(test_features, "test_features")
    y_train = _labels(train_labels, "train_labels")
    y_test = _labels(test_labels, "test_labels")
    content_hash = _canonical_hash(
        {
            "dataset_id": dataset_id,
            "train_features": _sha256_bytes(np.ascontiguousarray(x_train).tobytes()),
            "train_labels": _canonical_hash(list(y_train)),
            "test_features": _sha256_bytes(np.ascontiguousarray(x_test).tobytes()),
            "test_labels": _canonical_hash(list(y_test)),
        }
    )
    source = UCISourceRecord(
        role="in_memory_test_only",
        file_name="in-memory",
        sha256=content_hash,
        byte_count=1,
        row_count=len(y_train) + len(y_test),
        source_url="urn:qgapselect:in-memory-test-only",
    )
    return LoadedUCIDataset(
        dataset_id=dataset_id,
        train_features=x_train,
        train_labels=y_train,
        test_features=x_test,
        test_labels=y_test,
        sources=(source,),
        split=OfficialSplitMetadata(
            strategy="test_only_caller_supplied_split_not_an_official_uci_split",
            train_rows=len(y_train),
            test_rows=len(y_test),
            train_source_roles=("in_memory_test_only",),
            test_source_roles=("in_memory_test_only",),
        ),
        landing_url="urn:qgapselect:in-memory-test-only",
        official_source=False,
    )


def load_sklearn_digits_offline(
    *,
    test_fraction: float = 0.25,
) -> LoadedUCIDataset:
    """Load sklearn's bundled Digits copy with a deterministic local split.

    This is an offline execution fallback, not the full UCI Optdigits dataset
    and not its official ``optdigits.tra``/``optdigits.tes`` split.  Within
    each class, original row indices are ordered by a stable hash; the first
    preregistered fraction is assigned to test and the remainder to train.
    """

    try:
        import sklearn
        from sklearn.datasets import load_digits
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "offline Digits fixtures require the 'datasets' optional dependency"
        ) from error

    bunch = load_digits()
    features = _immutable_matrix(bunch.data, "digits_features")
    labels = tuple(str(int(value)) for value in np.asarray(bunch.target).tolist())
    split = stable_stratified_hash_split(
        SKLEARN_DIGITS_OFFLINE_ID,
        labels,
        test_fraction=test_fraction,
    )
    train_indices = np.fromiter(split.train_indices, dtype=np.intp)
    test_indices = np.fromiter(split.test_indices, dtype=np.intp)

    data_hash = _sha256_bytes(np.asarray(features, dtype="<f8", order="C").tobytes())
    target_hash = _canonical_hash(list(labels))
    description_hash = _sha256_bytes(str(bunch.DESCR).encode("utf-8"))
    source_payload = _canonical_json(
        {
            "loader": "sklearn.datasets.load_digits",
            "sklearn_version": sklearn.__version__,
            "data_sha256": data_hash,
            "target_sha256": target_hash,
            "description_sha256": description_hash,
            "original_row_count": len(labels),
        }
    )
    source = UCISourceRecord(
        role="sklearn_bundled_digits_offline_copy",
        file_name="sklearn.datasets.load_digits",
        sha256=_sha256_bytes(source_payload),
        byte_count=len(source_payload),
        row_count=len(labels),
        source_url=(
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.datasets.load_digits.html"
        ),
    )
    return LoadedUCIDataset(
        dataset_id=SKLEARN_DIGITS_OFFLINE_ID,
        train_features=features[train_indices],
        train_labels=tuple(labels[index] for index in split.train_indices),
        test_features=features[test_indices],
        test_labels=tuple(labels[index] for index in split.test_indices),
        sources=(source,),
        split=OfficialSplitMetadata(
            strategy=(
                "non_official_deterministic_stratified_hash_split_v1_"
                f"test_fraction_{split.test_fraction:.12g}"
            ),
            train_rows=len(split.train_indices),
            test_rows=len(split.test_indices),
            train_source_roles=("sklearn_bundled_digits_offline_copy:hash_remainder",),
            test_source_roles=("sklearn_bundled_digits_offline_copy:hash_prefix",),
            train_row_index_sha256=split.train_row_index_sha256,
            test_row_index_sha256=split.test_row_index_sha256,
        ),
        landing_url=(
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.datasets.load_digits.html"
        ),
        official_source=False,
    )


_JSON_SCALAR = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """One outcome-independent entry in the preregistered arm catalogue."""

    config_id: str
    model_family: str
    parameters: tuple[tuple[str, _JSON_SCALAR], ...]
    feature_fraction: float = 1.0
    standardize: bool = True
    random_state: int = 0
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        config_id = _nonempty_string(self.config_id, "config_id")
        if self.model_family not in {"ridge", "nearest_centroid", "gaussian_nb", "knn"}:
            raise ValueError(f"unsupported model_family {self.model_family!r}")
        parameters = tuple((str(key), value) for key, value in self.parameters)
        if any(not key for key, _ in parameters) or len({key for key, _ in parameters}) != len(
            parameters
        ):
            raise ValueError("classifier parameter names must be non-empty and unique")
        if any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for _, value in parameters
        ):
            raise TypeError("classifier parameters must be JSON scalar values")
        fraction = float(self.feature_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("feature_fraction must lie in (0, 1]")
        random_state = _integer(self.random_state, "random_state", minimum=0)
        if not isinstance(self.standardize, bool):
            raise TypeError("standardize must be bool")
        object.__setattr__(self, "config_id", config_id)
        object.__setattr__(self, "parameters", tuple(sorted(parameters)))
        object.__setattr__(self, "feature_fraction", fraction)
        object.__setattr__(self, "random_state", random_state)
        object.__setattr__(self, "config_hash", _canonical_hash(self.document()))

    def document(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "model_family": self.model_family,
            "parameters": dict(self.parameters),
            "feature_fraction": self.feature_fraction,
            "standardize": self.standardize,
            "random_state": self.random_state,
            "feature_selection": "train_only_stable_anova_ranking",
        }


def _build_default_catalog() -> tuple[ClassifierConfig, ...]:
    alphas = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    shrinkages: tuple[float | None, ...] = (None, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    smoothings = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5)
    neighbors = (1, 3, 5, 7, 9, 11, 13, 15)
    fractions = (1.0, 0.75, 0.5, 0.25)
    records: list[ClassifierConfig] = []
    ordinal = 0
    for index in range(8):
        fraction = fractions[index % len(fractions)]
        entries: tuple[tuple[str, tuple[tuple[str, _JSON_SCALAR], ...], bool], ...] = (
            ("ridge", (("alpha", alphas[index]),), True),
            ("nearest_centroid", (("shrink_threshold", shrinkages[index]),), True),
            ("gaussian_nb", (("var_smoothing", smoothings[index]),), False),
            (
                "knn",
                (
                    ("n_neighbors", neighbors[index]),
                    ("p", 1 if index % 2 == 0 else 2),
                    ("weights", "uniform" if index < 4 else "distance"),
                ),
                True,
            ),
        )
        for family, parameters, standardize in entries:
            records.append(
                ClassifierConfig(
                    config_id=f"cfg{ordinal:03d}_{family}",
                    model_family=family,
                    parameters=parameters,
                    feature_fraction=fraction,
                    standardize=standardize,
                    random_state=0,
                )
            )
            ordinal += 1
    return tuple(records)


DEFAULT_CLASSIFIER_CATALOG = _build_default_catalog()
DEFAULT_CATALOG_MANIFEST_HASH = _canonical_hash(
    [config.document() for config in DEFAULT_CLASSIFIER_CATALOG]
)


def training_feature_ranking(
    train_features: object,
    train_labels: Sequence[object],
) -> tuple[int, ...]:
    """Return a deterministic ANOVA-style ranking using training data only."""

    matrix = _immutable_matrix(train_features, "train_features")
    labels = np.asarray(_labels(train_labels, "train_labels"), dtype=object)
    if len(labels) != matrix.shape[0]:
        raise ValueError("training features and labels have different row counts")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("classifier training requires at least two classes")
    overall = np.mean(matrix, axis=0)
    between = np.zeros(matrix.shape[1], dtype=np.float64)
    within = np.zeros(matrix.shape[1], dtype=np.float64)
    for label in classes:
        class_rows = matrix[labels == label]
        class_mean = np.mean(class_rows, axis=0)
        between += len(class_rows) * np.square(class_mean - overall)
        within += np.sum(np.square(class_rows - class_mean), axis=0)
    scores = np.divide(
        between,
        within,
        out=np.zeros_like(between),
        where=within > 0.0,
    )
    scores[(within == 0.0) & (between > 0.0)] = np.inf
    return tuple(sorted(range(matrix.shape[1]), key=lambda index: (-scores[index], index)))


def _prediction_hash(predictions: Sequence[str]) -> str:
    return _canonical_hash({"labels": list(predictions)})


@dataclass(frozen=True, slots=True)
class FittedClassifierArm:
    """Frozen test predictions; no test correctness or accuracy is stored here."""

    config: ClassifierConfig
    selected_feature_indices: tuple[int, ...]
    training_ranking_sha256: str
    test_predictions: tuple[str, ...]
    prediction_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, ClassifierConfig):
            raise TypeError("config must be a ClassifierConfig")
        indices = tuple(
            _integer(value, "selected_feature_index", minimum=0)
            for value in self.selected_feature_indices
        )
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("selected feature indices must be non-empty and unique")
        predictions = _labels(self.test_predictions, "test_predictions")
        ranking_hash = _validate_sha256(self.training_ranking_sha256, "training_ranking_sha256")
        prediction_hash = _validate_sha256(self.prediction_sha256, "prediction_sha256")
        if prediction_hash != _prediction_hash(predictions):
            raise ValueError("prediction_sha256 does not commit to test_predictions")
        object.__setattr__(self, "selected_feature_indices", indices)
        object.__setattr__(self, "test_predictions", predictions)
        object.__setattr__(self, "training_ranking_sha256", ranking_hash)
        object.__setattr__(self, "prediction_sha256", prediction_hash)

    def manifest_record(self) -> dict[str, object]:
        return {
            "config_id": self.config.config_id,
            "config_hash": self.config.config_hash,
            "model_family": self.config.model_family,
            "selected_feature_indices": list(self.selected_feature_indices),
            "training_ranking_sha256": self.training_ranking_sha256,
            "prediction_sha256": self.prediction_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExcludedPredictionDuplicate:
    config_id: str
    retained_config_id: str
    prediction_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "config_id": self.config_id,
            "retained_config_id": self.retained_config_id,
            "prediction_sha256": self.prediction_sha256,
        }


@dataclass(frozen=True, slots=True)
class ClassifierArmSelection:
    """Outcome-blind result of config sorting and prediction-hash deduplication."""

    selected_arms: tuple[FittedClassifierArm, ...]
    unselected_unique_arms: tuple[FittedClassifierArm, ...]
    excluded_duplicates: tuple[ExcludedPredictionDuplicate, ...]
    catalog: tuple[ClassifierConfig, ...]
    catalog_manifest_hash: str

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(arm.config.config_id for arm in self.selected_arms)

    def manifest_document(self) -> dict[str, object]:
        return {
            "selection_rule": (
                "sort_config_id_then_deduplicate_by_full_test_prediction_sha256_"
                "keeping_first_then_take_first_n_without_heldout_outcomes"
            ),
            "catalog_manifest_hash": self.catalog_manifest_hash,
            "catalog": [
                config.document() | {"config_hash": config.config_hash}
                for config in self.catalog
            ],
            "selected_arms": [arm.manifest_record() for arm in self.selected_arms],
            "unselected_unique_arms": [
                arm.manifest_record() for arm in self.unselected_unique_arms
            ],
            "excluded_prediction_duplicates": [
                duplicate.as_dict() for duplicate in self.excluded_duplicates
            ],
        }


def _classifier_pipeline(config: ClassifierConfig) -> Any:
    try:
        from sklearn.base import BaseEstimator
        from sklearn.linear_model import RidgeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "UCI classifier fixtures require the 'datasets' optional dependency"
        ) from error

    parameters = dict(config.parameters)
    estimator: BaseEstimator
    if config.model_family == "ridge":
        estimator = RidgeClassifier(alpha=float(parameters["alpha"]))
    elif config.model_family == "nearest_centroid":
        estimator = NearestCentroid(shrink_threshold=parameters["shrink_threshold"])
    elif config.model_family == "gaussian_nb":
        estimator = GaussianNB(var_smoothing=float(parameters["var_smoothing"]))
    elif config.model_family == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=int(parameters["n_neighbors"]),
            p=int(parameters["p"]),
            weights=str(parameters["weights"]),
            n_jobs=1,
        )
    else:  # pragma: no cover - ClassifierConfig validates this
        raise AssertionError(config.model_family)
    steps: list[tuple[str, Any]] = []
    if config.standardize:
        steps.append(("standard_scaler_fit_on_train_only", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def fit_preregistered_classifier_arms(
    dataset: LoadedUCIDataset,
    *,
    n_arms: int,
    catalog: Sequence[ClassifierConfig] = DEFAULT_CLASSIFIER_CATALOG,
) -> ClassifierArmSelection:
    """Fit sorted configs, deduplicate predictions, and take the first ``n``.

    This function never reads ``dataset.test_labels``.  Test labels first enter
    the pipeline later, when the trusted harness freezes correctness streams.
    """

    if not isinstance(dataset, LoadedUCIDataset):
        raise TypeError("dataset must be a LoadedUCIDataset")
    n_arms = _integer(n_arms, "n_arms", minimum=2)
    records = tuple(catalog)
    if not records or any(not isinstance(item, ClassifierConfig) for item in records):
        raise TypeError("catalog must contain ClassifierConfig objects")
    if len({item.config_id for item in records}) != len(records):
        raise ValueError("catalog config_id values must be unique")
    records = tuple(sorted(records, key=lambda item: item.config_id))
    catalog_hash = _canonical_hash([config.document() for config in records])
    ranking = training_feature_ranking(dataset.train_features, dataset.train_labels)
    ranking_hash = _canonical_hash({"feature_ranking": list(ranking)})
    unique: list[FittedClassifierArm] = []
    retained_by_prediction: dict[str, str] = {}
    duplicates: list[ExcludedPredictionDuplicate] = []
    for config in records:
        selected_count = max(1, math.ceil(config.feature_fraction * len(ranking)))
        selected_indices = tuple(sorted(ranking[:selected_count]))
        train_features = dataset.train_features[:, selected_indices]
        test_features = dataset.test_features[:, selected_indices]
        estimator = _classifier_pipeline(config)
        estimator.fit(train_features, np.asarray(dataset.train_labels))
        predictions = tuple(str(value) for value in estimator.predict(test_features).tolist())
        prediction_hash = _prediction_hash(predictions)
        retained = retained_by_prediction.get(prediction_hash)
        if retained is not None:
            duplicates.append(
                ExcludedPredictionDuplicate(
                    config_id=config.config_id,
                    retained_config_id=retained,
                    prediction_sha256=prediction_hash,
                )
            )
            continue
        retained_by_prediction[prediction_hash] = config.config_id
        unique.append(
            FittedClassifierArm(
                config=config,
                selected_feature_indices=selected_indices,
                training_ranking_sha256=ranking_hash,
                test_predictions=predictions,
                prediction_sha256=prediction_hash,
            )
        )
    if len(unique) < n_arms:
        raise ValueError(
            f"catalog produced only {len(unique)} unique test-prediction vectors; "
            f"cannot select {n_arms} arms"
        )
    return ClassifierArmSelection(
        selected_arms=tuple(unique[:n_arms]),
        unselected_unique_arms=tuple(unique[n_arms:]),
        excluded_duplicates=tuple(duplicates),
        catalog=records,
        catalog_manifest_hash=catalog_hash,
    )


@dataclass(frozen=True, slots=True)
class StableTestShard:
    shard_id: str
    ordinal: int
    row_indices: tuple[int, ...]
    row_index_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "ordinal": self.ordinal,
            "row_count": len(self.row_indices),
            "row_index_sha256": self.row_index_sha256,
        }


def stable_label_blind_test_shards(
    dataset_id: str,
    n_rows: int,
    *,
    n_shards: int = DEFAULT_SHARD_COUNT,
) -> tuple[StableTestShard, ...]:
    """Partition rows using only dataset identity and row index.

    Rows are ordered by a stable SHA-256 score and dealt round-robin, ensuring
    balanced, non-empty shards when ``n_rows >= n_shards``.  Labels and model
    outcomes are not arguments and therefore cannot influence the partition.
    """

    dataset_id = _nonempty_string(dataset_id, "dataset_id")
    n_rows = _integer(n_rows, "n_rows", minimum=1)
    n_shards = _integer(n_shards, "n_shards", minimum=5)
    if n_rows < n_shards:
        raise ValueError("n_rows must be at least n_shards")
    ordered_rows = sorted(
        range(n_rows),
        key=lambda index: (
            hashlib.sha256(
                f"qgapselect.uci-test-shard.v1\0{dataset_id}\0{index}".encode()
            ).digest(),
            index,
        ),
    )
    buckets: list[list[int]] = [[] for _ in range(n_shards)]
    for rank, row_index in enumerate(ordered_rows):
        buckets[rank % n_shards].append(row_index)
    return tuple(
        StableTestShard(
            shard_id=f"{dataset_id}__shard-{ordinal:03d}-of-{n_shards:03d}",
            ordinal=ordinal,
            row_indices=tuple(sorted(bucket)),
            row_index_sha256=_canonical_hash({"row_indices": sorted(bucket)}),
        )
        for ordinal, bucket in enumerate(buckets)
    )


@dataclass(frozen=True, slots=True)
class BoundaryFailure:
    """Trusted-harness report for a shard rejected before algorithm execution."""

    instance_id: str
    shard_id: str
    reason: str
    k: int
    kth_success_count: int
    next_success_count: int
    shard_size: int

    def manifest_record(self) -> dict[str, object]:
        # Counts are deliberately omitted from the public manifest.
        return {
            "instance_id": self.instance_id,
            "shard_id": self.shard_id,
            "status": "rejected_fail_closed",
            "reason": self.reason,
        }


class NonUniqueTopKError(ValueError):
    """Raised when exact Top-k has a tie or a zero angular boundary gap."""

    def __init__(self, report: BoundaryFailure) -> None:
        self.report = report
        super().__init__(
            f"{report.instance_id} rejected fail-closed: {report.reason} "
            f"({report.kth_success_count} versus {report.next_success_count})"
        )


def build_exact_topk_frozen_instance(
    *,
    family_id: str,
    instance_id: str,
    shard: StableTestShard,
    fixture: FrozenSourceFixture,
    k: int,
    structure_metrics: Mapping[str, object] | None = None,
) -> FrozenQuantumReferenceInstance:
    """Build one trusted exact Top-k instance or reject a tied boundary."""

    means = fixture.evaluator.frozen_means
    if not 1 <= k < len(means):
        raise ValueError("k must be positive and strictly smaller than the arm count")
    order = sorted(range(len(means)), key=lambda index: (-means[index], index))
    kth_index, next_index = order[k - 1], order[k]
    kth_mean, next_mean = means[kth_index], means[next_index]
    stream_length = len(fixture.tensor.reward_streams[0])
    kth_count = sum(fixture.tensor.reward_streams[kth_index])
    next_count = sum(fixture.tensor.reward_streams[next_index])
    if kth_count == next_count or kth_mean <= next_mean:
        raise NonUniqueTopKError(
            BoundaryFailure(
                instance_id=instance_id,
                shard_id=shard.shard_id,
                reason="boundary_tie_non_unique_exact_top_k",
                k=k,
                kth_success_count=kth_count,
                next_success_count=next_count,
                shard_size=stream_length,
            )
        )
    threshold = (kth_mean + next_mean) / 2.0
    threshold_angle = math.asin(math.sqrt(threshold))
    angular_gap = min(
        abs(math.asin(math.sqrt(mean)) - threshold_angle) for mean in means
    )
    if not math.isfinite(angular_gap) or angular_gap <= 0.0:
        raise NonUniqueTopKError(
            BoundaryFailure(
                instance_id=instance_id,
                shard_id=shard.shard_id,
                reason="zero_angular_boundary_gap",
                k=k,
                kth_success_count=kth_count,
                next_success_count=next_count,
                shard_size=stream_length,
            )
        )
    return FrozenQuantumReferenceInstance(
        family_id=family_id,
        instance_id=instance_id,
        fixture=fixture,
        public_threshold=threshold,
        public_gap_floor=angular_gap,
        k=k,
        structure_metrics={} if structure_metrics is None else structure_metrics,
    )


@dataclass(frozen=True, slots=True)
class UCIAlgorithmFixture:
    """Algorithm-side view: commitments, candidate IDs, k, and a blind oracle."""

    instance_id: str
    fixture_manifest_hash: str
    candidate_ids: tuple[str, ...]
    k: int
    oracle: BlindSourceRewardOracle

    def public_document(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "fixture_manifest_hash": self.fixture_manifest_hash,
            "candidate_ids": list(self.candidate_ids),
            "k": self.k,
            "information_regime": "k_only_plus_blind_reward_oracle",
        }


@dataclass(frozen=True, slots=True)
class UCIClassifierBenchmarkManifest:
    document: Mapping[str, object]
    manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        # Canonical JSON round trip also rejects non-JSON and non-finite values.
        normalized = json.loads(_canonical_json(dict(self.document)).decode("utf-8"))
        object.__setattr__(self, "document", MappingProxyType(normalized))
        object.__setattr__(self, "manifest_hash", _canonical_hash(normalized))

    def as_dict(self) -> dict[str, object]:
        return json.loads(_canonical_json(dict(self.document)).decode("utf-8"))


@dataclass(frozen=True, slots=True)
class UCIClassifierBenchmark:
    """Trusted-harness UCI benchmark; do not pass this object to an algorithm."""

    dataset: LoadedUCIDataset
    arm_selection: ClassifierArmSelection
    shards: tuple[StableTestShard, ...]
    instances: tuple[FrozenQuantumReferenceInstance, ...]
    boundary_failures: tuple[BoundaryFailure, ...]
    manifest: UCIClassifierBenchmarkManifest

    def open_algorithm_fixture(
        self,
        instance_id: str,
        budget: SourceOracleBudget,
    ) -> UCIAlgorithmFixture:
        matches = tuple(item for item in self.instances if item.instance_id == instance_id)
        if len(matches) != 1:
            raise KeyError(instance_id)
        instance = matches[0]
        return UCIAlgorithmFixture(
            instance_id=instance.instance_id,
            fixture_manifest_hash=instance.fixture.manifest_hash,
            candidate_ids=instance.fixture.tensor.graph.candidate_ids,
            k=instance.k,
            oracle=instance.fixture.open_oracle(budget),
        )


def build_uci_classifier_benchmark(
    dataset: LoadedUCIDataset,
    *,
    n_arms: int,
    k: int,
    n_shards: int = DEFAULT_SHARD_COUNT,
    catalog: Sequence[ClassifierConfig] = DEFAULT_CLASSIFIER_CATALOG,
) -> UCIClassifierBenchmark:
    """Build exact correctness fixtures on stable label-blind test shards.

    Boundary-tied shards are excluded and reported; no jitter, pseudocount, or
    outcome-dependent arm replacement is allowed.  If every shard is tied,
    construction fails instead of returning an unusable benchmark.
    """

    if not isinstance(dataset, LoadedUCIDataset):
        raise TypeError("dataset must be a LoadedUCIDataset")
    n_arms = _integer(n_arms, "n_arms", minimum=2)
    k = _integer(k, "k", minimum=1)
    if k >= n_arms:
        raise ValueError("k must be strictly smaller than n_arms")
    selection = fit_preregistered_classifier_arms(
        dataset,
        n_arms=n_arms,
        catalog=catalog,
    )
    shards = stable_label_blind_test_shards(
        dataset.dataset_id,
        len(dataset.test_labels),
        n_shards=n_shards,
    )
    graph = FrozenCandidateGraph(
        candidates=tuple(
            SourceCandidate(
                candidate_id=arm.config.config_id,
                payload_hash=arm.config.config_hash,
                family=arm.config.model_family,
            )
            for arm in selection.selected_arms
        )
    )
    family_id = f"uci_{dataset.dataset_id}_classifier_selection_shards_v1"
    instances: list[FrozenQuantumReferenceInstance] = []
    failures: list[BoundaryFailure] = []
    fixture_records: list[dict[str, object]] = []
    prediction_set_hash = _canonical_hash(
        [arm.prediction_sha256 for arm in selection.selected_arms]
    )
    for shard in shards:
        rewards = {
            arm.config.config_id: tuple(
                int(arm.test_predictions[row] == dataset.test_labels[row])
                for row in shard.row_indices
            )
            for arm in selection.selected_arms
        }
        costs = {
            candidate_id: (1.0,) * len(shard.row_indices)
            for candidate_id in selection.candidate_ids
        }
        fixture = freeze_source_streams(
            graph,
            rewards,
            costs,
            metadata={
                "claim_scope": CLAIM_SCOPE,
                "dataset_id": dataset.dataset_id,
                "source_manifest_hash": dataset.source_manifest_hash,
                "catalog_manifest_hash": selection.catalog_manifest_hash,
                "prediction_set_hash": prediction_set_hash,
                "shard_id": shard.shard_id,
                "row_index_sha256": shard.row_index_sha256,
                "reward_definition": "classifier_test_prediction_equals_frozen_test_label",
                "noise_policy": "none_exact_correctness_bits",
            },
        )
        instance_id = f"{dataset.dataset_id}__classifier-top{k}__{shard.shard_id}"
        try:
            instance = build_exact_topk_frozen_instance(
                family_id=family_id,
                instance_id=instance_id,
                shard=shard,
                fixture=fixture,
                k=k,
                structure_metrics={
                    "external_dataset_id": dataset.dataset_id,
                    "shard_ordinal": shard.ordinal,
                    "shard_size": len(shard.row_indices),
                    "n_shards": len(shards),
                    "semi_synthetic_external_validity": True,
                },
            )
        except NonUniqueTopKError as error:
            failures.append(error.report)
            fixture_records.append(error.report.manifest_record())
            continue
        instances.append(instance)
        fixture_records.append(
            {
                "instance_id": instance.instance_id,
                "shard_id": shard.shard_id,
                "status": "accepted_unique_exact_top_k",
                "fixture_manifest_hash": fixture.manifest_hash,
            }
        )
    if not instances:
        first = failures[0]
        raise NonUniqueTopKError(first)
    manifest_document: dict[str, object] = {
        "schema": "qgapselect.uci-classifier-benchmark-manifest.v1",
        "claim_scope": CLAIM_SCOPE,
        "claim_boundary": (
            "semi-synthetic external-validity evidence only; not quantum hardware, "
            "wall-clock, asymptotic-speedup, or quantum-advantage evidence"
        ),
        "dataset": dataset.source_document()
        | {"source_manifest_hash": dataset.source_manifest_hash},
        "arm_selection": selection.manifest_document(),
        "n_arms": n_arms,
        "k": k,
        "test_shards": [shard.as_dict() for shard in shards],
        "fixtures": fixture_records,
        "threshold_and_truth_visibility": "trusted_harness_only_never_algorithm_input",
        "noise_policy": "none",
    }
    return UCIClassifierBenchmark(
        dataset=dataset,
        arm_selection=selection,
        shards=shards,
        instances=tuple(instances),
        boundary_failures=tuple(failures),
        manifest=UCIClassifierBenchmarkManifest(manifest_document),
    )


__all__ = [
    "CLAIM_SCOPE",
    "COVERTYPE_SPEC",
    "DATASET_SPECS",
    "DEFAULT_CATALOG_MANIFEST_HASH",
    "DEFAULT_CLASSIFIER_CATALOG",
    "DEFAULT_SHARD_COUNT",
    "LETTER_SPEC",
    "LICENSE_NAME",
    "LICENSE_URL",
    "OPTDIGITS_SPEC",
    "SKLEARN_DIGITS_OFFLINE_ID",
    "BoundaryFailure",
    "ClassifierArmSelection",
    "ClassifierConfig",
    "ExcludedPredictionDuplicate",
    "FittedClassifierArm",
    "LoadedUCIDataset",
    "NonUniqueTopKError",
    "OfficialSplitMetadata",
    "ParsedUCISource",
    "StableTestShard",
    "StableStratifiedHashSplit",
    "UCIAlgorithmFixture",
    "UCIClassifierBenchmark",
    "UCIClassifierBenchmarkManifest",
    "UCIDatasetSpec",
    "UCISourceRecord",
    "build_uci_classifier_benchmark",
    "build_exact_topk_frozen_instance",
    "fit_preregistered_classifier_arms",
    "load_letter_recognition",
    "load_covertype",
    "load_optdigits",
    "load_sklearn_digits_offline",
    "make_in_memory_dataset",
    "parse_letter_source",
    "parse_covertype_source",
    "parse_optdigits_source",
    "stable_label_blind_test_shards",
    "stable_stratified_hash_split",
    "training_feature_ranking",
    "validate_official_covertype_shape",
]
