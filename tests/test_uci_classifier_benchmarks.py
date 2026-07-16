from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qgapselect.attack_oracles import (
    FrozenCandidateGraph,
    SourceOracleBudget,
    freeze_source_streams,
)
from qgapselect.uci_classifier_benchmarks import (
    CLAIM_SCOPE,
    COVERTYPE_SPEC,
    DEFAULT_CLASSIFIER_CATALOG,
    LETTER_SPEC,
    LICENSE_URL,
    OPTDIGITS_SPEC,
    ClassifierConfig,
    NonUniqueTopKError,
    StableTestShard,
    UCIClassifierBenchmarkManifest,
    build_exact_topk_frozen_instance,
    build_uci_classifier_benchmark,
    fit_preregistered_classifier_arms,
    load_letter_recognition,
    load_sklearn_digits_offline,
    make_in_memory_dataset,
    parse_covertype_source,
    parse_letter_source,
    parse_optdigits_source,
    stable_label_blind_test_shards,
    stable_stratified_hash_split,
    training_feature_ranking,
    validate_official_covertype_shape,
)


def _xor_dataset(*, test_labels: tuple[str, ...] | None = None):
    train_features = (
        (-1.0, -1.0),
        (-0.9, -1.0),
        (1.0, 1.0),
        (0.9, 1.0),
        (-1.0, 1.0),
        (-0.9, 1.0),
        (1.0, -1.0),
        (0.9, -1.0),
    )
    train_labels = ("A", "A", "A", "A", "B", "B", "B", "B")
    test_features = (
        (-1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (0.8, 0.8),
        (-0.8, 0.8),
        (0.8, -0.8),
        (-0.8, -0.8),
        (0.0, 0.0),
        (0.1, -0.1),
    )
    return make_in_memory_dataset(
        dataset_id="unit_xor",
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels or ("A",) * len(test_features),
    )


def _dedup_catalog() -> tuple[ClassifierConfig, ...]:
    return (
        ClassifierConfig(
            config_id="z_ridge_duplicate",
            model_family="ridge",
            parameters=(("alpha", 1.0),),
        ),
        ClassifierConfig(
            config_id="a_ridge_retained",
            model_family="ridge",
            parameters=(("alpha", 1.0),),
        ),
        ClassifierConfig(
            config_id="m_knn_unique",
            model_family="knn",
            parameters=(("weights", "uniform"), ("p", 2), ("n_neighbors", 1)),
        ),
    )


def _letter_row(label: str = "A", value: int = 0) -> str:
    return ",".join((label, *(str(value) for _ in range(16))))


def _optdigits_row(label: int = 0, value: int = 0) -> str:
    return ",".join((*(str(value) for _ in range(64)), str(label)))


def _covertype_row(label: int = 1) -> str:
    continuous = (100, 20, 3, 4, -5, 6, 7, 8, 9, 10)
    wilderness = (1, 0, 0, 0)
    soil = (1, *(0 for _ in range(39)))
    return ",".join(str(value) for value in (*continuous, *wilderness, *soil, label))


def test_default_catalog_is_large_preregistered_sorted_and_deterministic() -> None:
    assert len(DEFAULT_CLASSIFIER_CATALOG) == 32
    assert len(DEFAULT_CLASSIFIER_CATALOG) >= 24
    assert tuple(config.config_id for config in DEFAULT_CLASSIFIER_CATALOG) == tuple(
        sorted(config.config_id for config in DEFAULT_CLASSIFIER_CATALOG)
    )
    assert len({config.config_id for config in DEFAULT_CLASSIFIER_CATALOG}) == 32
    assert len({config.config_hash for config in DEFAULT_CLASSIFIER_CATALOG}) == 32
    assert all(config.random_state == 0 for config in DEFAULT_CLASSIFIER_CATALOG)
    assert {config.model_family for config in DEFAULT_CLASSIFIER_CATALOG} == {
        "ridge",
        "nearest_centroid",
        "gaussian_nb",
        "knn",
    }


def test_letter_parser_is_strict_and_commits_exact_source_bytes(tmp_path: Path) -> None:
    path = tmp_path / "letter-recognition.data"
    payload = (_letter_row("A", 0) + "\n" + _letter_row("Z", 15) + "\n").encode()
    path.write_bytes(payload)

    parsed = parse_letter_source(path)

    assert parsed.features.shape == (2, 16)
    assert parsed.labels == ("A", "Z")
    assert parsed.sha256 == hashlib.sha256(payload).hexdigest()
    assert not parsed.features.flags.writeable

    path.write_text(_letter_row("A", 16) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"outside \[0, 15\]"):
        parse_letter_source(path)
    path.write_text(_letter_row("A", 0).replace(",0", ", 0", 1) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="whitespace"):
        parse_letter_source(path)


def test_letter_loader_enforces_official_contiguous_split_and_license(tmp_path: Path) -> None:
    path = tmp_path / LETTER_SPEC.file_names[0]
    rows = [_letter_row(chr(ord("A") + index % 26), index % 16) for index in range(20_000)]
    path.write_text("\n".join(rows) + "\n", encoding="ascii")

    dataset = load_letter_recognition(path)

    assert dataset.train_features.shape == (16_000, 16)
    assert dataset.test_features.shape == (4_000, 16)
    assert dataset.train_labels[-1] == rows[15_999][0]
    assert dataset.test_labels[0] == rows[16_000][0]
    assert dataset.split.strategy == LETTER_SPEC.split_strategy
    assert dataset.license_url == LICENSE_URL
    assert dataset.sources[0].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset.official_source


def test_optdigits_parser_rejects_bad_shape_range_and_label(tmp_path: Path) -> None:
    path = tmp_path / "optdigits.tra"
    path.write_text(_optdigits_row(9, 16) + "\n", encoding="ascii")
    parsed = parse_optdigits_source(path)
    assert parsed.features.shape == (1, 64)
    assert parsed.labels == ("9",)

    path.write_text(_optdigits_row(0, 17) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"outside \[0, 16\]"):
        parse_optdigits_source(path)
    fields = _optdigits_row(0, 0).split(",")
    path.write_text(",".join(fields[:-1] + ["10"]) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="invalid Optdigits label"):
        parse_optdigits_source(path)


def test_supported_official_specs_commit_predefined_rows_and_urls() -> None:
    assert LETTER_SPEC.train_rows == 16_000 and LETTER_SPEC.test_rows == 4_000
    assert OPTDIGITS_SPEC.train_rows == 3_823 and OPTDIGITS_SPEC.test_rows == 1_797
    assert COVERTYPE_SPEC.train_rows == 464_810
    assert COVERTYPE_SPEC.test_rows == 116_202
    assert COVERTYPE_SPEC.feature_count == 54
    assert LETTER_SPEC.landing_url.startswith("https://archive.ics.uci.edu/")
    assert OPTDIGITS_SPEC.landing_url.startswith("https://archive.ics.uci.edu/")
    assert COVERTYPE_SPEC.landing_url.startswith("https://archive.ics.uci.edu/")


def test_covertype_parser_supports_plain_and_gzip_without_network(tmp_path: Path) -> None:
    payload = (_covertype_row(1) + "\n" + _covertype_row(7) + "\n").encode("ascii")
    plain_path = tmp_path / "covtype.data"
    gzip_path = tmp_path / "covtype.data.gz"
    plain_path.write_bytes(payload)
    with gzip.open(gzip_path, "wb") as target:
        target.write(payload)

    plain = parse_covertype_source(plain_path, expected_rows=2)
    compressed = parse_covertype_source(gzip_path, expected_rows=2)

    assert plain.features.shape == (2, 54)
    assert plain.labels == compressed.labels == ("1", "7")
    assert np.array_equal(plain.features, compressed.features)
    assert plain.sha256 == hashlib.sha256(payload).hexdigest()
    assert compressed.sha256 == hashlib.sha256(gzip_path.read_bytes()).hexdigest()
    assert not plain.features.flags.writeable
    with pytest.raises(ValueError, match="expected exactly 3"):
        parse_covertype_source(plain_path, expected_rows=3)


def test_covertype_parser_rejects_indicator_and_label_violations(tmp_path: Path) -> None:
    path = tmp_path / "covtype.data"
    fields = _covertype_row(1).split(",")
    fields[11] = "1"
    path.write_text(",".join(fields) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="wilderness indicators are not one-hot"):
        parse_covertype_source(path)

    path.write_text(_covertype_row(8) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="invalid Covertype label"):
        parse_covertype_source(path)


def test_stratified_hash_split_is_complete_stable_and_commits_original_rows() -> None:
    labels = tuple(label for label in ("1", "2", "3") for _ in range(10))
    first = stable_stratified_hash_split("covertype", labels, test_fraction=0.2)
    second = stable_stratified_hash_split("covertype", labels, test_fraction=0.2)

    assert first == second
    assert len(first.train_indices) == 24
    assert len(first.test_indices) == 6
    assert sorted((*first.train_indices, *first.test_indices)) == list(range(30))
    assert set(first.train_indices).isdisjoint(first.test_indices)
    assert first.per_label_counts == (("1", 8, 2), ("2", 8, 2), ("3", 8, 2))
    assert len(first.train_row_index_sha256) == 64
    assert len(first.test_row_index_sha256) == 64
    assert first.as_dict()["test_row_index_sha256"] == first.test_row_index_sha256


def test_official_covertype_dimension_enforcement_is_independently_auditable() -> None:
    validate_official_covertype_shape(581_012, 54)
    with pytest.raises(ValueError, match="581,012 rows and 54 features"):
        validate_official_covertype_shape(581_011, 54)
    with pytest.raises(ValueError, match="581,012 rows and 54 features"):
        validate_official_covertype_shape(581_012, 53)


def test_stable_shards_are_balanced_complete_disjoint_and_label_blind() -> None:
    first = stable_label_blind_test_shards("letter_recognition", 103, n_shards=7)
    second = stable_label_blind_test_shards("letter_recognition", 103, n_shards=7)
    assert first == second
    flattened = [index for shard in first for index in shard.row_indices]
    assert sorted(flattened) == list(range(103))
    assert len(flattened) == len(set(flattened))
    assert max(map(lambda shard: len(shard.row_indices), first)) - min(
        map(lambda shard: len(shard.row_indices), first)
    ) <= 1
    assert all(len(shard.row_indices) > 0 for shard in first)
    # No label vector is accepted by this API; changing labels cannot change it.
    assert "label" not in stable_label_blind_test_shards.__annotations__


def test_arm_selection_is_config_sorted_prediction_deduped_and_outcome_blind() -> None:
    pytest.importorskip("sklearn")
    first_dataset = _xor_dataset(test_labels=("A",) * 10)
    second_dataset = _xor_dataset(test_labels=("B", "A") * 5)

    first = fit_preregistered_classifier_arms(
        first_dataset,
        n_arms=2,
        catalog=tuple(reversed(_dedup_catalog())),
    )
    second = fit_preregistered_classifier_arms(
        second_dataset,
        n_arms=2,
        catalog=_dedup_catalog(),
    )

    assert first.candidate_ids == ("a_ridge_retained", "m_knn_unique")
    assert second.candidate_ids == first.candidate_ids
    assert [arm.prediction_sha256 for arm in second.selected_arms] == [
        arm.prediction_sha256 for arm in first.selected_arms
    ]
    assert [arm.selected_feature_indices for arm in second.selected_arms] == [
        arm.selected_feature_indices for arm in first.selected_arms
    ]
    assert tuple(
        (item.config_id, item.retained_config_id) for item in first.excluded_duplicates
    ) == (("z_ridge_duplicate", "a_ridge_retained"),)
    manifest_text = json.dumps(first.manifest_document(), sort_keys=True)
    assert "accuracy" not in manifest_text and "test_labels" not in manifest_text


def test_feature_ranking_is_stable_train_only_with_index_tie_breaks() -> None:
    matrix = np.asarray(
        [
            [0.0, 1.0, 5.0],
            [0.1, 1.0, 5.0],
            [9.9, 1.0, 5.0],
            [10.0, 1.0, 5.0],
        ]
    )
    labels = ("low", "low", "high", "high")

    ranking = training_feature_ranking(matrix, labels)

    assert ranking == (0, 1, 2)
    assert training_feature_ranking(matrix.copy(), labels) == ranking


def test_tied_exact_topk_boundary_is_rejected_without_jitter() -> None:
    graph = FrozenCandidateGraph.from_ids(("arm-a", "arm-b", "arm-c"))
    fixture = freeze_source_streams(
        graph,
        reward_streams={
            "arm-a": (1, 1, 0, 0),
            "arm-b": (1, 0, 1, 0),
            "arm-c": (0, 0, 0, 0),
        },
        cost_streams={candidate_id: (1.0,) * 4 for candidate_id in graph.candidate_ids},
    )
    shard = StableTestShard(
        shard_id="tie-shard",
        ordinal=0,
        row_indices=(0, 1, 2, 3),
        row_index_sha256=hashlib.sha256(b"tie").hexdigest(),
    )

    with pytest.raises(NonUniqueTopKError) as raised:
        build_exact_topk_frozen_instance(
            family_id="tie-family",
            instance_id="tie-instance",
            shard=shard,
            fixture=fixture,
            k=1,
        )

    assert raised.value.report.reason == "boundary_tie_non_unique_exact_top_k"
    assert raised.value.report.kth_success_count == raised.value.report.next_success_count


@pytest.fixture(scope="module")
def offline_digits_benchmark():
    pytest.importorskip("sklearn")
    dataset = load_sklearn_digits_offline()
    return build_uci_classifier_benchmark(
        dataset,
        n_arms=24,
        k=8,
        n_shards=5,
    )


def test_offline_digits_is_real_local_data_but_not_mislabelled_official(
    offline_digits_benchmark,
) -> None:
    benchmark = offline_digits_benchmark
    dataset = benchmark.dataset
    assert dataset.dataset_id == "sklearn_digits_offline"
    assert dataset.train_features.shape == (1_347, 64)
    assert dataset.test_features.shape == (450, 64)
    assert not dataset.official_source
    assert dataset.split.strategy.startswith("non_official_deterministic_stratified_hash_split")
    assert "sklearn.datasets.load_digits" in dataset.sources[0].file_name
    assert len(benchmark.instances) + len(benchmark.boundary_failures) == 5


def test_manifest_is_stable_complete_and_has_no_outcome_adaptive_fields(
    offline_digits_benchmark,
) -> None:
    manifest = offline_digits_benchmark.manifest
    reconstructed = UCIClassifierBenchmarkManifest(manifest.as_dict())
    assert reconstructed.manifest_hash == manifest.manifest_hash
    document = manifest.as_dict()
    assert document["claim_scope"] == CLAIM_SCOPE
    assert document["dataset"]["license"]["url"] == LICENSE_URL
    assert len(document["dataset"]["sources"][0]["sha256"]) == 64
    assert len(document["arm_selection"]["catalog"]) >= 24
    assert all(
        len(record["prediction_sha256"]) == 64
        for record in document["arm_selection"]["selected_arms"]
    )
    assert document["arm_selection"]["selection_rule"].startswith("sort_config_id")
    text = json.dumps(document, sort_keys=True)
    assert "accuracy" not in text
    assert "configured_means" not in text
    assert "empirical_means" not in text

    repeated = build_uci_classifier_benchmark(
        load_sklearn_digits_offline(),
        n_arms=24,
        k=8,
        n_shards=5,
    )
    assert repeated.manifest.manifest_hash == manifest.manifest_hash


def test_algorithm_view_exposes_neither_means_threshold_nor_truth(
    offline_digits_benchmark,
) -> None:
    benchmark = offline_digits_benchmark
    instance = benchmark.instances[0]
    view = benchmark.open_algorithm_fixture(
        instance.instance_id,
        SourceOracleBudget(max_queries=10, max_cost=10.0),
    )
    public = view.public_document()
    assert set(public) == {
        "instance_id",
        "fixture_manifest_hash",
        "candidate_ids",
        "k",
        "information_regime",
    }
    forbidden = ("mean", "threshold", "truth", "accuracy", "label", "prediction")
    assert not any(token in json.dumps(public).lower() for token in forbidden)
    observation = view.oracle.query(view.candidate_ids[0])
    assert observation.reward in {0, 1}
    assert view.oracle.snapshot().queries_used == 1
