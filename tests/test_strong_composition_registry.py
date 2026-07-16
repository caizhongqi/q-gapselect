from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from qgapselect.strong_composition_registry import (
    EVIDENCE_ARTIFACT_TYPE,
    EVIDENCE_CHECK_IDS,
    KNOWN_PRIMARY_SOURCE_URLS,
    REQUIRED_BASELINE_IDS,
    RuntimeFidelityEvidence,
    StrongCompositionRegistry,
    audit_strong_composition_registry,
    default_registry_path,
    load_self_reported_runtime_fidelity_evidence,
    load_strong_composition_registry,
    machine_readable_registry_audit,
    strong_composition_registry_sha256,
)


def _canonical_sha256(value: object) -> str:
    document = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _evidence_payload(
    registry: StrongCompositionRegistry,
    baseline_id: str,
) -> dict[str, object]:
    interface = registry.canonical_interface
    baseline = registry.by_id[baseline_id]
    runtime_test: dict[str, object] = {
        "framework": "pytest",
        "command": ["python", "-m", "pytest", f"tests/fidelity/{baseline_id}.py"],
        "exit_code": 0,
        "tests_collected": len(EVIDENCE_CHECK_IDS),
        "tests_passed": len(EVIDENCE_CHECK_IDS),
        "tests_failed": 0,
        "tests_errored": 0,
        "checks": [
            {
                "check_id": check_id,
                "status": "passed",
                "evidence_sha256": hashlib.sha256(
                    f"{baseline_id}:{check_id}".encode()
                ).hexdigest(),
            }
            for check_id in EVIDENCE_CHECK_IDS
        ],
    }
    return {
        "schema_version": 1,
        "artifact_type": EVIDENCE_ARTIFACT_TYPE,
        "registry_binding": {
            "registry_id": registry.registry_id,
            "registry_sha256": strong_composition_registry_sha256(registry),
            "interface_id": interface.interface_id,
            "oracle_model": interface.oracle_model,
            "output_relation": interface.output_relation,
            "query_unit": interface.query_unit,
        },
        "baseline_contract": {
            "baseline_id": baseline.baseline_id,
            "source_key": baseline.source.source_key,
            "source_version_locator": baseline.source.version_locator,
            "oracle_model": baseline.oracle_model,
            "output_relation": baseline.output_relation,
            "query_bound_template": baseline.query_bound_template,
        },
        "charged_compilation_items": list(baseline.required_compilation_charges),
        "provenance": {
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "source_tree_dirty_at_execution": False,
            "python_version": "3.12.0",
            "platform": "test-platform",
            "runtime_test_sha256": _canonical_sha256(runtime_test),
        },
        "runtime_test": runtime_test,
    }


def _write_and_parse_self_report(
    tmp_path: Path,
    registry: StrongCompositionRegistry,
    baseline_id: str,
    *,
    payload: dict[str, object] | None = None,
) -> RuntimeFidelityEvidence:
    path = tmp_path / f"{baseline_id}.json"
    path.write_text(
        json.dumps(payload or _evidence_payload(registry, baseline_id), sort_keys=True),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_self_reported_runtime_fidelity_evidence(
        path,
        registry=registry,
        expected_sha256=digest,
    )


def test_default_registry_has_identity_and_version_locator_inventory() -> None:
    registry = load_strong_composition_registry()
    audit = audit_strong_composition_registry(registry)

    assert registry.schema_version == 1
    assert registry.declarative_text_is_proof is False
    assert registry.declared_required_baseline_ids == REQUIRED_BASELINE_IDS
    assert tuple(registry.by_id) == REQUIRED_BASELINE_IDS
    assert len(registry.baselines) == 10
    assert {
        baseline.source.source_key: baseline.source.url
        for baseline in registry.baselines
    } == KNOWN_PRIMARY_SOURCE_URLS
    assert all(source.source.identity_pinned for source in registry.baselines)
    assert all(source.source.version_locator_pinned for source in registry.baselines)
    assert audit.source_identity_pin_complete
    assert audit.source_version_locator_pin_complete


def test_canonical_interface_does_not_reveal_fixture_family_or_gap() -> None:
    registry = load_strong_composition_registry()
    interface = registry.canonical_interface

    assert interface.public_inputs == ("n", "k", "delta", "atomic_query_cap")
    assert "public_family_promise" not in interface.public_inputs
    assert {
        "fixture_family_or_label",
        "gap_values_or_scales",
        "tie_pattern",
        "stopping_schedule",
    }.issubset(interface.forbidden_free_information)
    assert "B_theta" in interface.oracle_model
    assert "theta_i in [0, pi/2]" in interface.oracle_model
    assert "mu_i = sin^2(theta_i)" in interface.oracle_model
    assert "mean order is monotone" in interface.oracle_model
    assert "complete exact Top-k" in interface.output_relation
    assert "INCONCLUSIVE" in interface.output_relation
    assert "No completeness guarantee" in interface.output_relation
    assert "ties and non-promise instances" in interface.output_relation
    assert "at most delta" in interface.output_relation
    assert all(baseline.required_compilation_charges for baseline in registry.baselines)
    exact = registry.by_id["exact_value_k_minimum_control"]
    assert "Theorem 3.4" in exact.source.theorem_locator


def test_unversioned_identity_is_not_misreported_as_version_locator_pinned(
    tmp_path: Path,
) -> None:
    payload = json.loads(default_registry_path().read_text(encoding="utf-8"))
    source = payload["baselines"][0]["primary_source"]
    source["url"] = "https://arxiv.org/abs/2208.14612"
    source["version_locator"] = "unverified-current-version"
    path = tmp_path / "unversioned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    registry = load_strong_composition_registry(path)
    audit = audit_strong_composition_registry(registry)
    row = audit.entries[0]

    assert row.source_identity_pinned
    assert not row.source_version_locator_pinned
    assert not audit.source_version_locator_pin_complete
    assert "primary_source_version_locator_not_pinned" in row.blockers
    assert not audit.strongest_composition_claimable


def test_default_audit_fails_closed_and_names_every_uncovered_row() -> None:
    registry = load_strong_composition_registry()
    audit = audit_strong_composition_registry(registry)
    rows = {row.baseline_id: row for row in audit.entries}

    assert audit.inventory_complete
    assert len(audit.uncovered_required_baseline_ids) == 9
    assert not audit.strongest_composition_coverage_complete
    assert not audit.strongest_composition_claimable
    assert not audit.ccf_a_quantum_advantage_claimable
    assert all(
        f"uncovered_required_baseline:{baseline_id}" in audit.blockers
        for baseline_id in audit.uncovered_required_baseline_ids
    )
    assert "runtime_self_report_missing" in rows["miqae_per_arm_sort"].blockers
    assert (
        "trusted_runtime_attestation_not_implemented"
        in rows["miqae_per_arm_sort"].blockers
    )
    assert rows["exact_value_k_minimum_control"].blockers == (
        "stronger_information_control_not_primary_eligible",
    )


def test_directly_constructed_true_booleans_cannot_activate_coverage() -> None:
    registry = load_strong_composition_registry()
    baseline = registry.by_id["miqae_per_arm_sort"]
    direct = RuntimeFidelityEvidence(
        baseline_id=baseline.baseline_id,
        evidence_digest="a" * 64,
        implementation_executed=True,
        theorem_to_code_mapping_checked=True,
        same_interface_checked=True,
        output_relation_checked=True,
        query_bound_instantiated=True,
        fidelity_tests_passed=True,
        charged_compilation_items=baseline.required_compilation_charges,
    )

    audit = audit_strong_composition_registry(registry, (direct,))
    row = audit.entries[0]

    assert not direct.self_report_integrity_checked
    assert row.runtime_self_report_supplied
    assert not row.self_report_integrity_checked
    assert not row.self_report_registry_reconciled
    assert not row.trusted_runtime_attestation
    assert "runtime_self_report_integrity_not_checked" in row.blockers
    assert "trusted_runtime_attestation_not_implemented" in row.blockers
    assert not row.covered
    assert not audit.strongest_composition_claimable


def test_strict_parser_reconciles_but_never_activates_row(tmp_path: Path) -> None:
    registry = load_strong_composition_registry()
    evidence = _write_and_parse_self_report(tmp_path, registry, "miqae_per_arm_sort")

    audit = audit_strong_composition_registry(registry, (evidence,))
    rows = {row.baseline_id: row for row in audit.entries}

    assert evidence.self_report_integrity_checked
    assert rows["miqae_per_arm_sort"].self_report_registry_reconciled
    assert not rows["miqae_per_arm_sort"].trusted_runtime_attestation
    assert not rows["miqae_per_arm_sort"].covered
    assert not rows["gao_approximate_k_minimum"].covered
    assert len(audit.uncovered_required_baseline_ids) == 9
    assert not audit.strongest_composition_claimable


def test_post_parse_artifact_tamper_revokes_integrity_flag(tmp_path: Path) -> None:
    registry = load_strong_composition_registry()
    evidence = _write_and_parse_self_report(tmp_path, registry, "miqae_per_arm_sort")
    Path(evidence.artifact_path).write_text('{"tampered":true}', encoding="utf-8")

    audit = audit_strong_composition_registry(registry, (evidence,))
    row = audit.entries[0]

    assert not evidence.self_report_integrity_checked
    assert not row.self_report_integrity_checked
    assert not row.covered


def test_parser_rejects_wrong_file_digest_and_post_digest_tamper(tmp_path: Path) -> None:
    registry = load_strong_composition_registry()
    path = tmp_path / "evidence.json"
    payload = _evidence_payload(registry, "miqae_per_arm_sort")
    path.write_text(json.dumps(payload), encoding="utf-8")
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="does not match expected_sha256"):
        load_self_reported_runtime_fidelity_evidence(
            path,
            registry=registry,
            expected_sha256="0" * 64,
        )

    payload["baseline_contract"]["output_relation"] = "tampered output"  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match expected_sha256"):
        load_self_reported_runtime_fidelity_evidence(
            path,
            registry=registry,
            expected_sha256=original_digest,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("registry_binding", "interface_id", "wrong_interface", "interface_id"),
        ("registry_binding", "query_unit", "free queries", "query_unit"),
        ("baseline_contract", "output_relation", "wrong output", "output_relation"),
        ("baseline_contract", "query_bound_template", "O(1)", "query_bound_template"),
    ],
)
def test_parser_rejects_rehashed_contract_tampering(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: str,
    message: str,
) -> None:
    registry = load_strong_composition_registry()
    payload = _evidence_payload(registry, "miqae_per_arm_sort")
    payload[section][field] = replacement  # type: ignore[index]
    path = tmp_path / f"{section}-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_self_reported_runtime_fidelity_evidence(
            path,
            registry=registry,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_parser_rejects_missing_compilation_charge(tmp_path: Path) -> None:
    registry = load_strong_composition_registry()
    payload = _evidence_payload(registry, "van_apeldoorn_all_marked")
    payload["charged_compilation_items"] = payload["charged_compilation_items"][:-1]  # type: ignore[index]
    path = tmp_path / "missing-charge.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match"):
        load_self_reported_runtime_fidelity_evidence(
            path,
            registry=registry,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("failure", ["dirty", "test_failure", "test_digest"])
def test_parser_rejects_invalid_provenance_or_runtime_test(
    tmp_path: Path,
    failure: str,
) -> None:
    registry = load_strong_composition_registry()
    payload = _evidence_payload(registry, "miqae_per_arm_sort")
    if failure == "dirty":
        payload["provenance"]["source_tree_dirty_at_execution"] = True  # type: ignore[index]
    elif failure == "test_failure":
        payload["runtime_test"]["tests_failed"] = 1  # type: ignore[index]
        payload["runtime_test"]["tests_passed"] = len(EVIDENCE_CHECK_IDS) - 1  # type: ignore[index]
        payload["provenance"]["runtime_test_sha256"] = _canonical_sha256(  # type: ignore[index]
            payload["runtime_test"]
        )
    else:
        payload["provenance"]["runtime_test_sha256"] = "f" * 64  # type: ignore[index]
    path = tmp_path / f"{failure}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_self_reported_runtime_fidelity_evidence(
            path,
            registry=registry,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_parser_rejects_unknown_keys_and_duplicate_json_keys(tmp_path: Path) -> None:
    registry = load_strong_composition_registry()
    payload = _evidence_payload(registry, "miqae_per_arm_sort")
    payload["claim"] = "trust me"
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_self_reported_runtime_fidelity_evidence(
            extra,
            registry=registry,
            expected_sha256=hashlib.sha256(extra.read_bytes()).hexdigest(),
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_self_reported_runtime_fidelity_evidence(
            duplicate,
            registry=registry,
            expected_sha256=hashlib.sha256(duplicate.read_bytes()).hexdigest(),
        )


def test_nine_forged_self_reports_cannot_close_any_coverage(
    tmp_path: Path,
) -> None:
    registry = load_strong_composition_registry()
    evidence = tuple(
        _write_and_parse_self_report(tmp_path, registry, row.baseline_id)
        for row in registry.baselines
        if row.coverage_required
    )

    audit = audit_strong_composition_registry(registry, evidence)

    assert all(item.self_report_integrity_checked for item in evidence)
    assert all(
        row.self_report_registry_reconciled
        for row in audit.entries
        if row.coverage_required
    )
    assert all(not row.covered for row in audit.entries if row.coverage_required)
    assert len(audit.uncovered_required_baseline_ids) == 9
    assert not audit.trusted_runtime_attestation_pipeline_implemented
    assert not audit.strongest_composition_coverage_complete
    assert not audit.strongest_composition_claimable
    assert not audit.ccf_a_quantum_advantage_claimable
    assert "trusted_runtime_attestation_pipeline_not_implemented" in audit.blockers


def test_declarative_status_strings_are_not_evidence() -> None:
    registry = load_strong_composition_registry()
    promoted = replace(
        registry,
        baselines=tuple(
            replace(
                row,
                implementation_status="implemented",
                fidelity_status="verified",
                bound_status="instantiated",
            )
            for row in registry.baselines
        ),
    )

    audit = audit_strong_composition_registry(promoted)

    assert len(audit.uncovered_required_baseline_ids) == 9
    assert not audit.strongest_composition_claimable


def test_machine_readable_audit_disclaims_prose_and_file_integrity_as_proof() -> None:
    audit = audit_strong_composition_registry(load_strong_composition_registry())
    first = machine_readable_registry_audit(audit)
    second = machine_readable_registry_audit(audit)

    assert first == second
    assert first["query_bound_templates_are_proofs"] is False
    assert first["registry_strings_are_fidelity_evidence"] is False
    assert first["evidence_file_integrity_is_theorem_fidelity"] is False
    assert len(first["audit_digest"]) == 64


def test_registry_loader_rejects_unknown_keys_and_wrong_source_identity(
    tmp_path: Path,
) -> None:
    original = json.loads(default_registry_path().read_text(encoding="utf-8"))
    with_extra = dict(original)
    with_extra["proof"] = "a string"
    extra_path = tmp_path / "extra-registry.json"
    extra_path.write_text(json.dumps(with_extra), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_strong_composition_registry(extra_path)

    original["baselines"][0]["primary_source"]["url"] = "https://example.test/paper"
    source_path = tmp_path / "bad-source.json"
    source_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="source URL mismatch"):
        load_strong_composition_registry(source_path)


@pytest.mark.parametrize(
    "call",
    [
        lambda: audit_strong_composition_registry(object()),  # type: ignore[arg-type]
        lambda: audit_strong_composition_registry(
            load_strong_composition_registry(), "not evidence"  # type: ignore[arg-type]
        ),
        lambda: RuntimeFidelityEvidence(
            baseline_id="miqae_per_arm_sort",
            evidence_digest="claim text",
            implementation_executed=True,
            theorem_to_code_mapping_checked=True,
            same_interface_checked=True,
            output_relation_checked=True,
            query_bound_instantiated=True,
            fidelity_tests_passed=True,
            charged_compilation_items=(),
        ),
        lambda: strong_composition_registry_sha256(object()),  # type: ignore[arg-type]
        lambda: machine_readable_registry_audit(object()),  # type: ignore[arg-type]
    ],
)
def test_audit_inputs_are_strict(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]
