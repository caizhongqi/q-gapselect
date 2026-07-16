"""Fail-closed registry for the strongest quantum composition baselines.

The JSON registry records primary-source theorem interfaces and the compilation
work needed to compare them with Q-GapSelect.  It is deliberately not a proof
engine: a citation, query-bound string, declarative ``implemented`` flag, or
self-reported JSON artifact never establishes fidelity.  The self-report parser
below checks byte integrity and registry reconciliation only.  Coverage remains
disabled until baseline-specific trusted runners and attestations exist.

This separation prevents an analytic proxy from being renamed after a paper
and then counted as a strongest-baseline implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

PrimaryEligibility = Literal[
    "eligible_primary",
    "composition_component",
    "stronger_information_control",
]
InterfaceStatus = Literal[
    "same_interface",
    "requires_charged_reduction",
    "requires_specialization",
    "stronger_information",
]
ImplementationStatus = Literal[
    "not_implemented",
    "analytic_proxy_only",
    "partial_executable",
    "implemented",
]
FidelityStatus = Literal[
    "not_audited",
    "known_nonfaithful",
    "pending_machine_audit",
    "verified",
]
BoundStatus = Literal[
    "not_instantiated",
    "symbolic_template_only",
    "instantiated",
]

PRIMARY_ELIGIBILITY_VALUES = frozenset(
    {"eligible_primary", "composition_component", "stronger_information_control"}
)
INTERFACE_STATUS_VALUES = frozenset(
    {
        "same_interface",
        "requires_charged_reduction",
        "requires_specialization",
        "stronger_information",
    }
)
IMPLEMENTATION_STATUS_VALUES = frozenset(
    {"not_implemented", "analytic_proxy_only", "partial_executable", "implemented"}
)
FIDELITY_STATUS_VALUES = frozenset(
    {"not_audited", "known_nonfaithful", "pending_machine_audit", "verified"}
)
BOUND_STATUS_VALUES = frozenset(
    {"not_instantiated", "symbolic_template_only", "instantiated"}
)

REQUIRED_BASELINE_IDS = (
    "miqae_per_arm_sort",
    "rall_coherent_ae_rounding",
    "wang_qbai_repeated",
    "gao_approximate_k_minimum",
    "vihrovs_unknown_time_vts",
    "jeffery_subroutine_composition",
    "jeffery_loop_composition",
    "low_su_tunable_vtaa",
    "van_apeldoorn_all_marked",
    "exact_value_k_minimum_control",
)

# Identity pins and version-locator pins are deliberately separate.  These
# validate registry strings only: they do not pin PDF bytes or prove that a
# theorem locator accurately describes the cited document.
KNOWN_PRIMARY_SOURCE_IDENTITIES = {
    "fukuzawa_ho_irani_zion_miqae": "https://arxiv.org/abs/2208.14612",
    "rall_coherent_estimation": "https://arxiv.org/abs/2103.09717",
    "wang_you_li_childs_qbai": "https://arxiv.org/abs/2007.07049",
    "gao_ji_wang_approx_kmin": "https://arxiv.org/abs/2412.16586",
    "vihrovs_computation_trees": "https://arxiv.org/abs/2505.22405",
    "jeffery_subroutine_composition": "https://arxiv.org/abs/2209.14146",
    "jeffery_loop_composition": "https://arxiv.org/abs/2605.07518",
    "low_su_tunable_vtaa": "https://arxiv.org/abs/2410.18178",
    "van_apeldoorn_gribling_nieuwboer_all_marked": (
        "https://arxiv.org/abs/2302.10244"
    ),
    "durr_heiligman_hoyer_mhalla_exact_kmin": (
        "https://doi.org/10.1137/050644719"
    ),
}
KNOWN_PRIMARY_SOURCE_VERSION_LOCATORS = {
    "fukuzawa_ho_irani_zion_miqae": (
        "https://arxiv.org/abs/2208.14612v4",
        "arXiv:2208.14612v4",
    ),
    "rall_coherent_estimation": (
        "https://arxiv.org/abs/2103.09717v4",
        "arXiv:2103.09717v4",
    ),
    "wang_you_li_childs_qbai": (
        "https://arxiv.org/abs/2007.07049v2",
        "arXiv:2007.07049v2",
    ),
    "gao_ji_wang_approx_kmin": (
        "https://arxiv.org/abs/2412.16586v1",
        "arXiv:2412.16586v1",
    ),
    "vihrovs_computation_trees": (
        "https://arxiv.org/abs/2505.22405v2",
        "arXiv:2505.22405v2",
    ),
    "jeffery_subroutine_composition": (
        "https://arxiv.org/abs/2209.14146v3",
        "arXiv:2209.14146v3",
    ),
    "jeffery_loop_composition": (
        "https://arxiv.org/abs/2605.07518v1",
        "arXiv:2605.07518v1",
    ),
    "low_su_tunable_vtaa": (
        "https://arxiv.org/abs/2410.18178v2",
        "arXiv:2410.18178v2",
    ),
    "van_apeldoorn_gribling_nieuwboer_all_marked": (
        "https://arxiv.org/abs/2302.10244v3",
        "arXiv:2302.10244v3",
    ),
    "durr_heiligman_hoyer_mhalla_exact_kmin": (
        "https://doi.org/10.1137/050644719",
        "DOI:10.1137/050644719-version-of-record",
    ),
}
KNOWN_PRIMARY_SOURCE_URLS = {
    key: value[0] for key, value in KNOWN_PRIMARY_SOURCE_VERSION_LOCATORS.items()
}

EVIDENCE_ARTIFACT_TYPE = "q_gapselect_baseline_runtime_fidelity_self_report"
EVIDENCE_CHECK_IDS = (
    "implementation_execution",
    "theorem_to_code_mapping",
    "same_interface_conversion",
    "output_relation",
    "query_bound_instantiation",
    "fidelity_regression",
)

_SLUG = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_ARXIV_VERSION = re.compile(r"v[0-9]+$")
_PARSED_SELF_REPORT_SEAL = object()


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_sha256(value: object, name: str) -> str:
    result = _require_string(value, name)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _require_git_oid(value: object, name: str) -> str:
    result = _require_string(value, name)
    if not _GIT_OID.fullmatch(result):
        raise ValueError(f"{name} must be a 40-character lowercase Git object id")
    return result


def _string_tuple(value: object, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(_require_string(item, f"{name} item") for item in value)
    if nonempty and not result:
        raise ValueError(f"{name} cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _strict_object(
    value: object,
    *,
    required: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    keys = frozenset(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        raise ValueError(f"{context} keys mismatch: missing={missing}, extra={extra}")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{context} keys must be strings")
    return cast(Mapping[str, object], value)


def _enum(value: object, name: str, allowed: frozenset[str]) -> str:
    result = _require_string(value, name)
    if result not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return result


def _strict_json_loads(document: bytes, context: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(document.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} must be UTF-8 JSON") from error


def _canonical_sha256(value: object) -> str:
    document = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PrimarySource:
    """Primary-source identity and explicit version-locator strings."""

    source_key: str
    title: str
    url: str
    version_locator: str
    theorem_locator: str
    source_kind: str

    def __post_init__(self) -> None:
        _require_string(self.source_key, "source_key")
        _require_string(self.title, "title")
        _require_string(self.url, "url")
        _require_string(self.version_locator, "version_locator")
        _require_string(self.theorem_locator, "theorem_locator")
        if self.source_kind not in {"arxiv_primary", "peer_reviewed_primary"}:
            raise ValueError("source_kind must identify a primary source")
        expected_identity = KNOWN_PRIMARY_SOURCE_IDENTITIES.get(self.source_key)
        if expected_identity is None:
            raise ValueError(f"unknown pinned primary source: {self.source_key}")
        identity_url = _ARXIV_VERSION.sub("", self.url)
        if identity_url != expected_identity:
            raise ValueError(f"source URL mismatch for {self.source_key}")

    @property
    def identity_pinned(self) -> bool:
        expected = KNOWN_PRIMARY_SOURCE_IDENTITIES.get(self.source_key)
        return expected is not None and _ARXIV_VERSION.sub("", self.url) == expected

    @property
    def version_locator_pinned(self) -> bool:
        expected = KNOWN_PRIMARY_SOURCE_VERSION_LOCATORS.get(self.source_key)
        return expected is not None and (self.url, self.version_locator) == expected


@dataclass(frozen=True, slots=True)
class CanonicalComparisonInterface:
    """The information contract every primary comparison must compile to."""

    interface_id: str
    oracle_model: str
    output_relation: str
    query_unit: str
    public_inputs: tuple[str, ...]
    forbidden_free_information: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(self.interface_id):
            raise ValueError("interface_id must be a lowercase underscore slug")
        _require_string(self.oracle_model, "oracle_model")
        _require_string(self.output_relation, "output_relation")
        _require_string(self.query_unit, "query_unit")
        _string_tuple(self.public_inputs, "public_inputs")
        _string_tuple(self.forbidden_free_information, "forbidden_free_information")


@dataclass(frozen=True, slots=True)
class StrongCompositionBaseline:
    """One source-identified baseline or mandatory composition threat."""

    baseline_id: str
    display_name: str
    category: str
    source: PrimarySource
    oracle_model: str
    output_relation: str
    interface_assumptions: tuple[str, ...]
    query_bound_template: str
    required_compilation_charges: tuple[str, ...]
    primary_eligibility: PrimaryEligibility
    primary_eligibility_reason: str
    interface_status: InterfaceStatus
    implementation_status: ImplementationStatus
    fidelity_status: FidelityStatus
    bound_status: BoundStatus

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(self.baseline_id):
            raise ValueError("baseline_id must be a lowercase underscore slug")
        for field_name in (
            "display_name",
            "category",
            "oracle_model",
            "output_relation",
            "query_bound_template",
            "primary_eligibility_reason",
        ):
            _require_string(getattr(self, field_name), field_name)
        _string_tuple(self.interface_assumptions, "interface_assumptions")
        _string_tuple(self.required_compilation_charges, "required_compilation_charges")
        _enum(self.primary_eligibility, "primary_eligibility", PRIMARY_ELIGIBILITY_VALUES)
        _enum(self.interface_status, "interface_status", INTERFACE_STATUS_VALUES)
        _enum(self.implementation_status, "implementation_status", IMPLEMENTATION_STATUS_VALUES)
        _enum(self.fidelity_status, "fidelity_status", FIDELITY_STATUS_VALUES)
        _enum(self.bound_status, "bound_status", BOUND_STATUS_VALUES)
        if self.primary_eligibility == "stronger_information_control":
            if self.interface_status != "stronger_information":
                raise ValueError("stronger-information controls need stronger_information status")
        elif self.interface_status == "stronger_information":
            raise ValueError("primary rows cannot use the stronger-information interface")

    @property
    def coverage_required(self) -> bool:
        return self.primary_eligibility != "stronger_information_control"


@dataclass(frozen=True, slots=True)
class StrongCompositionRegistry:
    """Validated, machine-readable registry inventory."""

    schema_version: int
    registry_id: str
    canonical_interface: CanonicalComparisonInterface
    declared_required_baseline_ids: tuple[str, ...]
    baselines: tuple[StrongCompositionBaseline, ...]
    declarative_text_is_proof: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must equal 1")
        if not _SLUG.fullmatch(self.registry_id):
            raise ValueError("registry_id must be a lowercase underscore slug")
        if self.declarative_text_is_proof is not False:
            raise ValueError("declarative text cannot be designated as proof")
        _string_tuple(
            self.declared_required_baseline_ids,
            "declared_required_baseline_ids",
        )
        ids = tuple(item.baseline_id for item in self.baselines)
        if not ids:
            raise ValueError("baselines cannot be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("baseline ids must be unique")

    @property
    def by_id(self) -> Mapping[str, StrongCompositionBaseline]:
        return {item.baseline_id: item for item in self.baselines}


@dataclass(frozen=True, slots=True)
class RuntimeFidelityEvidence:
    """Untrusted self-report parsed and reconciled against the registry.

    Neither direct construction nor parsing activates coverage.  The private
    seal records only that the self-report's bytes and schema were checked; it
    is deliberately not a trusted runtime attestation.
    """

    baseline_id: str
    evidence_digest: str
    implementation_executed: bool
    theorem_to_code_mapping_checked: bool
    same_interface_checked: bool
    output_relation_checked: bool
    query_bound_instantiated: bool
    fidelity_tests_passed: bool
    charged_compilation_items: tuple[str, ...]
    evidence_kind: str = "unparsed_direct_construction"
    registry_digest: str = ""
    artifact_path: str = ""
    runtime_test_digest: str = ""
    _verification_seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not _SLUG.fullmatch(self.baseline_id):
            raise ValueError("baseline_id must be a lowercase underscore slug")
        _require_sha256(self.evidence_digest, "evidence_digest")
        if self.evidence_kind not in {
            "unparsed_direct_construction",
            "parsed_self_report_v1",
        }:
            raise ValueError("unknown evidence_kind")
        for field_name in (
            "implementation_executed",
            "theorem_to_code_mapping_checked",
            "same_interface_checked",
            "output_relation_checked",
            "query_bound_instantiated",
            "fidelity_tests_passed",
        ):
            _require_bool(getattr(self, field_name), field_name)
        _string_tuple(
            self.charged_compilation_items,
            "charged_compilation_items",
            nonempty=False,
        )
        if self.registry_digest:
            _require_sha256(self.registry_digest, "registry_digest")
        if self.runtime_test_digest:
            _require_sha256(self.runtime_test_digest, "runtime_test_digest")

    @property
    def self_report_integrity_checked(self) -> bool:
        if (
            self._verification_seal is not _PARSED_SELF_REPORT_SEAL
            or self.evidence_kind != "parsed_self_report_v1"
            or not self.artifact_path
        ):
            return False
        try:
            current_digest = hashlib.sha256(Path(self.artifact_path).read_bytes()).hexdigest()
        except OSError:
            return False
        return current_digest == self.evidence_digest


@dataclass(frozen=True, slots=True)
class BaselineCoverageAudit:
    """Fail-closed coverage decision for one registry row."""

    baseline_id: str
    primary_eligibility: str
    coverage_required: bool
    source_identity_pinned: bool
    source_version_locator_pinned: bool
    runtime_self_report_supplied: bool
    self_report_integrity_checked: bool
    self_report_registry_reconciled: bool
    trusted_runtime_attestation: bool
    implementation_executed: bool
    theorem_to_code_mapping_checked: bool
    same_interface_checked: bool
    output_relation_checked: bool
    query_bound_instantiated: bool
    fidelity_tests_passed: bool
    missing_compilation_charges: tuple[str, ...]
    covered: bool
    blockers: tuple[str, ...]
    declarative_implementation_status: str
    declarative_fidelity_status: str
    declarative_bound_status: str


@dataclass(frozen=True, slots=True)
class StrongCompositionRegistryAudit:
    """Whole-registry coverage result; false until every primary row closes."""

    registry_id: str
    interface_id: str
    inventory_complete: bool
    source_identity_pin_complete: bool
    source_version_locator_pin_complete: bool
    trusted_runtime_attestation_pipeline_implemented: bool
    missing_required_baseline_ids: tuple[str, ...]
    unexpected_baseline_ids: tuple[str, ...]
    entries: tuple[BaselineCoverageAudit, ...]
    uncovered_required_baseline_ids: tuple[str, ...]
    strongest_composition_coverage_complete: bool
    strongest_composition_claimable: bool
    ccf_a_quantum_advantage_claimable: bool
    blockers: tuple[str, ...]
    claim_status: str


def _parse_source(value: object) -> PrimarySource:
    row = _strict_object(
        value,
        required=frozenset(
            {
                "source_key",
                "title",
                "url",
                "version_locator",
                "theorem_locator",
                "source_kind",
            }
        ),
        context="primary_source",
    )
    return PrimarySource(**{key: _require_string(item, key) for key, item in row.items()})


def _parse_interface(value: object) -> CanonicalComparisonInterface:
    row = _strict_object(
        value,
        required=frozenset(
            {
                "interface_id",
                "oracle_model",
                "output_relation",
                "query_unit",
                "public_inputs",
                "forbidden_free_information",
            }
        ),
        context="canonical_interface",
    )
    return CanonicalComparisonInterface(
        interface_id=_require_string(row["interface_id"], "interface_id"),
        oracle_model=_require_string(row["oracle_model"], "oracle_model"),
        output_relation=_require_string(row["output_relation"], "output_relation"),
        query_unit=_require_string(row["query_unit"], "query_unit"),
        public_inputs=_string_tuple(row["public_inputs"], "public_inputs"),
        forbidden_free_information=_string_tuple(
            row["forbidden_free_information"],
            "forbidden_free_information",
        ),
    )


def _parse_baseline(value: object) -> StrongCompositionBaseline:
    required = frozenset(
        {
            "baseline_id",
            "display_name",
            "category",
            "primary_source",
            "oracle_model",
            "output_relation",
            "interface_assumptions",
            "query_bound_template",
            "required_compilation_charges",
            "primary_eligibility",
            "primary_eligibility_reason",
            "interface_status",
            "implementation_status",
            "fidelity_status",
            "bound_status",
        }
    )
    row = _strict_object(value, required=required, context="baseline")
    return StrongCompositionBaseline(
        baseline_id=_require_string(row["baseline_id"], "baseline_id"),
        display_name=_require_string(row["display_name"], "display_name"),
        category=_require_string(row["category"], "category"),
        source=_parse_source(row["primary_source"]),
        oracle_model=_require_string(row["oracle_model"], "oracle_model"),
        output_relation=_require_string(row["output_relation"], "output_relation"),
        interface_assumptions=_string_tuple(
            row["interface_assumptions"], "interface_assumptions"
        ),
        query_bound_template=_require_string(
            row["query_bound_template"], "query_bound_template"
        ),
        required_compilation_charges=_string_tuple(
            row["required_compilation_charges"],
            "required_compilation_charges",
        ),
        primary_eligibility=cast(
            PrimaryEligibility,
            _enum(
                row["primary_eligibility"],
                "primary_eligibility",
                PRIMARY_ELIGIBILITY_VALUES,
            ),
        ),
        primary_eligibility_reason=_require_string(
            row["primary_eligibility_reason"], "primary_eligibility_reason"
        ),
        interface_status=cast(
            InterfaceStatus,
            _enum(row["interface_status"], "interface_status", INTERFACE_STATUS_VALUES),
        ),
        implementation_status=cast(
            ImplementationStatus,
            _enum(
                row["implementation_status"],
                "implementation_status",
                IMPLEMENTATION_STATUS_VALUES,
            ),
        ),
        fidelity_status=cast(
            FidelityStatus,
            _enum(row["fidelity_status"], "fidelity_status", FIDELITY_STATUS_VALUES),
        ),
        bound_status=cast(
            BoundStatus,
            _enum(row["bound_status"], "bound_status", BOUND_STATUS_VALUES),
        ),
    )


def default_registry_path() -> Path:
    """Return the repository's checked-in registry path."""

    return Path(__file__).resolve().parents[2] / "configs" / "strong_composition_registry.json"


def load_strong_composition_registry(
    path: str | Path | None = None,
) -> StrongCompositionRegistry:
    """Load and strictly validate a registry; unknown keys are rejected."""

    source_path = default_registry_path() if path is None else Path(path)
    payload = _strict_json_loads(source_path.read_bytes(), "registry")
    root = _strict_object(
        payload,
        required=frozenset(
            {
                "schema_version",
                "registry_id",
                "canonical_interface",
                "required_baseline_ids",
                "declarative_text_is_proof",
                "baselines",
            }
        ),
        context="registry",
    )
    schema_version = root["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("schema_version must be an integer")
    baselines_raw = root["baselines"]
    if isinstance(baselines_raw, (str, bytes)) or not isinstance(baselines_raw, Sequence):
        raise TypeError("baselines must be a sequence")
    return StrongCompositionRegistry(
        schema_version=schema_version,
        registry_id=_require_string(root["registry_id"], "registry_id"),
        canonical_interface=_parse_interface(root["canonical_interface"]),
        declared_required_baseline_ids=_string_tuple(
            root["required_baseline_ids"], "required_baseline_ids"
        ),
        baselines=tuple(_parse_baseline(value) for value in baselines_raw),
        declarative_text_is_proof=_require_bool(
            root["declarative_text_is_proof"], "declarative_text_is_proof"
        ),
    )


def strong_composition_registry_sha256(registry: StrongCompositionRegistry) -> str:
    """Return the canonical content digest used by evidence bindings."""

    if not isinstance(registry, StrongCompositionRegistry):
        raise TypeError("registry must be a StrongCompositionRegistry")
    return _canonical_sha256(asdict(registry))


def _reconcile(value: object, expected: str, name: str) -> str:
    result = _require_string(value, name)
    if result != expected:
        raise ValueError(f"{name} does not match the registry")
    return result


def load_self_reported_runtime_fidelity_evidence(
    path: str | Path,
    *,
    registry: StrongCompositionRegistry,
    expected_sha256: str,
) -> RuntimeFidelityEvidence:
    """Parse, hash, and reconcile one untrusted self-reported artifact.

    ``expected_sha256`` is mandatory so callers must obtain the content digest
    from a separate trusted manifest or execution step.  The digest is over the
    exact file bytes; the internal runtime-test digest is over canonical JSON.
    Passing this parser establishes neither execution authenticity nor baseline
    fidelity.  Parsed records remain ineligible for coverage until separate,
    baseline-specific trusted runners and attestations are implemented.
    """

    if not isinstance(registry, StrongCompositionRegistry):
        raise TypeError("registry must be a StrongCompositionRegistry")
    expected_digest = _require_sha256(expected_sha256, "expected_sha256")
    source_path = Path(path)
    document = source_path.read_bytes()
    actual_digest = hashlib.sha256(document).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("evidence file SHA-256 does not match expected_sha256")
    root = _strict_object(
        _strict_json_loads(document, "runtime evidence"),
        required=frozenset(
            {
                "schema_version",
                "artifact_type",
                "registry_binding",
                "baseline_contract",
                "charged_compilation_items",
                "provenance",
                "runtime_test",
            }
        ),
        context="runtime evidence",
    )
    if _require_integer(root["schema_version"], "evidence schema_version") != 1:
        raise ValueError("evidence schema_version must equal 1")
    if _require_string(root["artifact_type"], "artifact_type") != EVIDENCE_ARTIFACT_TYPE:
        raise ValueError(f"artifact_type must equal {EVIDENCE_ARTIFACT_TYPE}")

    registry_digest = strong_composition_registry_sha256(registry)
    interface = registry.canonical_interface
    binding = _strict_object(
        root["registry_binding"],
        required=frozenset(
            {
                "registry_id",
                "registry_sha256",
                "interface_id",
                "oracle_model",
                "output_relation",
                "query_unit",
            }
        ),
        context="registry_binding",
    )
    _reconcile(binding["registry_id"], registry.registry_id, "registry_id")
    if _require_sha256(binding["registry_sha256"], "registry_sha256") != registry_digest:
        raise ValueError("registry_sha256 does not match the loaded registry")
    _reconcile(binding["interface_id"], interface.interface_id, "interface_id")
    _reconcile(binding["oracle_model"], interface.oracle_model, "canonical oracle_model")
    _reconcile(
        binding["output_relation"],
        interface.output_relation,
        "canonical output_relation",
    )
    _reconcile(binding["query_unit"], interface.query_unit, "canonical query_unit")

    contract = _strict_object(
        root["baseline_contract"],
        required=frozenset(
            {
                "baseline_id",
                "source_key",
                "source_version_locator",
                "oracle_model",
                "output_relation",
                "query_bound_template",
            }
        ),
        context="baseline_contract",
    )
    baseline_id = _require_string(contract["baseline_id"], "baseline_id")
    baseline = registry.by_id.get(baseline_id)
    if baseline is None:
        raise ValueError(f"evidence baseline_id is not in registry: {baseline_id}")
    _reconcile(contract["source_key"], baseline.source.source_key, "source_key")
    _reconcile(
        contract["source_version_locator"],
        baseline.source.version_locator,
        "source_version_locator",
    )
    _reconcile(contract["oracle_model"], baseline.oracle_model, "baseline oracle_model")
    _reconcile(
        contract["output_relation"],
        baseline.output_relation,
        "baseline output_relation",
    )
    _reconcile(
        contract["query_bound_template"],
        baseline.query_bound_template,
        "query_bound_template",
    )

    charged = _string_tuple(
        root["charged_compilation_items"],
        "charged_compilation_items",
        nonempty=False,
    )
    if charged != baseline.required_compilation_charges:
        raise ValueError(
            "charged_compilation_items must exactly match the registry contract"
        )

    provenance = _strict_object(
        root["provenance"],
        required=frozenset(
            {
                "git_commit",
                "git_tree",
                "source_tree_dirty_at_execution",
                "python_version",
                "platform",
                "runtime_test_sha256",
            }
        ),
        context="provenance",
    )
    _require_git_oid(provenance["git_commit"], "git_commit")
    _require_git_oid(provenance["git_tree"], "git_tree")
    if _require_bool(
        provenance["source_tree_dirty_at_execution"],
        "source_tree_dirty_at_execution",
    ):
        raise ValueError("self report must declare a clean source tree")
    _require_string(provenance["python_version"], "python_version")
    _require_string(provenance["platform"], "platform")
    declared_test_digest = _require_sha256(
        provenance["runtime_test_sha256"], "runtime_test_sha256"
    )

    runtime_test = _strict_object(
        root["runtime_test"],
        required=frozenset(
            {
                "framework",
                "command",
                "exit_code",
                "tests_collected",
                "tests_passed",
                "tests_failed",
                "tests_errored",
                "checks",
            }
        ),
        context="runtime_test",
    )
    if _require_string(runtime_test["framework"], "framework") != "pytest":
        raise ValueError("runtime_test framework must equal pytest")
    command = _string_tuple(runtime_test["command"], "runtime_test command")
    if not any("pytest" in item for item in command):
        raise ValueError("runtime_test command must invoke pytest")
    if _require_integer(runtime_test["exit_code"], "exit_code") != 0:
        raise ValueError("runtime_test exit_code must equal zero")
    collected = _require_integer(runtime_test["tests_collected"], "tests_collected", minimum=1)
    passed = _require_integer(runtime_test["tests_passed"], "tests_passed")
    failed = _require_integer(runtime_test["tests_failed"], "tests_failed")
    errored = _require_integer(runtime_test["tests_errored"], "tests_errored")
    if passed != collected or failed != 0 or errored != 0:
        raise ValueError("runtime_test must report every collected test passed")

    raw_checks = runtime_test["checks"]
    if isinstance(raw_checks, (str, bytes)) or not isinstance(raw_checks, Sequence):
        raise TypeError("runtime_test checks must be a sequence")
    check_ids: list[str] = []
    check_digests: list[str] = []
    for raw_check in raw_checks:
        check = _strict_object(
            raw_check,
            required=frozenset({"check_id", "status", "evidence_sha256"}),
            context="runtime_test check",
        )
        check_ids.append(_require_string(check["check_id"], "check_id"))
        if _require_string(check["status"], "check status") != "passed":
            raise ValueError("every required runtime check must have passed")
        check_digests.append(
            _require_sha256(check["evidence_sha256"], "check evidence_sha256")
        )
    if tuple(check_ids) != EVIDENCE_CHECK_IDS:
        raise ValueError("runtime_test checks must exactly match the required check ids")
    if len(set(check_digests)) != len(check_digests):
        raise ValueError("runtime_test check evidence digests must be distinct")
    if collected < len(EVIDENCE_CHECK_IDS):
        raise ValueError("tests_collected cannot be smaller than required checks")
    runtime_test_digest = _canonical_sha256(runtime_test)
    if declared_test_digest != runtime_test_digest:
        raise ValueError("runtime_test_sha256 does not match runtime_test content")

    evidence = RuntimeFidelityEvidence(
        baseline_id=baseline_id,
        evidence_digest=actual_digest,
        implementation_executed=True,
        theorem_to_code_mapping_checked=True,
        same_interface_checked=True,
        output_relation_checked=True,
        query_bound_instantiated=True,
        fidelity_tests_passed=True,
        charged_compilation_items=charged,
        evidence_kind="parsed_self_report_v1",
        registry_digest=registry_digest,
        artifact_path=str(source_path.resolve()),
        runtime_test_digest=runtime_test_digest,
    )
    object.__setattr__(evidence, "_verification_seal", _PARSED_SELF_REPORT_SEAL)
    return evidence


def _audit_entry(
    baseline: StrongCompositionBaseline,
    evidence: RuntimeFidelityEvidence | None,
    registry_digest: str,
) -> BaselineCoverageAudit:
    required = baseline.coverage_required
    supplied = evidence is not None
    integrity_checked = bool(evidence and evidence.self_report_integrity_checked)
    registry_reconciled = bool(
        evidence
        and integrity_checked
        and evidence.registry_digest == registry_digest
    )
    # There is intentionally no positive activation path.  A self-report can
    # be internally consistent and still be entirely fabricated.  Future
    # baseline-specific trusted runners must introduce a distinct attestation
    # type before any of these fields may become active.
    trusted_attestation = False
    executed = False
    mapping = False
    same_interface = False
    output_checked = False
    bound = False
    fidelity = False
    charged: frozenset[str] = frozenset()
    missing_charges = tuple(
        item for item in baseline.required_compilation_charges if item not in charged
    )
    blockers: list[str] = []
    if required:
        if not baseline.source.identity_pinned:
            blockers.append("primary_source_identity_not_pinned")
        if not baseline.source.version_locator_pinned:
            blockers.append("primary_source_version_locator_not_pinned")
        if not supplied:
            blockers.append("runtime_self_report_missing")
        elif not integrity_checked:
            blockers.append("runtime_self_report_integrity_not_checked")
        elif not registry_reconciled:
            blockers.append("runtime_self_report_not_reconciled_to_registry")
        blockers.append("trusted_runtime_attestation_not_implemented")
        checks = (
            (executed, "implementation_not_machine_executed"),
            (mapping, "theorem_to_code_mapping_not_machine_checked"),
            (same_interface, "same_interface_conversion_not_machine_checked"),
            (output_checked, "output_relation_not_machine_checked"),
            (bound, "query_bound_not_instantiated"),
            (fidelity, "fidelity_tests_not_passed"),
        )
        blockers.extend(message for passed, message in checks if not passed)
        blockers.extend(f"compilation_charge_missing:{item}" for item in missing_charges)
    else:
        blockers.append("stronger_information_control_not_primary_eligible")
    covered = required and not blockers
    return BaselineCoverageAudit(
        baseline_id=baseline.baseline_id,
        primary_eligibility=baseline.primary_eligibility,
        coverage_required=required,
        source_identity_pinned=baseline.source.identity_pinned,
        source_version_locator_pinned=baseline.source.version_locator_pinned,
        runtime_self_report_supplied=supplied,
        self_report_integrity_checked=integrity_checked,
        self_report_registry_reconciled=registry_reconciled,
        trusted_runtime_attestation=trusted_attestation,
        implementation_executed=executed,
        theorem_to_code_mapping_checked=mapping,
        same_interface_checked=same_interface,
        output_relation_checked=output_checked,
        query_bound_instantiated=bound,
        fidelity_tests_passed=fidelity,
        missing_compilation_charges=missing_charges,
        covered=covered,
        blockers=tuple(blockers),
        declarative_implementation_status=baseline.implementation_status,
        declarative_fidelity_status=baseline.fidelity_status,
        declarative_bound_status=baseline.bound_status,
    )


def audit_strong_composition_registry(
    registry: StrongCompositionRegistry,
    runtime_self_reports: Sequence[RuntimeFidelityEvidence] = (),
) -> StrongCompositionRegistryAudit:
    """Audit inventory and fidelity coverage without treating prose as proof."""

    if not isinstance(registry, StrongCompositionRegistry):
        raise TypeError("registry must be a StrongCompositionRegistry")
    if isinstance(runtime_self_reports, (str, bytes)) or not isinstance(
        runtime_self_reports, Sequence
    ):
        raise TypeError("runtime_self_reports must be a sequence")
    self_report_rows = tuple(runtime_self_reports)
    if any(not isinstance(item, RuntimeFidelityEvidence) for item in self_report_rows):
        raise TypeError(
            "runtime_self_reports must contain RuntimeFidelityEvidence values"
        )
    self_reports_by_id = {item.baseline_id: item for item in self_report_rows}
    if len(self_reports_by_id) != len(self_report_rows):
        raise ValueError("runtime self-report baseline ids must be unique")

    expected = frozenset(REQUIRED_BASELINE_IDS)
    declared = frozenset(registry.declared_required_baseline_ids)
    present = frozenset(registry.by_id)
    missing = tuple(sorted(expected - present))
    unexpected = tuple(sorted(present - expected))
    unknown_self_reports = tuple(sorted(frozenset(self_reports_by_id) - present))
    inventory_complete = not missing and not unexpected and declared == expected
    source_identity_pin_complete = all(
        item.source.identity_pinned for item in registry.baselines
    )
    source_version_locator_pin_complete = all(
        item.source.version_locator_pinned for item in registry.baselines
    )
    registry_digest = strong_composition_registry_sha256(registry)

    entries = tuple(
        _audit_entry(item, self_reports_by_id.get(item.baseline_id), registry_digest)
        for item in registry.baselines
    )
    uncovered = tuple(
        item.baseline_id for item in entries if item.coverage_required and not item.covered
    )
    blockers: list[str] = []
    blockers.extend(f"missing_required_baseline:{item}" for item in missing)
    blockers.extend(f"unexpected_baseline:{item}" for item in unexpected)
    if declared != expected:
        blockers.append("declared_required_ids_do_not_match_code_pinned_inventory")
    if not source_identity_pin_complete:
        blockers.append("primary_source_identity_inventory_not_fully_pinned")
    if not source_version_locator_pin_complete:
        blockers.append("primary_source_version_locator_inventory_not_fully_pinned")
    blockers.append("trusted_runtime_attestation_pipeline_not_implemented")
    blockers.extend(
        f"runtime_self_report_for_unknown_baseline:{item}"
        for item in unknown_self_reports
    )
    blockers.extend(f"uncovered_required_baseline:{item}" for item in uncovered)
    # Fail closed even if nine syntactically perfect self-reports are supplied.
    # No trusted, baseline-specific runner/attestation pipeline exists yet.
    complete = False
    return StrongCompositionRegistryAudit(
        registry_id=registry.registry_id,
        interface_id=registry.canonical_interface.interface_id,
        inventory_complete=inventory_complete,
        source_identity_pin_complete=source_identity_pin_complete,
        source_version_locator_pin_complete=source_version_locator_pin_complete,
        trusted_runtime_attestation_pipeline_implemented=False,
        missing_required_baseline_ids=missing,
        unexpected_baseline_ids=unexpected,
        entries=entries,
        uncovered_required_baseline_ids=uncovered,
        strongest_composition_coverage_complete=complete,
        strongest_composition_claimable=complete,
        ccf_a_quantum_advantage_claimable=False,
        blockers=tuple(blockers),
        claim_status="blocked_trusted_runtime_attestation_not_implemented",
    )


def machine_readable_registry_audit(
    audit: StrongCompositionRegistryAudit,
) -> dict[str, object]:
    """Serialize an audit with an explicit non-proof disclaimer and digest."""

    if not isinstance(audit, StrongCompositionRegistryAudit):
        raise TypeError("audit must be a StrongCompositionRegistryAudit")
    payload = cast(dict[str, object], asdict(audit))
    payload["query_bound_templates_are_proofs"] = False
    payload["registry_strings_are_fidelity_evidence"] = False
    payload["evidence_file_integrity_is_theorem_fidelity"] = False
    payload["self_report_integrity_is_trusted_attestation"] = False
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["audit_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload
