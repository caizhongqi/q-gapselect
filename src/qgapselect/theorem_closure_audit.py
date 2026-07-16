"""Unified upper/composition/lower-bound closure audit.

The older theory scaffolds use the same scale parameter ``m`` but do not define
one common input set, oracle, output relation, and information interface.  This
module makes that mismatch executable.  It instantiates the charged-history
proxy on a concrete layered block family and checks two exhaustive interface
cases:

* a public static block partition, where known selection composition is legal;
* a hidden partition, where the advertised activity-history cost still needs a
  discovery transducer and therefore is not a proved upper bound.

The module is deliberately a falsification/closure audit.  It does not turn a
query proxy into a theorem and never reports a quantum-advantage claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import asdict, dataclass
from typing import Literal

from .variable_time_charged_history import (
    VariableTimeChargedHistoryRecord,
    variable_time_charged_history_record,
)

CLAIM_STATUS = "unified_theorem_chain_audited_not_closed"
CLOSURE_BLOCKED = "no_single_interface_closes_upper_composition_and_lower_bound"
PUBLIC_PARTITION_MATCH = "candidate_matched_by_public_partition_composition"
HIDDEN_PARTITION_OPEN = "hidden_partition_activity_discovery_upper_bound_open"

PartitionVisibility = Literal["public_static", "hidden_instance_dependent"]
StoppingTimeVisibility = Literal["known", "unknown"]
ObligationStatus = Literal[
    "proved_local_lemma",
    "known_primary_source",
    "falsified",
    "proof_obligation",
    "blocked",
]


PRIMARY_SOURCES = {
    "ambainis_variable_time_amplification": "https://arxiv.org/abs/1010.4458",
    "akv_variable_time_search": "https://arxiv.org/abs/2302.06749",
    "vihrovs_computation_trees": "https://arxiv.org/abs/2505.22405",
    "jeffery_subroutine_composition": "https://arxiv.org/abs/2209.14146",
    "belovs_jeffery_yolcu_transducers": "https://arxiv.org/abs/2311.15873",
    "jeffery_et_al_loop_composition": "https://arxiv.org/abs/2605.07518",
    "wang_et_al_quantum_bai": "https://arxiv.org/abs/2007.07049",
    "gao_ji_wang_approximate_k_minimum": "https://arxiv.org/abs/2412.16586",
    "van_apeldoorn_et_al_all_marked": "https://arxiv.org/abs/2302.10244",
    "klauck_spalek_de_wolf_direct_product": (
        "https://arxiv.org/abs/quant-ph/0402123"
    ),
    "ambainis_search_direct_product": "https://arxiv.org/abs/quant-ph/0508200",
    "lee_roland_direct_product": "https://arxiv.org/abs/1104.4468",
}


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class UnifiedQueryInterface:
    """One oracle/information contract shared by every compared quantity."""

    oracle_id: str
    output_relation_id: str
    query_unit: str
    failure_probability: float
    partition_visibility: PartitionVisibility
    stopping_time_visibility: StoppingTimeVisibility
    boundary_supplied: bool
    active_history_supplied: bool
    qram_history_supplied: bool

    @property
    def interface_id(self) -> str:
        return "qgap-interface-v1-" + _canonical_digest(asdict(self))[:20]


@dataclass(frozen=True, slots=True)
class LayeredBlock:
    """One exact-output block in the concrete hard-family schema."""

    level: int
    candidate_count: int
    required_output_count: int
    angular_separation: float
    checker_query_units: int


@dataclass(frozen=True, slots=True)
class UnifiedHardFamily:
    """Concrete input-set schema replacing a count-only proxy family.

    Each block contains ``required_output_count`` high-angle arms and the rest
    low-angle arms.  The exact global Top-k set is the union of the high arms.
    The remaining ``far_arm_count`` arms lie below every block.  A global angle
    shift is hidden, so no numeric Top-k boundary is supplied.
    """

    family_schema: str
    m: int
    n: int
    k: int
    far_arm_count: int
    hidden_global_angle_shift: bool
    center_angle_interval: tuple[float, float]
    far_arm_angle: float
    blocks: tuple[LayeredBlock, ...]
    interface: UnifiedQueryInterface

    @property
    def base_family_id(self) -> str:
        payload = {
            "family_schema": self.family_schema,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "far_arm_count": self.far_arm_count,
            "hidden_global_angle_shift": self.hidden_global_angle_shift,
            "center_angle_interval": self.center_angle_interval,
            "far_arm_angle": self.far_arm_angle,
            "blocks": [asdict(block) for block in self.blocks],
        }
        return "qgap-layered-block-v1-" + _canonical_digest(payload)[:20]

    @property
    def family_id(self) -> str:
        payload = {
            "base_family_id": self.base_family_id,
            "interface_id": self.interface.interface_id,
        }
        return "qgap-family-contract-v1-" + _canonical_digest(payload)[:20]


@dataclass(frozen=True, slots=True)
class ComplexityInstantiation:
    """Upper, composition, and lower-bound quantities on one family contract."""

    family_id: str
    interface_id: str
    candidate_proxy: float
    candidate_upper_proved: bool
    candidate_upper_status: str
    partitioned_selection_composition_proxy: float
    partitioned_composition_same_interface: bool
    partitioned_over_candidate: float
    partitioned_matches_candidate: bool
    global_all_marked_proxy: float
    published_direct_product_floor: float
    weighted_direct_product_target: float
    weighted_direct_product_proved: bool
    strict_composition_advantage_established: bool
    matching_lower_bound_established: bool
    closure_status: str


@dataclass(frozen=True, slots=True)
class StoppingTimeDichotomy:
    """Known-time/unknown-time consequence for a branch-RMS claim."""

    known_time_rms_covered_by_prior_work: bool
    unknown_time_universal_rms_falsified_by_prior_lower_bound: bool
    published_known_time_scale: str
    published_unknown_time_upper_scale: str
    published_unknown_time_lower_scale: str
    special_structure_escape_status: str
    source_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DefaultAsymptoticWitness:
    """Closed-form exponent comparison for the configured default profile."""

    family_assumptions: str
    candidate_proxy_exponent: float
    partitioned_composition_upper_exponent: float
    composition_over_candidate_exponent: float
    partitioned_is_little_o_of_candidate_proxy: bool
    proof_status: str


@dataclass(frozen=True, slots=True)
class TheoremObligation:
    """One dependency-checked theorem-chain item."""

    obligation_id: str
    status: ObligationStatus
    statement: str
    depends_on: tuple[str, ...]
    evidence: str
    falsifier: str
    source_urls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UnifiedTheoremClosureAudit:
    """Complete two-interface closure audit for one value of ``m``."""

    m: int
    public_family: UnifiedHardFamily
    hidden_family: UnifiedHardFamily
    public_instantiation: ComplexityInstantiation
    hidden_instantiation: ComplexityInstantiation
    stopping_time_dichotomy: StoppingTimeDichotomy
    default_asymptotic_witness: DefaultAsymptoticWitness
    obligations: tuple[TheoremObligation, ...]
    dependency_graph_valid: bool
    theorem_claimable: bool
    ccf_a_quantum_advantage_claimable: bool
    closure_status: str = CLOSURE_BLOCKED
    claim_status: str = CLAIM_STATUS


def _interface(
    *,
    partition_visibility: PartitionVisibility,
    failure_probability: float,
) -> UnifiedQueryInterface:
    if partition_visibility not in {"public_static", "hidden_instance_dependent"}:
        raise ValueError("unsupported partition_visibility")
    if not 0.0 < failure_probability < 0.5:
        raise ValueError("failure_probability must lie strictly between zero and 1/2")
    public = partition_visibility == "public_static"
    return UnifiedQueryInterface(
        oracle_id="canonical_layer_c_controlled_block_rotation_v1",
        output_relation_id="exact_top_k_strict_angular_gap_set_v1",
        query_unit="one controlled B_theta or B_theta_dagger call",
        failure_probability=failure_probability,
        partition_visibility=partition_visibility,
        stopping_time_visibility="known" if public else "unknown",
        boundary_supplied=False,
        active_history_supplied=False,
        qram_history_supplied=False,
    )


def build_layered_block_family(
    record: VariableTimeChargedHistoryRecord | int,
    *,
    partition_visibility: PartitionVisibility,
    failure_probability: float = 0.05,
) -> UnifiedHardFamily:
    """Instantiate an actual exact-Top-k family from the charged profile."""

    if isinstance(record, VariableTimeChargedHistoryRecord):
        charged = record
    else:
        charged = variable_time_charged_history_record(_integer(record, "record"))
    blocks = tuple(
        LayeredBlock(
            level=layer.level,
            candidate_count=layer.active_count,
            required_output_count=layer.output_births,
            angular_separation=layer.epsilon,
            checker_query_units=layer.qpe_query_units,
        )
        for layer in charged.layers
    )
    block_arm_count = sum(block.candidate_count for block in blocks)
    if block_arm_count > charged.n:
        raise ValueError("charged profile has more block arms than total arms")
    if sum(block.required_output_count for block in blocks) != charged.k:
        raise ValueError("block outputs do not equal k")
    max_separation = max(block.angular_separation for block in blocks)
    center_half_width = min(
        math.pi / 16.0,
        (math.pi / 4.0 - max_separation) / 2.0,
    )
    if center_half_width <= 0.0:
        raise ValueError("angular schedule does not fit in the canonical angle range")
    return UnifiedHardFamily(
        family_schema="layered_block_exact_top_k_with_hidden_global_shift_v1",
        m=charged.m,
        n=charged.n,
        k=charged.k,
        far_arm_count=charged.n - block_arm_count,
        hidden_global_angle_shift=True,
        center_angle_interval=(
            math.pi / 4.0 - center_half_width,
            math.pi / 4.0 + center_half_width,
        ),
        far_arm_angle=0.0,
        blocks=blocks,
        interface=_interface(
            partition_visibility=partition_visibility,
            failure_probability=failure_probability,
        ),
    )


def _direct_product_bucket_floor(family: UnifiedHardFamily) -> float:
    """Return the strongest directly cited fixed-cost search-product floor.

    The strong direct-product theorems apply cleanly to equal-size/equal-cost
    blocks.  We therefore take the largest homogeneous bucket and deliberately
    do not multiply by the finite-QPE cost.  Lifting that cost into the
    canonical rotation model is a separate weighted adversary obligation.
    """

    buckets: dict[tuple[int, int], int] = {}
    for block in family.blocks:
        key = (block.candidate_count, block.checker_query_units)
        buckets[key] = buckets.get(key, 0) + 1
    return max(
        block_count * math.sqrt(candidate_count)
        for (candidate_count, _), block_count in buckets.items()
    )


def instantiate_complexities(
    record: VariableTimeChargedHistoryRecord,
    family: UnifiedHardFamily,
    *,
    match_tolerance: float = 1.2,
) -> ComplexityInstantiation:
    """Put every compared quantity on exactly ``family``'s contract."""

    if record.m != family.m or record.n != family.n or record.k != family.k:
        raise ValueError("record and family dimensions do not match")
    if not math.isfinite(match_tolerance) or match_tolerance < 1.0:
        raise ValueError("match_tolerance must be finite and at least one")
    candidate = record.charged_candidate_total_proxy
    partitioned = sum(
        math.sqrt(block.candidate_count * block.required_output_count)
        * block.checker_query_units
        for block in family.blocks
    )
    total_block_arms = sum(block.candidate_count for block in family.blocks)
    total_outputs = sum(block.required_output_count for block in family.blocks)
    max_checker = max(block.checker_query_units for block in family.blocks)
    global_marked = math.sqrt(total_block_arms * total_outputs) * max_checker
    public = family.interface.partition_visibility == "public_static"
    ratio = partitioned / candidate
    matches = public and ratio <= match_tolerance
    candidate_proved = False
    candidate_status = (
        "proxy_not_a_proved_upper_bound_even_with_public_partition"
        if public
        else "hidden_activity_partition_discovery_and_cleanup_not_constructed"
    )
    closure = PUBLIC_PARTITION_MATCH if matches else HIDDEN_PARTITION_OPEN
    return ComplexityInstantiation(
        family_id=family.family_id,
        interface_id=family.interface.interface_id,
        candidate_proxy=candidate,
        candidate_upper_proved=candidate_proved,
        candidate_upper_status=candidate_status,
        partitioned_selection_composition_proxy=partitioned,
        partitioned_composition_same_interface=public,
        partitioned_over_candidate=ratio,
        partitioned_matches_candidate=matches,
        global_all_marked_proxy=global_marked,
        published_direct_product_floor=_direct_product_bucket_floor(family),
        weighted_direct_product_target=partitioned,
        weighted_direct_product_proved=False,
        strict_composition_advantage_established=False,
        matching_lower_bound_established=False,
        closure_status=closure,
    )


def stopping_time_dichotomy() -> StoppingTimeDichotomy:
    """Encode the literature-constrained stopping-time alternatives."""

    return StoppingTimeDichotomy(
        known_time_rms_covered_by_prior_work=True,
        unknown_time_universal_rms_falsified_by_prior_lower_bound=True,
        published_known_time_scale="Theta(sqrt(sum_i t_i^2))",
        published_unknown_time_upper_scale=(
            "O(sqrt(T log min(n,t_max))) in the 2025 computation-tree preprint"
        ),
        published_unknown_time_lower_scale=(
            "Omega(sqrt(T log T)) on a variable-time-search family"
        ),
        special_structure_escape_status=(
            "open: must state and prove a promise excluding the published "
            "unknown-time lower-bound family"
        ),
        source_urls=(
            PRIMARY_SOURCES["akv_variable_time_search"],
            PRIMARY_SOURCES["vihrovs_computation_trees"],
            PRIMARY_SOURCES["jeffery_et_al_loop_composition"],
        ),
    )


def default_asymptotic_witness() -> DefaultAsymptoticWitness:
    """Prove the public-partition proxy beats the default candidate proxy.

    For the default profile, ``n=m^3``, there are ``m`` blocks of size ``m``,
    and ``epsilon_r=m^-2 r^1/4``.  The finite-QPE construction satisfies
    ``c_r < 4/epsilon_r``.  Hence partitioned selection costs at most
    ``4 sqrt(m) m^2 sum_{r<=m} r^-1/4 = O(m^13/4)``.  The candidate proxy is at
    least its boundary term ``sqrt(m^3) m^2 = m^7/2``.  Their ratio is therefore
    ``O(m^-1/4)``.  This is a statement about the encoded proxies, not about a
    proved candidate algorithm.
    """

    candidate_exponent = 7.0 / 2.0
    partitioned_exponent = 13.0 / 4.0
    ratio_exponent = partitioned_exponent - candidate_exponent
    return DefaultAsymptoticWitness(
        family_assumptions=(
            "n=m^3; R=k=A_r=m; M_r=1; epsilon_r=m^-2(r+1)^1/4; "
            "public static blocks"
        ),
        candidate_proxy_exponent=candidate_exponent,
        partitioned_composition_upper_exponent=partitioned_exponent,
        composition_over_candidate_exponent=ratio_exponent,
        partitioned_is_little_o_of_candidate_proxy=ratio_exponent < 0.0,
        proof_status="proved_local_proxy_asymptotic_lemma",
    )


def _obligations(
    public: ComplexityInstantiation,
    hidden: ComplexityInstantiation,
) -> tuple[TheoremObligation, ...]:
    return (
        TheoremObligation(
            obligation_id="UC-DEF",
            status="proved_local_lemma",
            statement=(
                "Candidate, composition baselines, and lower targets carry one "
                "family_id and one oracle/output interface_id."
            ),
            depends_on=(),
            evidence=(
                f"public={public.family_id}; hidden={hidden.family_id}; interface "
                "fingerprints are generated from canonical serialized contracts."
            ),
            falsifier="Any comparison row has a different family_id or interface_id.",
        ),
        TheoremObligation(
            obligation_id="VT-KNOWN",
            status="known_primary_source",
            statement=(
                "Known stopping times already admit RMS variable-time search "
                "complexity up to the stated source assumptions."
            ),
            depends_on=("UC-DEF",),
            evidence="Ambainis/AKV state the known-times Theta(sqrt(sum t_i^2)) scale.",
            falsifier="The candidate uses a strictly different output primitive, not search.",
            source_urls=(
                PRIMARY_SOURCES["ambainis_variable_time_amplification"],
                PRIMARY_SOURCES["akv_variable_time_search"],
            ),
        ),
        TheoremObligation(
            obligation_id="VT-UNKNOWN",
            status="falsified",
            statement=(
                "A universal unknown-stopping-time O(sqrt(sum t_i^2)) theorem "
                "would hold without an additional structural promise."
            ),
            depends_on=("UC-DEF",),
            evidence=(
                "AKV give an Omega(sqrt(T log T)) lower-bound family; Vihrovs "
                "reports a matching-log upper bound."
            ),
            falsifier="A proved special promise excludes the lower-bound family.",
            source_urls=(
                PRIMARY_SOURCES["akv_variable_time_search"],
                PRIMARY_SOURCES["vihrovs_computation_trees"],
            ),
        ),
        TheoremObligation(
            obligation_id="CF-PUBLIC",
            status="falsified",
            statement="The candidate strictly beats known composition on public blocks.",
            depends_on=("UC-DEF", "VT-KNOWN"),
            evidence=(
                "The information-matched partitioned selection proxy is "
                f"{public.partitioned_over_candidate:.6g} times the candidate "
                "and is within the declared matching frontier; on the default "
                "profile its O(m^13/4) proxy is o(m^7/2)."
            ),
            falsifier=(
                "A formal cost term absent from the candidate but required by the "
                "partitioned baseline is proved under the same interface."
            ),
            source_urls=(
                PRIMARY_SOURCES["wang_et_al_quantum_bai"],
                PRIMARY_SOURCES["gao_ji_wang_approximate_k_minimum"],
                PRIMARY_SOURCES["van_apeldoorn_et_al_all_marked"],
            ),
        ),
        TheoremObligation(
            obligation_id="UB-HIDDEN",
            status="proof_obligation",
            statement=(
                "The hidden activity partition is discovered, used, and uncomputed "
                "within the candidate proxy."
            ),
            depends_on=("UC-DEF",),
            evidence="No compiled hidden-partition discovery transducer exists.",
            falsifier=(
                "Any construction materializes a free activity list, pays a rebuilt "
                "history, or inherits the general unknown-time logarithmic barrier."
            ),
            source_urls=(
                PRIMARY_SOURCES["jeffery_subroutine_composition"],
                PRIMARY_SOURCES["belovs_jeffery_yolcu_transducers"],
                PRIMARY_SOURCES["jeffery_et_al_loop_composition"],
            ),
        ),
        TheoremObligation(
            obligation_id="LB-PRODUCT",
            status="known_primary_source",
            statement=(
                "Homogeneous independent search blocks obey a strong direct-product "
                "barrier in the standard query model."
            ),
            depends_on=("UC-DEF",),
            evidence=(
                "The audit exposes only the largest directly supported homogeneous "
                f"bucket floor, {public.published_direct_product_floor:.6g}."
            ),
            falsifier="The Top-k reduction does not preserve success or query access.",
            source_urls=(
                PRIMARY_SOURCES["klauck_spalek_de_wolf_direct_product"],
                PRIMARY_SOURCES["ambainis_search_direct_product"],
                PRIMARY_SOURCES["lee_roland_direct_product"],
            ),
        ),
        TheoremObligation(
            obligation_id="LB-WEIGHTED",
            status="proof_obligation",
            statement=(
                "The direct-product barrier lifts to heterogeneous angular checker "
                "costs and equals the partitioned weighted target."
            ),
            depends_on=("LB-PRODUCT",),
            evidence="No weighted adversary matrix for the canonical rotation family is supplied.",
            falsifier=(
                "A cross-block quantum algorithm beats the weighted sum, or the "
                "angle-oracle reduction loses the claimed checker factor."
            ),
            source_urls=(PRIMARY_SOURCES["lee_roland_direct_product"],),
        ),
        TheoremObligation(
            obligation_id="CLOSE",
            status="blocked",
            statement="One contract has a proved upper, strict frontier, and matching lower bound.",
            depends_on=("CF-PUBLIC", "UB-HIDDEN", "LB-WEIGHTED"),
            evidence=(
                "Public blocks fail strict composition; hidden blocks have an open "
                "upper; the weighted canonical lower bound is open."
            ),
            falsifier="All dependencies are replaced by reviewed proofs on one family_id.",
        ),
    )


def dependency_graph_is_valid(obligations: tuple[TheoremObligation, ...]) -> bool:
    """Check identifiers, dependency references, and acyclic declaration order."""

    seen: set[str] = set()
    for obligation in obligations:
        if obligation.obligation_id in seen:
            return False
        if any(dependency not in seen for dependency in obligation.depends_on):
            return False
        seen.add(obligation.obligation_id)
    return True


def build_unified_theorem_closure_audit(
    m: int,
    *,
    failure_probability: float = 0.05,
    match_tolerance: float = 1.2,
) -> UnifiedTheoremClosureAudit:
    """Build the public/hidden interface closure audit for one scale."""

    m = _integer(m, "m")
    charged = variable_time_charged_history_record(m)
    public_family = build_layered_block_family(
        charged,
        partition_visibility="public_static",
        failure_probability=failure_probability,
    )
    hidden_family = build_layered_block_family(
        charged,
        partition_visibility="hidden_instance_dependent",
        failure_probability=failure_probability,
    )
    public = instantiate_complexities(
        charged,
        public_family,
        match_tolerance=match_tolerance,
    )
    hidden = instantiate_complexities(
        charged,
        hidden_family,
        match_tolerance=match_tolerance,
    )
    obligations = _obligations(public, hidden)
    return UnifiedTheoremClosureAudit(
        m=m,
        public_family=public_family,
        hidden_family=hidden_family,
        public_instantiation=public,
        hidden_instantiation=hidden,
        stopping_time_dichotomy=stopping_time_dichotomy(),
        default_asymptotic_witness=default_asymptotic_witness(),
        obligations=obligations,
        dependency_graph_valid=dependency_graph_is_valid(obligations),
        theorem_claimable=False,
        ccf_a_quantum_advantage_claimable=False,
    )


def machine_readable_status_map(
    audit: UnifiedTheoremClosureAudit,
) -> dict[str, object]:
    """Return the stable JSON-safe status surface for CI and paper tooling."""

    if not isinstance(audit, UnifiedTheoremClosureAudit):
        raise TypeError("audit must be a UnifiedTheoremClosureAudit")
    return {
        "claim_status": audit.claim_status,
        "closure_status": audit.closure_status,
        "family_ids": {
            "base": audit.public_family.base_family_id,
            "public_contract": audit.public_family.family_id,
            "hidden_contract": audit.hidden_family.family_id,
        },
        "interface_ids": {
            "public": audit.public_family.interface.interface_id,
            "hidden": audit.hidden_family.interface.interface_id,
        },
        "upper": {
            "public_proved": audit.public_instantiation.candidate_upper_proved,
            "public_status": audit.public_instantiation.candidate_upper_status,
            "hidden_proved": audit.hidden_instantiation.candidate_upper_proved,
            "hidden_status": audit.hidden_instantiation.candidate_upper_status,
        },
        "composition": {
            "public_same_interface": (
                audit.public_instantiation.partitioned_composition_same_interface
            ),
            "public_matches_candidate": (
                audit.public_instantiation.partitioned_matches_candidate
            ),
            "public_status": audit.public_instantiation.closure_status,
            "hidden_same_interface": (
                audit.hidden_instantiation.partitioned_composition_same_interface
            ),
            "hidden_status": audit.hidden_instantiation.closure_status,
            "strict_advantage_established": False,
        },
        "lower_bound": {
            "published_homogeneous_floor": (
                audit.public_instantiation.published_direct_product_floor
            ),
            "weighted_target": audit.public_instantiation.weighted_direct_product_target,
            "weighted_target_proved": (
                audit.public_instantiation.weighted_direct_product_proved
            ),
            "matching_lower_bound_established": False,
        },
        "asymptotic_public_witness": asdict(audit.default_asymptotic_witness),
        "obligation_statuses": {
            item.obligation_id: item.status for item in audit.obligations
        },
        "dependency_graph_valid": audit.dependency_graph_valid,
        "theorem_claimable": audit.theorem_claimable,
        "ccf_a_quantum_advantage_claimable": (
            audit.ccf_a_quantum_advantage_claimable
        ),
    }


def local_component_max_factor(components: tuple[float, ...]) -> float:
    """Return ``sum(components) / max(components)`` for a local closure check.

    This proves only the elementary fact that a *common-family* lower bound for
    every one of a constant number of components matches their sum up to that
    constant.  It cannot combine lower bounds established on different input
    families.
    """

    if not components:
        raise ValueError("components cannot be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in components):
        raise ValueError("components must be positive and finite")
    return sum(components) / max(components)
