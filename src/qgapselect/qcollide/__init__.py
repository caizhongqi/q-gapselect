"""Q-COLLIDE calibration, geometry, packing, and analytic cost models."""

from .capacity_meta import CapacityCell, build_capacity_meta_summary, extract_capacity_cells
from .closure_spectrum import (
    compact_closure_spectrum_summary,
    dangerous_direction_spectrum,
    residual_dangerous_energy,
    run_closure_spectrum_campaign,
)
from .contracts import CollisionCriteria, CollisionInstance, EndpointRecord
from .costs import (
    classical_packed_cost,
    pair_grover_cost,
    prefix_rms_cost,
    prefix_survival_rates,
    product_johnson_cost,
    weighted_qcollide_cost,
)
from .dual_head import (
    DualHeadModel,
    compact_dual_head_summary,
    run_dual_head_causal_campaign,
    train_dual_head_model,
)
from .evaluation import evaluate_instance
from .fixtures import (
    geometry_to_packing,
    packed_claw,
    pair_oracle_negative,
    random_range,
    weighted_prefix_claw,
)
from .geometry import (
    TunnelCertificate,
    TunnelGeometry,
    targeted_control_closure,
    tunnel_certificate,
    tunnel_geometry,
)
from .graph import PackingStatistics, build_adjacency, packing_statistics
from .prefix_lower_bound import (
    PrefixRestrictionProfile,
    RandomRangeClawBound,
    RandomRangePrefixProfile,
    RandomRangeStageBound,
    StageRestrictionBound,
    homogeneous_packed_claw_lower_bound,
    prefix_stage_restriction_profile,
    random_range_claw_distributional_bound,
    random_range_prefix_restriction_profile,
)
from .scaling import compact_summary, fit_log_linear, run_scaling_campaign
from .stage_selector_lower_bound import (
    CollisionWeightedCostNorm,
    StageSelectorComplexityScale,
    StageSelectorComponent,
    StageSelectorProfile,
    collision_weighted_cost_norm,
    conditional_stage_selector_profile,
    disjoint_single_claw_stage_selector_complexity_scale,
    disjoint_single_claw_stage_selector_profile,
)
from .topology import (
    CollisionFiltrationPoint,
    CollisionTopologyMetrics,
    collision_filtration,
    collision_topology_metrics,
)
from .topology_dynamics import (
    TopologyChangePoint,
    TopologyMetricName,
    TopologySnapshot,
    analyze_topology_trajectory,
    detect_topology_change_point,
    topology_snapshot,
)
from .topology_evidence import (
    TopologyEvidenceCell,
    build_topology_evidence_audit,
    extract_topology_evidence_cells,
)

__all__ = [
    "CapacityCell",
    "CollisionCriteria",
    "CollisionFiltrationPoint",
    "CollisionInstance",
    "CollisionTopologyMetrics",
    "CollisionWeightedCostNorm",
    "DualHeadModel",
    "EndpointRecord",
    "PackingStatistics",
    "PrefixRestrictionProfile",
    "RandomRangeClawBound",
    "RandomRangePrefixProfile",
    "RandomRangeStageBound",
    "StageRestrictionBound",
    "StageSelectorComponent",
    "StageSelectorComplexityScale",
    "StageSelectorProfile",
    "TopologyChangePoint",
    "TopologyEvidenceCell",
    "TopologyMetricName",
    "TopologySnapshot",
    "TunnelCertificate",
    "TunnelGeometry",
    "analyze_topology_trajectory",
    "build_adjacency",
    "build_capacity_meta_summary",
    "build_topology_evidence_audit",
    "classical_packed_cost",
    "collision_filtration",
    "collision_topology_metrics",
    "collision_weighted_cost_norm",
    "compact_closure_spectrum_summary",
    "compact_dual_head_summary",
    "compact_summary",
    "conditional_stage_selector_profile",
    "dangerous_direction_spectrum",
    "detect_topology_change_point",
    "disjoint_single_claw_stage_selector_complexity_scale",
    "disjoint_single_claw_stage_selector_profile",
    "evaluate_instance",
    "extract_capacity_cells",
    "extract_topology_evidence_cells",
    "fit_log_linear",
    "geometry_to_packing",
    "homogeneous_packed_claw_lower_bound",
    "packed_claw",
    "packing_statistics",
    "pair_grover_cost",
    "pair_oracle_negative",
    "prefix_rms_cost",
    "prefix_stage_restriction_profile",
    "prefix_survival_rates",
    "product_johnson_cost",
    "random_range",
    "random_range_claw_distributional_bound",
    "random_range_prefix_restriction_profile",
    "residual_dangerous_energy",
    "run_closure_spectrum_campaign",
    "run_dual_head_causal_campaign",
    "run_scaling_campaign",
    "targeted_control_closure",
    "topology_snapshot",
    "train_dual_head_model",
    "tunnel_certificate",
    "tunnel_geometry",
    "weighted_prefix_claw",
    "weighted_qcollide_cost",
]
