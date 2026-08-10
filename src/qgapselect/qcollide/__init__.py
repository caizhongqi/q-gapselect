"""Q-COLLIDE calibration, geometry, packing, and analytic cost models."""

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
from .scaling import compact_summary, fit_log_linear, run_scaling_campaign

__all__ = [
    "CollisionCriteria",
    "CollisionInstance",
    "DualHeadModel",
    "EndpointRecord",
    "PackingStatistics",
    "TunnelCertificate",
    "TunnelGeometry",
    "build_adjacency",
    "classical_packed_cost",
    "compact_dual_head_summary",
    "compact_summary",
    "evaluate_instance",
    "fit_log_linear",
    "geometry_to_packing",
    "packed_claw",
    "packing_statistics",
    "pair_grover_cost",
    "pair_oracle_negative",
    "prefix_rms_cost",
    "prefix_survival_rates",
    "product_johnson_cost",
    "random_range",
    "run_dual_head_causal_campaign",
    "run_scaling_campaign",
    "targeted_control_closure",
    "train_dual_head_model",
    "tunnel_certificate",
    "tunnel_geometry",
    "weighted_prefix_claw",
    "weighted_qcollide_cost",
]
