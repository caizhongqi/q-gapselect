"""Public research API for Q-GapSelect.

The package exposes analytic and exact-state reference backends alongside
separately labelled candidate-complexity accounting.  Exact-state execution is
still a classical simulation, and importing a ``candidate_*`` symbol must not
be interpreted as importing a proved quantum bound.
"""

from .attack_metrics import AttackMetrics, aggregate_attack_metrics
from .coherent import CanonicalRyStatevectorOracle
from .contracts import CoherentOracleContract, CoherentRewardGate, OracleModel
from .estimators import AnalyticIterativeAmplitudeEstimator, IterativeAmplitudeEstimator
from .gapselect import QGapSelect
from .llm_attack import (
    AttackStudyPlan,
    CallableGenerationValidator,
    CallableLocalModelAdapter,
    CounterfactualRateSelector,
    EvaluatedGeneration,
    GenerationRecord,
    GenerationRequest,
    OfflineReplayBackend,
    QGapSelectPortfolioAdapter,
    QueryBudget,
    Seed,
    SemanticVariant,
    Task,
    ValidatorResult,
    collect_local_records,
    paired_counterfactual_event,
    run_attack_study,
)
from .models import GapSelectConfig, GapSelectResult, IAEConfig, TopKInstance
from .natural_oracles import (
    NaturalArmDistribution,
    NaturalPurificationStatevectorOracle,
)
from .oracles import CanonicalBernoulliOracleSimulator, QueryKind, QueryLedger
from .primitives import (
    DovetailTopKController,
    QBoundaryEstimator,
    QGapFlag,
    qbatch_extract,
    qboundary,
    qgap_flag,
)

__all__ = [
    "AnalyticIterativeAmplitudeEstimator",
    "AttackMetrics",
    "AttackStudyPlan",
    "CallableGenerationValidator",
    "CallableLocalModelAdapter",
    "CanonicalBernoulliOracleSimulator",
    "CanonicalRyStatevectorOracle",
    "CoherentOracleContract",
    "CoherentRewardGate",
    "CounterfactualRateSelector",
    "DovetailTopKController",
    "EvaluatedGeneration",
    "GapSelectConfig",
    "GapSelectResult",
    "IAEConfig",
    "IterativeAmplitudeEstimator",
    "GenerationRecord",
    "GenerationRequest",
    "NaturalArmDistribution",
    "NaturalPurificationStatevectorOracle",
    "OfflineReplayBackend",
    "OracleModel",
    "QGapSelect",
    "QGapSelectPortfolioAdapter",
    "QBoundaryEstimator",
    "QGapFlag",
    "QueryKind",
    "QueryBudget",
    "QueryLedger",
    "Seed",
    "SemanticVariant",
    "Task",
    "TopKInstance",
    "ValidatorResult",
    "aggregate_attack_metrics",
    "collect_local_records",
    "paired_counterfactual_event",
    "qbatch_extract",
    "qboundary",
    "qgap_flag",
    "run_attack_study",
]

__version__ = "0.1.0"
