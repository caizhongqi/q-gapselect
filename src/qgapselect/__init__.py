"""Public research API for Q-GapSelect.

The package exposes analytic and exact-state reference backends alongside
separately labelled candidate-complexity accounting.  Exact-state execution is
still a classical simulation, and importing a ``candidate_*`` symbol must not
be interpreted as importing a proved quantum bound.
"""

from .attack_metrics import AttackMetrics, aggregate_attack_metrics
from .coherent import CanonicalRyStatevectorOracle
from .contracts import CoherentOracleContract, CoherentRewardGate, OracleModel
from .direct_baselines import ClassicalThresholdScan, IndependentQPEThresholdScan
from .direct_phase import (
    DirectAmplitudeThresholdFlag,
    DirectPhaseFlagResources,
    DirectPhaseFlagResult,
    DirectPhaseThresholdFlag,
    IndexVerificationResult,
)
from .direct_search import (
    DirectSearchAttempt,
    DirectSearchResources,
    DirectThresholdSearchResult,
    FullWorkspaceBBHT,
    full_workspace_rank_one_diffusion,
)
from .direct_topk import (
    CalibratedDirectTopKController,
    CalibratedDirectTopKResult,
    DirectTopKBranchTrace,
    DirectTopKResources,
)
from .estimators import AnalyticIterativeAmplitudeEstimator, IterativeAmplitudeEstimator
from .gapselect import QGapSelect
from .iterative_ae_baseline import IterativeAEThresholdScan
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
from .quantum_benchmarking import (
    QuantumBenchmarkConfig,
    QuantumBenchmarkRunner,
    aggregate_benchmark_records,
    make_benchmark_instance,
    paired_query_ratios,
)
from .quantum_diagnostics import (
    run_diffusion_ablation,
    run_phase_grid_sweep,
    run_qpe_acceptance_sweep,
)
from .quantum_validation import run_unitary_validation, run_verifier_calibration

__all__ = [
    "AnalyticIterativeAmplitudeEstimator",
    "AttackMetrics",
    "AttackStudyPlan",
    "CallableGenerationValidator",
    "CallableLocalModelAdapter",
    "CanonicalBernoulliOracleSimulator",
    "CanonicalRyStatevectorOracle",
    "CalibratedDirectTopKController",
    "CalibratedDirectTopKResult",
    "ClassicalThresholdScan",
    "CoherentOracleContract",
    "CoherentRewardGate",
    "CounterfactualRateSelector",
    "DirectAmplitudeThresholdFlag",
    "DirectPhaseFlagResources",
    "DirectPhaseFlagResult",
    "DirectPhaseThresholdFlag",
    "DirectSearchAttempt",
    "DirectSearchResources",
    "DirectThresholdSearchResult",
    "DirectTopKBranchTrace",
    "DirectTopKResources",
    "DovetailTopKController",
    "EvaluatedGeneration",
    "GapSelectConfig",
    "GapSelectResult",
    "IAEConfig",
    "IterativeAmplitudeEstimator",
    "GenerationRecord",
    "GenerationRequest",
    "FullWorkspaceBBHT",
    "IndexVerificationResult",
    "IndependentQPEThresholdScan",
    "IterativeAEThresholdScan",
    "NaturalArmDistribution",
    "NaturalPurificationStatevectorOracle",
    "OfflineReplayBackend",
    "OracleModel",
    "QGapSelect",
    "QGapSelectPortfolioAdapter",
    "QuantumBenchmarkConfig",
    "QuantumBenchmarkRunner",
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
    "aggregate_benchmark_records",
    "collect_local_records",
    "full_workspace_rank_one_diffusion",
    "make_benchmark_instance",
    "paired_counterfactual_event",
    "qbatch_extract",
    "qboundary",
    "qgap_flag",
    "paired_query_ratios",
    "run_diffusion_ablation",
    "run_phase_grid_sweep",
    "run_qpe_acceptance_sweep",
    "run_unitary_validation",
    "run_verifier_calibration",
    "run_attack_study",
]

__version__ = "0.1.0"
