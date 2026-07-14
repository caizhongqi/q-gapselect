# Rejected Q-GapSelect Candidates

Version: 2026-07-14

This is a permanent audit log. A rejected claim is not restored by renaming the same construction. Reopening an entry requires a new proof that directly resolves the recorded counterexample or model mismatch, a linked claim-matrix change, and independent review.

## RC-001: fixed smaller-cardinality orientation

Status: REJECTED

Former claim: enumerate the selected set when k <= n-k and enumerate the rejected set otherwise; this always gives the instance-optimal output-sensitive orientation.

Counterexample: take n=3m and k=m. Put all m selected arms at a fine angular boundary gap gamma. Put one rejected arm at gap gamma and the remaining 2m-1 rejected arms at a constant angular gap. At the fine layer the selected representation has m hard outputs while the rejected representation has one. A layer term proportional to sqrt(N_r(M_r+1))/epsilon_r therefore differs by a factor Theta(sqrt(m)). Choosing gamma sufficiently small makes the fine layer dominate all coarse work. The selected set has smaller cardinality but is asymptotically more expensive to certify.

Why it failed: output cardinality alone does not capture how hard outputs are distributed across gap layers.

Replacement candidate: define selected and rejected costs H_+ and H_- from their layer multiplicities and target H_orient = min(H_+,H_-). Run resumable procedures for both orientations and accept the first independently verified certificate.

Open burden: prove the certificate verifier and a first-finish dovetail bound. H_orient remains a conjectural functional.

Prohibited wording: smaller side, optimal complement orientation, or min(k,n-k) adaptation as an instance-optimal theorem.

## RC-002: raw mean gaps as the universal coherent difficulty

Status: REJECTED

Former claim: the instance-tight local quantum cost is c_i = Theta(1/Delta_i), where Delta_i is the raw mean distance to the Top-k boundary.

Counterexample: canonical reward states are parameterized by theta = arcsin sqrt(mu) and have overlap cos(theta-theta-prime). Near zero, means mu=0 and mu'=epsilon have mean difference epsilon but angular separation approximately sqrt(epsilon). Their coherent discrimination cost is Theta(1/sqrt(epsilon)), not Theta(1/epsilon). A symmetric endpoint issue occurs near one.

Why it failed: the map from mean to amplitude angle has an unbounded derivative at the endpoints.

Replacement: use the angular boundary gap gamma_i and local cost Theta(1/gamma_i). Keep Delta_i for reporting. Translate only under an explicit interior promise; for compared means in [eta,1-eta],

    2 sqrt(eta(1-eta)) gamma_i <= Delta_i <= gamma_i.

Prohibited wording: instance-tight inverse mean gap on all of [0,1], or silent replacement of gamma by Delta.

## RC-003: free purification-to-clean-rotation compression

Status: REJECTED

Former claim: a reversible stochastic generator can be treated as the clean block rotation R_y(2 arcsin sqrt(mu_i)) at one oracle call per use.

Model mismatch: a natural sampler prepares

    |i> sum_w sqrt(P_i(w)) |w>|R_i(w)>,

with arm-, seed-, output-, and reward-dependent work states. A clean block contains only the reward amplitude and specifies an orthogonal second column. Erasing w while preserving the unknown success amplitude is not a free relabeling; it is a coherent transduction or estimation task. The cost and error can affect every boundary, flagging, extraction, and verification step.

Replacement: maintain two explicit layers. Layer C is a fully defined canonical R_y block used for analytic work and hard subclasses. Layer P supplies A, A dagger, and a known good projector. Re-prove the upper directly under Layer-P access and charge generator/checker work, controls, precision, qubits, and uncomputation.

Prohibited wording: equivalent oracle, without-loss clean reward qubit, or one-call compression unless accompanied by a proved charged conversion.

## RC-004: primitive-level novelty for QBoundary, QBatchExtract, and equal gaps

Status: REJECTED

Former claim: noisy boundary localization, coherent batch extraction, or equal-gap stochastic Top-k is by itself a new quantum algorithmic direction.

Prior-art reason: approximate k-minimum with approximate values and expectation-value applications is treated by Gao, Ji, and Wang, ESA 2025. Finding all marked elements in O(sqrt(Nk)) queries is treated by van Apeldoorn, Gribling, and Nieuwboer, Quantum 2024. Variable-time search, order statistics, minimum/k-minimum procedures, and quantum best-arm identification already supply the surrounding components and sanity regimes.

Replacement novelty boundary: only a strict heterogeneous-angular-gap, fixed-confidence multi-output theorem that is asymptotically stronger than every valid black-box composition, with a matching all-adaptive-algorithms lower bound, remains a plausible new quantum contribution.

Prohibited wording: our novel QBoundary, our new batch extractor, or equal-gap novelty. Project-specific labels may describe architecture but not ownership of the underlying primitive.

## RC-005: arbitrary-purification instance-wise lower bound

Status: REJECTED

Former claim: a lower bound proved for the canonical B_mu oracle automatically applies to every purification sampler with the same vector of success probabilities.

Counterexample principle: work states may encode extra information about the arm, the ordering, or the hidden hypothesis while leaving each reward probability unchanged. Two samplers with the same means can therefore have different query complexity. A lower bound for the clean canonical unitary cannot forbid such side information.

Replacement scope: canonical instances are members of the compatible purification class. A canonical lower bound therefore supplies a worst-case lower bound over that class, not an instance-wise bound for all members. A stronger theorem needs an explicit no-leakage promise or a complexity parameter based on the full output states.

Prohibited wording: for any purification with these means, oracle-implementation independent lower bound, or automatic Layer-P lower bound.

## RC-006: simulator runtime as quantum speedup

Status: REJECTED

Former claim: lower wall-clock time in a classical quantum-circuit simulator demonstrates a quantum speedup.

Why it failed: simulator time depends on implementation, hardware, compiler, sparsity, memory layout, and finite test sizes. It is neither a physical quantum runtime nor a query-complexity lower-bound comparison.

Replacement: report query ledgers, gates, depth, qubits, state preparation, and classical simulator resources as separate measurements. Establish algorithmic advantage only through same-model upper/lower theorems and appropriately scoped physical resource estimates.

Prohibited wording: simulated quantum speedup or runtime proof of quantum advantage.

## RC-007: current analytic scenarios as completed experiments

Status: REJECTED-AS-CURRENT

Former claim: the repository already evaluates angle-space clustered, planted-dyadic, endpoint, and complement-asymmetry random-instance families with at least 500 completed instances per point.

Artifact audit: two analytic pipelines exist and must not be conflated. The run_scaling pipeline evaluates five deterministic mean-space equal/geometric analytic-proxy regimes and does not use its trials field for repetition. The run_reference pipeline runs four fixed mean-vector instances over configured derived seeds; it does use trials as a loop count and randomly samples the analytic Grover measurement laws requested by the estimator. The latter is stochastic analytic-measurement regression, but it is neither coherent batch execution nor random problem-instance generation. Its versioned diagnostic artifact has four trials per fixed scenario, 16 records total. The configuration schedules 500 trials per fixed scenario, but no artifact for that default run is committed. The reference schema reports certified_exact_recovery only for interval-resolved exact outputs and reports heuristic_inclusive_exact_recovery separately for equality after any empirical timeout completion. The second field is not fixed-confidence success.

Replacement: report both current pipelines with their exact semantics. Describe clustered, dyadic, endpoint, complement-asymmetry counterexamples, angle-space random-instance generation, and at least 500 independently generated instances per future configuration as a pre-registered campaign. Do not confuse that future instance count with the implemented 500-seed default over four fixed references. A future implementation and immutable manifests may support new empirical claim rows but do not alter this audit retroactively.

Prohibited wording now: the repository has only five scenarios; the current repository has no stochastic observations; trials are universally unused; the fixed complement_orientation scenario is the asymmetry counterexample; the configured 500-trial default has already produced a committed result artifact; or heuristic-inclusive recovery is fixed-confidence success.

## RC-008: timeout completion counted as fixed-confidence success

Status: REJECTED

Former claim: an exact selected set can count as fixed-confidence recovery even when confidence intervals did not resolve and the run reached its round limit before an empirical completion rule filled the output.

Why it failed: correctness of the completed set on a finite diagnostic record does not provide the algorithm's required certificate. Counting such timeouts in the success numerator hides failure to meet the stopping and confidence contract.

Replacement: certified_exact_recovery requires both interval_resolved and equality with the true Top-k set. heuristic_inclusive_exact_recovery is retained only as a separate diagnostic of the empirical completion rule. Aggregates report both and preserve timeout counts.

Prohibited wording: exact recovery rate without naming which field; fixed-confidence success based on heuristic-inclusive recovery; or removal of correct-looking timeouts from the failure denominator.

## Reopening checklist

For any entry proposed for reopening:

1. state the materially new claim rather than rename the rejected one;
2. reproduce the archived counterexample or model mismatch;
3. show exactly which new assumption, algorithm, or proof defeats it;
4. update the oracle and resource model consistently;
5. link a proof or immutable artifact;
6. obtain independent adversarial review;
7. preserve this historical entry and append the resolution instead of deleting it.
