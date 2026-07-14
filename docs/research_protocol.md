# Q-GapSelect Research Protocol

Version: audited protocol, 2026-07-14

## 1. Purpose and non-claims

Q-GapSelect is a research program for exact, fixed-confidence Top-k identification with unknown heterogeneous reward amplitudes. The potential contribution is a constructive quantum algorithm and matching lower bound whose complexity adapts to angular boundary gaps and to the cheaper of two verifiable output orientations.

Nothing in the current repository proves that contribution. The current artifact is an analytic sanity harness for a clean canonical oracle. It is not a coherent batch algorithm, a natural-sampler implementation, a proof of asymptotic advantage, or evidence of an LLM quantum advantage.

The protocol uses the following status vocabulary:

- SPECIFICATION: a definition or interface fixed for the project.
- KNOWN: a result already established in cited literature.
- CODE-SANITY: a finite deterministic or simulated check with no theorem status.
- CONJECTURE: a mathematical statement that remains unproved.
- PROOF-OBLIGATION: a required lemma, reduction, or resource bound.
- EMPIRICAL-PENDING: a pre-registered experiment that has not been run.
- REJECTED: a claim that must not be revived without resolving its counterexample.

## 2. Problem definition

There are n arms with unknown means μ_i in [0,1]. Let μ_(1) ≥ ... ≥ μ_(n), assume μ_(k) > μ_(k+1), and let S* be the indices of the k largest means. An algorithm must output S* with probability at least 1-δ.

Define amplitude angles

    θ_i = arcsin sqrt(μ_i),     θ_i in [0, π/2].

The strict mean boundary is equivalent to θ_(k) > θ_(k+1). The primary gaps are angular:

    γ_i = θ_i - θ_(k+1)       for i in S*,
    γ_i = θ_(k) - θ_i         for i not in S*.

The local coherent-discrimination cost is c_i = Θ(1/γ_i). Raw mean gaps Δ_i are retained only for reporting and for comparisons with classical literature. If all compared means lie in [η,1-η], then

    2 sqrt(η(1-η)) γ_i ≤ Δ_i ≤ γ_i.

No statement may replace γ_i by Δ_i near probability endpoints without an explicit additional argument.

## 3. Oracle models and accounting

### 3.1 Layer C: canonical block rotation

Layer C is the fully specified unitary

    B_μ = sum_i |i><i| tensor R_y(2θ_i),

where

    R_y(2θ) = [[cos θ, -sin θ], [sin θ, cos θ]].

Thus

    B_μ |i>|0> = |i>(sqrt(1-μ_i)|0> + sqrt(μ_i)|1>),
    B_μ |i>|1> = |i>(-sqrt(μ_i)|0> + sqrt(1-μ_i)|1>).

Layer C supplies B_μ and B_μ dagger. Controlled calls are available only when a theorem or experiment declares and counts them. This model is appropriate for analytic calculations, canonical hard instances, and lower-bound subclasses. It is not silently identified with a reversible LLM generator.

### 3.2 Layer P: natural purification sampler

Layer P supplies A, A dagger, and a known good-projector reflection. In general,

    A |i>|0_W>|0_R>
      = |i>(sqrt(1-μ_i)|φ_i,0>|0> + sqrt(μ_i)|φ_i,1>|1>).

A reversible classical sampler or generator naturally has the form

    A |i>|0_W>|0_R>
      = |i> sum_w sqrt(P_i(w)) |w>|R_i(w)>.

The good projector acts on the final reward bit. Work states may depend on the arm and reward. Controlled A is not free. State preparation, reversible generator/checker gates, precision, work qubits, uncomputation, and A/A-dagger/good-reflection calls are reported separately.

There is no free conversion from Layer P to a garbage-free R_y block. Such a conversion would erase generator information and synthesize an unknown success amplitude. Any upper bound intended for Layer P must be proved directly with A, A dagger, and the good projector, or charge a proved conversion.

### 3.3 Lower-bound scope

Canonical block instances are compatible Layer-P samplers, so a Layer-C lower bound implies a worst-case lower bound over the compatible-sampler class. It does not imply an instance-wise lower bound for every purification having the same μ vector: work registers could leak extra information.

Permitted wording is canonical lower bound and hence compatible-sampler worst-case lower bound. Forbidden wording is lower bound for every implementation of the same means unless a no-leakage promise or a full-state-fidelity lower bound has been proved.

## 4. Adaptive certificate orientation

The answer has two exact representations:

    S*_+ = S*,
    S*_- = [n] minus S*.

The former fixed smaller-cardinality rule is REJECTED. Gap heterogeneity can make the smaller set contain many more hard outputs.

Let ε_r = (π/2)2^(-r), and let

    ell_i = min {r ≥ 0 : ε_r ≤ γ_i},
    N_r = number of i with ell_i ≥ r,
    M_(r,b) = number of i in S*_b with ell_i = r.

Define the candidate orientation costs

    H_b = sum_r sqrt(N_r (M_(r,b)+1)) / ε_r,
    H_orient = min(H_+, H_-).

These are candidate accounting quantities, not proved query complexities. The plus one explicitly charges certification even when no output is first resolved at a layer.

The proposed controller runs two resumable procedures with geometrically increasing budgets. One constructs and verifies a size-k selected certificate. The other constructs and verifies a size-(n-k) rejected certificate and returns its complement. Failure budgets are summable across rounds and orientations. The first independently verified certificate wins.

Required theorem: this dovetail must be correct and cost at most a constant or polylogarithmic factor more than the faster valid orientation, despite variable stopping times, measurements, restarts, and bounded-error quantum subroutines.

## 5. Main candidate theorem

The canonical upper-bound candidate is

    queries = soft-O(H_orient polylog(n, 1/δ, 1/γ_min)).

This is a CONJECTURE. A proof must provide an implementable algorithm, not just add the costs of named primitives. It must include:

1. approximate boundary acquisition at each angular scale;
2. a reversible three-way classifier: certified in, certified out, unresolved;
3. variable-time composition of arm-dependent classifier costs;
4. multiple-marked extraction of newly certified outputs;
5. reversible activity filtering with history access charged;
6. independent cross-boundary certificate verification;
7. orientation dovetail and full confidence accounting;
8. uncomputation, QROM, arithmetic, precision, gates, depth, and qubits.

The Layer-P transport candidate asks for the same reward-query expression using A, A dagger, and the good projector, plus explicit generator/checker resource costs. It is a separate CONJECTURE and may fail even if the Layer-C theorem is true.

## 6. Literature and black-box baseline audit

The following components are prior work and cannot be claimed as new:

- amplitude and quantum mean estimation;
- approximate k-minimum finding with approximate values;
- Durr-Hoyer-style minimum and k-minimum selection;
- finding all marked elements in O(sqrt(Nk)) queries;
- variable-time quantum search;
- order-statistic and median query bounds;
- single-output fixed-confidence quantum best-arm identification;
- equal-gap Top-k recovery;
- a noisy boundary primitive called QBoundary;
- batch extraction called QBatchExtract.

The mandatory comparisons include Gao-Ji-Wang approximate k-minimum, van Apeldoorn-Gribling-Nieuwboer marked extraction, Ambainis-Kokainis-Vihrovs variable-time search, Wang et al. quantum best-arm identification, Nayak-Wu order statistics, Wang-Xu-Zhang time-efficient k-minimum, and the strongest same-interface classical Top-k or Best-Set allocation.

Novelty survives only if the proved heterogeneous fixed-confidence bound is asymptotically smaller than the strongest valid black-box composition on an explicit infinite family. A constant-factor improvement, simulator speedup, favorable equal-gap plot, or relabeling of known primitives does not satisfy this gate.

## 7. Work packages and proof gates

### WP0: model and prior-work audit

Deliverables:

- freeze Layer C and Layer P interfaces;
- derive all black-box upper bounds in the same query model;
- record input-model mismatches rather than comparing incomparable costs;
- maintain the rejected-candidate archive.

Exit gate: every novelty sentence maps to a claim-matrix row and every baseline has a checked citation.

### WP1: deterministic mathematical and code sanity

Deliverables:

- verify canonical block unitarity and both basis-column actions;
- verify monotonicity and angular/mean-gap conversion under interior promises;
- implement angular instance generation and both orientations;
- reproduce the complement-asymmetry counterexample;
- keep executed query ledgers separate from candidate-theory counters.

Current status: incomplete. Two analytic pipelines are implemented. The run_scaling pipeline contains five deterministic mean-space equal/geometric analytic-proxy regimes; its trials field is not used for repetition. The run_reference pipeline runs four fixed mean-vector instances over configured derived seeds, uses trials as its loop count, and randomly samples the analytic Grover measurement laws requested by the independent all-active estimator. It is neither a coherent index-register batch circuit nor random instance generation. Passing either pipeline activates no theorem.

### WP2: canonical upper bound

Deliverables:

- formal staged algorithm;
- noisy boundary and predicate-composition lemmas;
- batch enumeration lemma with output verification;
- activity-filter implementation and resource accounting;
- orientation-dovetail theorem;
- complete confidence proof;
- canonical H_orient upper bound.

Go criterion: a complete constructive proof and an explicit family separating it from every audited black-box upper bound.

No-go criterion: a hidden worst-gap factor, uncharged filter/history construction, invalid bounded-error oracle reuse, or no asymptotic black-box separation.

### WP3: purification transport

Deliverables:

- direct Layer-P versions of boundary tests and angular classifiers;
- handling of arm-dependent garbage and good-projector reflections;
- error and resource composition for A and A dagger;
- proof that no step assumes a free clean rotation;
- reversible generator/checker estimates for the intended application.

Go criterion: the claimed reward-query scaling survives with all ancillary costs stated.

No-go criterion: the proof needs clean amplitude blocks that cannot be implemented at the charged cost.

### WP4: lower bound

Deliverables:

- canonical angular hard instances;
- equal-gap and fixed-weight sanity reductions, clearly labeled non-new;
- disjoint dyadic heterogeneous blocks;
- a valid adversary composition or strong direct-product argument;
- an orientation-aware relation lower bound;
- matching upper/lower comparison up to declared logarithms;
- explicit compatible-sampler worst-case scope.

Go criterion: an all-adaptive-quantum-algorithms lower bound matches the final candidate functional on the relevant family.

No-go criterion: adding independent block lower bounds, restricting the adversary to estimate-then-sort, or claiming the result for arbitrary purifications.

### WP5: deferred matroid extension

Only after WP2 and WP4 survive, study maximum-mean matroid basis identification. Use angular exchange gaps. For a basis element e, define γ_e = +infinity when it has no outside exchange competitor. Charge reward and reversible independence-oracle queries, primal/dual certification, state preparation, time, and memory.

Exit gate: recover the audited uniform-matroid theorem as a special case and prove direct-sum behavior for independent partition groups. A global square root over independently required outputs is a rejection condition.

### WP6: code-LLM attack application

This package starts only after the algorithmic gates make its quantum interpretation meaningful. The task is attack-portfolio selection over complete semantics-preserving variants, not additive composition of atomic edits.

For variant e and task/seed randomness w, define reward one only when generated code is functionally correct and exploit-confirmed vulnerable. A reversible source model prepares a natural Layer-P sampler containing w, generated code, test outcomes, vulnerability outcomes, and the reward bit. Proprietary victims receive only measured classical prompts and are never modeled as coherent oracles.

No genuine LLM quantum advantage may be claimed before WP3 and a credible reversible resource analysis are complete.

## 8. Experimental protocol

### 8.1 Current artifact disclosure

The repository contains two distinct current pipelines:

- run_scaling evaluates five deterministic mean-space equal/geometric analytic-proxy regimes. Its trials configuration field is retained in provenance but is not used to repeat the deterministic calculations.
- run_reference executes four fixed mean-vector instances over a configurable number of derived seeds. It uses trials in the scenario/seed loop and samples the analytic measurement law of every requested Grover experiment. It emits per-trial executed-query records separately from conjectural layer charges.

The reference pipeline is stochastic at the analytic measurement-sampling level. It is not a coherent index-register batch circuit, does not generate random problem instances, and does not implement QBoundary, coherent batch extraction, orientation dovetail, Layer P, or an LLM attack. The versioned reference_diagnostic artifact contains four trials for each of four fixed scenarios, for 16 raw records. The frozen configuration schedules 500 trials per fixed scenario, but no artifact from that default 500-trial run is committed.

Reference success fields have strict, non-interchangeable semantics:

- certified_exact_recovery is true only when the confidence intervals resolve the Top-k set and the returned set equals truth;
- heuristic_inclusive_exact_recovery checks output equality even when a timeout is followed by empirical completion.

Raw records and aggregates report both fields separately, along with interval_resolved and timeout. Only certified_exact_recovery may count toward empirical fixed-confidence success. Heuristic-inclusive recovery is a diagnostic of the completion rule, and a timeout remains unresolved even if its completed output happens to be correct.

Current outputs may be labeled only CODE-SANITY, analytic proxy, or analytic-measurement regression as applicable. Logical calls recorded by the simulator are not physical quantum execution and must not be described as a measured quantum speedup.

### 8.2 Pre-registered future synthetic campaign

The following work is EMPIRICAL-PENDING and has not been executed:

- angle-space equal-gap instances;
- angle-space geometric-gap instances;
- clustered boundary instances;
- planted dyadic heterogeneous blocks;
- endpoint regimes near zero and one;
- complement-asymmetry counterexamples;
- random-instance generation and evaluation beyond the four fixed references;
- Layer-P purification instances with controlled garbage leakage tests.

Planned grid:

| parameter | future values |
|---|---|
| n | 8, 16, 32, 64, 128, 256 |
| k | 1, 2, 4, floor(n/4), floor(n/2) |
| delta | 0.05, 0.01 |
| random-instance repetitions | at least 500 independently generated instances per future configuration |

The future at-least-500 independently generated instances are not the same object as the 500-seed default in configs/reference.json. The latter repeats analytic measurement sampling on each of four fixed instances and is already implemented, although its default run artifact is not committed. The future campaign will store both γ_i and Δ_i and analyze endpoint cases in angular coordinates.

### 8.3 Required baselines

Quantum-side comparisons:

- uniform amplitude estimation followed by sorting;
- adaptive independent per-arm estimation;
- approximate k-minimum with the expectation-value oracle cost charged;
- variable-time search over arm-specific classifiers;
- multiple-marked extraction at each scale;
- quantum BAI only in its valid single-output regime;
- time-efficient k-minimum where its input model applies;
- the strongest derived black-box combination of these components.

Classical comparisons:

- strongest implemented fixed-confidence Top-k or Best-Set allocation;
- uniform sampling as a diagnostic only;
- same sampler, confidence target, output relation, and preprocessing assumptions.

### 8.4 Measurements

Primary synthetic outcomes:

- certified exact-set recovery and its empirical failure rate;
- interval-resolution and timeout rates;
- heuristic-inclusive exact recovery as a separately labeled diagnostic only;
- forward, inverse, and controlled oracle calls;
- good-projector and verifier calls;
- failure category;
- simple regret as a secondary diagnostic;
- peak coherent qubits and work-register size.

Resource outcomes reported separately:

- gates, depth, arithmetic precision, QROM accesses;
- simulator wall time;
- classical CPU/GPU time and memory;
- state-preparation and reversible-generator cost.

Finite scaling slopes are empirical diagnostics. They cannot prove a polynomial separation or a query lower bound.

For every fixed-confidence plot or table, the numerator for success is certified_exact_recovery. Heuristic-inclusive recovery must never be substituted, pooled with certified recovery, or used to exclude timeout failures.

## 9. LLM threat model and endpoints

The source side uses authorized local open-source models or explicitly permitted endpoints. The victim side is held out by task, repository, CWE, and model family. Repeated generations from one task remain in one split. Victim outputs stay sealed until variants, selector, baselines, primary Q, and analysis code are frozen.

Primary application metrics are:

- ASR@Q: fraction of tasks with at least one attack-predicate success in Q victim queries;
- FV-ASR@Q: fraction with at least one functionally correct, exploit-confirmed vulnerable completion;
- Delta-FV-ASR@Q: attacked FV-ASR@Q minus clean FV-ASR@Q.

Also report functionality retention, fresh-seed success, source-to-victim transfer, query-to-first-success, timeouts and indeterminate outcomes, and separate source/victim/checker budgets. Use paired task-level intervals and a pre-registered functionality non-inferiority gate. Unconfirmed static warnings are not exploit-confirmed successes.

## 10. Claim activation and termination rules

A new quantum algorithm claim requires all of the following:

1. a correct constructive Layer-C heterogeneous-angular-gap upper bound;
2. strict asymptotic separation from the strongest audited black-box composition;
3. a matching all-adaptive-algorithms canonical lower bound with honest sampler scope;
4. a proved orientation dovetail and certificate verifier;
5. charged filtering, state preparation, QROM, arithmetic, precision, gates, depth, and qubits.

An LLM quantum advantage claim additionally requires Layer-P transport and credible reversible generator/checker resource accounting.

Terminate or reframe the new-algorithm claim if the final upper bound collapses to a known composition, the lower bound cannot match it, the result depends on a fixed smaller-side rule, mean gaps are used outside an interior promise, or purification garbage cannot be handled without losing the claimed advantage.

## 11. Safety

Run generated code and exploit checks only in authorized network-isolated containers. Do not target third-party systems, accounts, infrastructure, or live repositories. Privately disclose newly found vulnerabilities. Redact operational payloads when publication would materially enable abuse. Do not automate deployment, persistence, credential theft, or lateral movement.
