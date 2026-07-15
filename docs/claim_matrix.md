# Q-GapSelect Claim Matrix

Version: audited matrix, 2026-07-14

This file controls manuscript wording. A status may be strengthened only by a change that links a proof or immutable artifact and survives independent review. Finite code checks are never mathematical proofs.

Status vocabulary: SPECIFICATION, KNOWN, CODE-SANITY, PROPOSED, CONJECTURE, PROOF-OBLIGATION, EMPIRICAL-PENDING, INACTIVE, REQUIRED-POLICY, REJECTED.

## A. Oracle and problem claims

| ID | Statement | Status | Evidence now | Activation requirement | Falsifier or downgrade | Permitted wording now |
|---|---|---|---|---|---|---|
| O-01 | Layer C is the complete controlled block rotation B_mu with both R_y basis-column actions specified. | SPECIFICATION | Equations in paper/main.tex | Implementations and proofs use this exact unitary and count inverse/controlled calls | One-column-only definition, hidden phase, or uncharged control | We study this canonical block model. |
| O-02 | Layer P is a natural purification A with arm/work/reward registers, A dagger, and a known good projector. | SPECIFICATION | Formal definition in paper/main.tex | Every Layer-P result accounts for work garbage and available reflections | Treating a generator as a clean reward qubit | The application naturally yields purification access. |
| O-03 | A general Layer-P sampler can be converted to a clean B_mu block at unit cost. | REJECTED | Conversion would erase work and synthesize an unknown angle | Not activatable without an explicit charged transduction theorem | Any step silently performs the conversion | Never state or assume free compression. |
| O-04 | A canonical lower bound gives a worst-case lower bound over compatible purification samplers. | PROPOSED-SCOPE | Canonical instances are members of the broader class | Formal reduction with identical query accounting | Extra promises exclude canonical instances | A canonical lower bound has compatible-sampler worst-case scope. |
| O-05 | A canonical lower bound applies instance-wise to every purification with the same means. | REJECTED | Work registers may leak arm or hypothesis information | Requires a no-leakage promise or full-state-fidelity theorem | Two purifications with different information content | Never claim an arbitrary-purification instance-wise bound. |
| G-01 | The primary local gap is gamma_i, the amplitude-angle distance to the opposite Top-k boundary. | SPECIFICATION | Definition and reward-state overlap calculation | Use throughout upper/lower statements | Raw mean gap substituted near endpoints | We measure local difficulty in angular gaps. |
| G-02 | The local coherent discrimination scale is Theta(1/gamma_i). | KNOWN-LOCAL-FACT | Overlap cos(theta-theta-prime) plus amplitude discrimination | Cite or prove the exact bounded-error statement used | A subroutine uses a different oracle or information promise | The canonical local scale is inverse angular gap. |
| G-03 | Raw mean-gap cost Theta(1/Delta_i) is instance-tight on the full interval [0,1]. | REJECTED | Endpoint counterexamples; derivative of arcsin sqrt(x) diverges | Not activatable without an interior promise or a different model | Means approach zero or one | Mean gaps are reporting quantities only. |
| G-04 | On [eta,1-eta], 2 sqrt(eta(1-eta)) gamma_i <= Delta_i <= gamma_i. | SPECIFICATION-COROLLARY | Mean-value theorem | State the common interior promise | Compared means leave the interval | Under an interior promise we translate angular to mean gaps. |

## B. Prior work and novelty boundary

| ID | Statement | Status | Evidence now | Activation requirement | Falsifier or downgrade | Permitted wording now |
|---|---|---|---|---|---|---|
| K-01 | Fixed-confidence quantum best-arm identification has heterogeneous gap-dependent results for one selected arm. | KNOWN | Wang et al. 2021 | Preserve its input model and output scope | Using it as an exact multi-output theorem | Prior work solves the single-output setting. |
| K-02 | Approximate k-minimum finding with approximate values is known. | KNOWN | Gao, Ji, and Wang, ESA 2025 | Compare against its strongest applicable corollary | Omitting expectation-value oracle construction cost | Approximate k-minimum is a mandatory baseline. |
| K-03 | All marked elements can be extracted in O(sqrt(Nk)) queries with efficient gates. | KNOWN | van Apeldoorn, Gribling, and Nieuwboer, Quantum 2024 | Preserve exact predicate/access assumptions | Relabeling it as a new batch primitive | Marked extraction is prior work. |
| K-04 | Variable-time quantum search has improved upper and lower bounds. | KNOWN | Ambainis, Kokainis, and Vihrovs, TQC 2023 | Charge variable stopping-time implementation | Claiming variable time itself as new | Variable-time search is a component. |
| K-05 | Median/order statistics and deterministic minimum/k-minimum have established query bounds. | KNOWN | Nayak-Wu; Durr et al.; later k-minimum work | Compare in matching access models | Treating deterministic exact values as stochastic samples | These results delimit novelty. |
| K-06 | QBoundary, QBatchExtract, and equal-gap Top-k are new quantum primitives. | REJECTED | They reduce to or overlap known approximate selection, marked extraction, and order-statistic machinery | Only a strictly stronger heterogeneous theorem can support novelty | Same asymptotic result from cited black boxes | Never advertise these labels as innovations. |
| N-01 | A strict heterogeneous fixed-confidence theorem stronger than every valid black-box composition may be new. | INACTIVE-CANDIDATE | Narrowed literature audit | Prove the theorem, strict separation, matching lower bound, and resource feasibility | Bound collapses to known composition or separation vanishes | This is the only retained algorithmic novelty candidate. |

## C. Orientation and upper-bound claims

| ID | Statement | Status | Evidence now | Activation requirement | Falsifier or downgrade | Permitted wording now |
|---|---|---|---|---|---|---|
| R-01 | Always enumerating min(k,n-k) is instance-optimal. | REJECTED | Explicit n=3m, k=m heterogeneous counterexample | Not activatable without defeating the archived family | Fine-layer ratio Theta(sqrt(m)) | The fixed smaller-side rule is false. |
| R-02 | H_b = sum_r sqrt(N_r(M_(r,b)+1))/epsilon_r and H_orient = min(H_+,H_-) are the correct candidate accounting quantities. | CONJECTURE | Definition and asymmetry sanity analysis | Constructive upper plus matching relation lower bound | Profile where a valid algorithm has asymptotically different cost | We investigate an orientation-aware candidate functional. |
| R-03 | Dovetailing selected and rejected procedures lets the first verified certificate achieve the faster valid orientation. | PROOF-OBLIGATION | High-level budget schedule only | Resumable algorithms, verifier, confidence split, and first-finish cost proof | Restart/measurement overhead or invalid certificate destroys the bound | Orientation dovetail remains to be proved. |
| U-01 | A reversible approximate angular boundary routine can be composed with bounded-error classifiers. | PROOF-OBLIGATION | Known approximate selection is only a component | Complete noisy-predicate correctness and query lemma | Reuse of approximate data creates correlated unbounded error | A boundary-composition lemma is required. |
| U-02 | Arm-dependent angular classifiers can be composed without a worst-gap multiplication. | PROOF-OBLIGATION | Variable-time search literature | Explicit unitary stopping construction and cleanup | Worst precision is paid on all arms | We seek a heterogeneous variable-time composition. |
| U-03 | Newly certified outputs can be batch-extracted and independently verified at the charged layer cost. | PROOF-OBLIGATION | Known exact marked extraction | Noisy predicate composition, output accounting, and verifier proof | Predicate errors or repeated setup dominate | Batch extraction itself is known; this composition is open. |
| U-04 | Reversible activity filtering avoids hidden linear history reconstruction. | PROOF-OBLIGATION | No construction yet | Data structure, QROM, uncomputation, and amortized bound | Rebuilding filters dominates H_orient | Filtering costs must be explicit. |
| U-05 | Layer-C Q-GapSelect uses soft-O(H_orient polylog(n,1/delta,1/gamma_min)) calls. | CONJECTURE | Candidate architecture only | U-01 through U-04, R-03, full confidence proof, resource analysis | Counterexample, hidden worst gap, or larger proved cost | We conjecture a canonical upper bound. |
| U-06 | The same reward-query expression holds under Layer-P access. | CONJECTURE | No transport proof | Direct construction with A, A dagger, good projector, and charged garbage handling | A clean-rotation step or generator cost erases the bound | Natural-sampler transport is a separate conjecture. |
| U-07 | The candidate upper strictly beats the strongest black-box combination on an explicit family. | PROOF-OBLIGATION | No completed separation | Derive all same-model black-box bounds and prove asymptotic separation | Any audited combination matches the candidate | Strict separation is required, not yet shown. |

## D. Lower-bound claims

| ID | Statement | Status | Evidence now | Activation requirement | Falsifier or downgrade | Permitted wording now |
|---|---|---|---|---|---|---|
| L-01 | Canonical single-coordinate angular discrimination costs Omega(1/gamma_i). | KNOWN-LOCAL-FACT | Two-state overlap/hybrid reasoning | State exact promise and bounded-error constants | Auxiliary information changes the model | This is a local lower-bound ingredient. |
| L-02 | Equal-gap fixed-weight recovery is a new lower-bound contribution. | REJECTED | Core phenomena covered by approximate k-minimum, marked extraction, and order statistics | Only use as a sanity reduction | Novelty rests only on equal gaps | Equal-gap recovery is non-new sanity evidence. |
| L-03 | Disjoint dyadic heterogeneous blocks obey the required direct sum under superposition queries. | PROOF-OBLIGATION | General adversary/direct-product tools exist | Explicit composition or strong direct-product proof | Merely adding per-block lower bounds | We seek an all-algorithms dyadic direct sum. |
| L-04 | A weighted adversary relation complexity matches H_orient up to declared logarithms. | CONJECTURE | No SDP factorization or dual witness | Both inequalities and explicit hard family | Asymptotic mismatch between quantities | Matching tightness remains open. |
| L-05 | The lower bound holds against all adaptive quantum algorithms, not only estimate-then-sort. | PROOF-OBLIGATION | Target scope fixed | Adversary/polynomial proof under the same oracle | Proof restricts the algorithm class | An all-algorithms lower bound is required. |
| L-06 | The final lower bound transfers to compatible samplers only in worst-case scope. | REQUIRED-SCOPE | O-04 and O-05 audit | State scope in theorem and abstract | Instance-wise arbitrary-purification wording appears | The sampler claim is worst-case only. |
| C-01 | The best classical comparator is evaluated under the same sampler, confidence, output relation, and preprocessing. | PROOF-OBLIGATION | Classical Best-Set literature identified | Strongest same-interface upper/lower comparison | Classical baseline is only uniform sampling or gets weaker access | A same-interface classical audit is required. |

## E. Matroid and application claims

| ID | Statement | Status | Evidence now | Activation requirement | Falsifier or downgrade | Permitted wording now |
|---|---|---|---|---|---|---|
| M-01 | Q-BatchExchange can identify a maximum-mean matroid basis without enumerating all bases or exchange pairs. | PROPOSED-DEFERRED | High-level exchange design | First complete Top-k theorem; constructive matroid proof | Exponential enumeration or uncharged independence data | We propose a deferred matroid extension. |
| M-02 | Angular exchange gaps govern local comparisons; gamma_e is +infinity if a basis element has no outside exchange competitor. | SPECIFICATION | Definition in paper/main.tex | Use consistently in later theorem | Undefined maximum over an empty exchange set | Empty exchange sets have infinite local gap. |
| M-03 | Matroid complexity matches an orientation-aware weighted adversary quantity. | CONJECTURE | No proof | Upper, lower, partition direct sum, uniform-matroid recovery | Global square root across independent required outputs | The matroid theorem is conjectural. |
| LM-01 | Complete semantics-preserving attack variants define arms whose reward is functional and exploit-confirmed vulnerability on a source generator. | PROPOSED-MAPPING | Formal reward definition | Frozen variants, tasks/seeds, functional tests, exploit checker | Interacting atomic edits are treated as additive independent arms | We map attack-portfolio selection to Top-k. |
| LM-02 | A reversible source generator naturally supplies Layer-P rather than clean Layer-C access. | MODEL-BOUNDARY | Generator state includes seed/output/checker garbage | Resource ledger and reversible construction | API calls counted as coherent or garbage erased for free | The application interface is purification access. |
| LM-03 | Proprietary victim APIs can be queried coherently. | REJECTED | APIs expose only classical request/response access | Not activatable | Any coherent-victim query count | Victims are queried classically after measurement. |
| LM-04 | Q-GapSelect currently yields a genuine LLM quantum-query advantage. | INACTIVE | U-05, U-06, lower bound, and resource analysis are open | Complete theorem stack plus credible reversible source resources | Transport fails or source cost dominates | No LLM quantum advantage is claimed. |
| E-01 | Synthetic heterogeneous profiles show a lower executed query count than all audited baselines. | EMPIRICAL-PENDING | No qualifying run | Frozen future protocol, executable algorithms, manifests, uncertainty analysis | No gain or accounting mismatch | We will evaluate this endpoint. |
| E-02 | The attack improves ASR@Q on held-out victims. | EMPIRICAL-PENDING | No victim results | Frozen Q, task/model splits, paired intervals | Interval includes no improvement or test leakage | Do not claim an improvement. |
| E-03 | The attack improves FV-ASR@Q and Delta-FV-ASR@Q while retaining functionality. | EMPIRICAL-PENDING | No victim results | Exploit confirmation, paired intervals, non-inferiority gate | Functional loss, unconfirmed vulnerability, or null result | These are future primary endpoints. |
| E-04 | The attack transfers across model families without adaptive victim tuning. | EMPIRICAL-PENDING | No transfer matrix | At least three frozen held-out families and zero victim-search protocol | Victim tuning or inconsistent transfer | Transfer will be tested. |
| E-05 | Simulator runtime demonstrates quantum speedup. | REJECTED | Query-model audit | Never activatable | Hardware/software simulation artifacts | Never use this wording. |

## F. Artifact and policy claims

| ID | Statement | Status | Evidence now | Activation requirement | Falsifier or downgrade | Permitted wording now |
|---|---|---|---|---|---|---|
| S-01 | The current artifact implements angular-gap clustered, planted-dyadic, endpoint, and complement-asymmetry random-instance campaigns. | REJECTED-AS-CURRENT | Repository audit finds deterministic scaling proxies and four fixed reference instances only | Future implementation and immutable random-instance manifests would create a new claim | Fixed complement_orientation reference is mislabeled as the asymmetry family | These campaigns are future work. |
| S-02A | run_scaling implements five deterministic mean-space equal/geometric analytic-proxy regimes. | CODE-SANITY | Script, configuration, and scaling artifact | Keep disclosure synchronized with code | Scenario suite changes | The scaling pipeline contains five deterministic proxies. |
| S-02B | run_reference implements four fixed instances with configurable derived seeds and sampled Grover measurement laws. | CODE-SANITY | reference_experiments.py, configuration, tests, and versioned diagnostic artifact | Keep claim limited to analytic-measurement regression | Described as coherent batch execution or random-instance generation | The reference pipeline is a multi-seed analytic-measurement regression. |
| S-03A | The run_scaling trials field creates repeated observations. | REJECTED-AS-CURRENT | run_scaling records but does not loop over trials | A future implementation could create a new claim | Current deterministic calculation | run_scaling does not use trials for repetition. |
| S-03B | The run_reference trials field controls its scenario/seed loop. | CODE-SANITY | run_reference_experiments loops over range(config.trials) | Tests and manifests preserve resolved configuration | Loop semantics change | run_reference uses trials for analytic measurement repetitions. |
| S-04 | A default 500-trial-per-fixed-scenario reference artifact is committed. | REJECTED-AS-CURRENT | Config schedules 500, but checked artifact is a four-trial diagnostic with 16 records | Recorded default run and immutable artifact | Missing or incomplete records | The 500-trial default is configured, not committed as a completed run. |
| S-05 | Passing current tests supports the asymptotic theorem. | REJECTED | Finite proxy and regression tests cannot prove asymptotics | Never activatable | Any theorem wording based on tests | Tests are code sanity only. |
| S-06 | certified_exact_recovery is the eligible reference fixed-confidence success field. | SPECIFICATION | New reference schema requires interval_resolved and exact output | Preserve raw/aggregate recomputation and timeout accounting | Unresolved or timeout trial counted as certified | Fixed-confidence success means certified exact recovery. |
| S-07 | heuristic_inclusive_exact_recovery can be used as fixed-confidence success. | REJECTED | It includes exact outputs after timeout and empirical completion | Never activatable under the current semantics | Any fixed-confidence table pools it with certified recovery | Heuristic-inclusive recovery is diagnostic only. |
| S-08 | The direct canonical threshold flag is computed from charged controlled reward-oracle calls rather than a supplied marked-index set. | CODE-SANITY | `direct_phase.py`, embedded-oracle tests, phase-grid and involution regressions | Preserve full-workspace compute/phase/uncompute semantics and exact query ledger | Hidden means or a target set enters the flag API | The artifact directly computes a finite-QPE threshold reflection in the canonical model. |
| S-09 | Full-workspace BBHT executes rank-one diffusion, joint predicate/index decoding, output exclusion, and fresh verification. | CODE-SANITY | `direct_search.py`, per-attempt trace and query-formula tests | Preserve peak-state, confidence-allocation, and non-absence failure semantics | Index-only diffusion, uncharged decode, or budget exhaustion called absence | The artifact directly searches the finite-QPE predicate; this is not a new theorem. |
| S-10 | The calibrated direct Top-k controller is a proved heterogeneous-gap quantum advantage. | INACTIVE | Its boundary calibration still samples every arm; the phase-resolution condition is an execution guard | New direct boundary algorithm, upper/lower bounds, and same-model separation | Calibration dominates or the bound reduces to known composition | The controller is an auditable end-to-end reference, not an advantage result. |
| ETH-01 | Experiments remain in authorized, isolated benchmarks with responsible disclosure. | REQUIRED-POLICY | Protocol is specified | Execution logs, sandboxing, licenses, disclosure record when needed | Third-party targeting or operational abuse | Experiments are restricted to authorized benchmarks. |

## Activation dependencies

    O-01 + G-01 + R-02 + U-01..U-04 + R-03  ->  U-05
    O-02 + U-05 + direct purification constructions             ->  U-06
    L-01 + L-03 + L-04 + L-05 + L-06                            ->  matched lower bound
    U-05 + U-07 + matched lower bound + full resources           ->  N-01
    N-01 + matroid construction + partition direct sum           ->  M-03
    U-06 + credible reversible source resources + frozen study   ->  eligible LLM quantum test

Empirical LLM improvements do not prove N-01. A canonical theorem without U-06 does not justify an LLM quantum advantage.

## Status-change checklist

1. Link the proof section or immutable run manifest.
2. Record exact commit, configuration, and command.
3. Confirm that oracle, comparator, and confidence target did not change.
4. Add a falsification test or counterexample search.
5. Search manuscript, abstract, captions, README, and documentation for stale stronger wording.
6. Record negative and null results.
7. Obtain an independent proof or code review.
