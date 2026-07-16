# Q-GapAttack CCF-A-grade experiment protocol

Version: preregistered design v1, 2026-07-15

Status: design complete; primary implementations, real runs, the quantum upper
bound, the matching lower bound, and the Layer-P transport theorem remain open.
No table in this protocol is a result table.

## 1. Paper question and the three claims that must remain separate

The application question is whether a source-side quantum Top-k selector can
discover a portfolio of semantics-valid attacks that transfers to held-out code
LLMs under a fixed budget.  The quantum question is whether unknown boundary,
unknown heterogeneous stopping time, and direct multi-output selection admit a
new complexity bound under a coherent Bernoulli reward oracle.

The paper must test three claims separately:

1. **Quantum-core claim:** exact fixed-confidence Top-k recovery with fully
   charged oracle, cleanup, history, gate, depth, and qubit resources.
2. **Selector claim:** on one frozen attack candidate graph and one immutable
   source reward tensor, Q-GapSelect selects a stronger portfolio than classical
   and independent-quantum selectors under the same total source budget.
3. **End-to-end attack claim:** the complete Q-GapAttack pipeline increases
   paired functional-vulnerability success on held-out tasks and models relative
   to strong code-security attacks under equal end-to-end budgets.

An empirical attack gain does not prove a quantum advantage.  A canonical
query theorem does not prove an LLM quantum advantage until a reversible Layer-P
generator/checker implementation and its resources are established.  A closed
API is always a classical transfer endpoint.

## 2. Threat models

Two threat models are preregistered and never pooled:

- **Zero-feedback transfer:** all optimization occurs on authorized local source
  models.  The measured Top-k portfolio is frozen before any victim output is
  revealed.  Victim calls are final evaluation only.
- **Online black-box:** victim feedback can guide the attack, with a separate
  victim-query budget.  Results appear in a different table and cannot support
  a zero-feedback transfer claim.

Prompt-only, code-context, dependency/RAG-context, and external-reference
attacks are also separate strata.  TPIA and HACKODE therefore appear only in the
extended-threat panel.

## 3. Baselines

### 3.1 Same-oracle quantum-core baselines

The primary quantum table uses the same coherent Bernoulli reward oracle

\[
U_R\lvert i,z,0\rangle=\lvert i,z,R_i(z)\rangle
\]

and charges every use of \(U_R\), \(U_R^\dagger\), controlled calls, phase-mark
compute/uncompute, verification, history reconstruction, and output maintenance.

| Baseline | Why it is required |
|---|---|
| Uniform QAE + sort | Fixed global precision baseline |
| Adaptive independent AE + sort | Tests whether per-arm adaptivity alone explains the gain |
| Repeated QBAI with winner removal | Strong single-output-to-multi-output composition |
| Coarse boundary localization + QBAI | Already falsified the previous orientation witness |
| Known-time VTS + AE checker | Favorable heterogeneous-time composition |
| Unknown-time VTS + AE checker | Closest published unknown-stopping comparator |
| Quantum subroutine composition | Direct novelty gate for heterogeneous coherent subroutines |
| Generated predicate + optimal all-marked extraction | Tests direct multi-output query, gate, and memory costs |
| Classical CLUCB/Best-Set | Strong same-reward classical fixed-confidence baseline |
| Classical successive accepts/rejects | Second independent adaptive classical implementation |

The relevant starting points are [quantum BAI](https://arxiv.org/abs/2007.07049),
[known-time variable-time search](https://arxiv.org/abs/quant-ph/0609168),
[unknown-time variable-time search](https://arxiv.org/abs/2302.06749),
[multiple-marked extraction](https://arxiv.org/abs/2302.10244), and
[quantum subroutine composition](https://arxiv.org/abs/2209.14146).

Exact-value Durr-Hoyer minimum finding, exact-value k-minima, and known-membership
all-marked enumeration are oracle-aided lower envelopes.  They are reported in
separate columns and are never used for a direct primary win unless their
comparison or membership oracle is compiled from \(U_R\) and fully charged.
Grover, BHMT QAE, QAES, IQAE, and MLAE are estimator or circuit ablations, not
complete Top-k competitors.

The expression

\[
\widetilde O\!\left(\sqrt{\sum_i T_i^2/\Delta_i^2}\right)
\]

is not by itself a novelty claim: an AE checker composed with variable-time
search already suggests this form.  The theorem must beat the strongest valid
composition on a declared infinite family.

### 3.2 Frozen-candidate-pool selector baselines

All methods below receive exactly the same candidate IDs, transformation graph,
source tasks, reward definition, source-model records, total source budget, and
seed schedule.  A method sees only rewards it queried.

| Method | Access and purpose |
|---|---|
| Random portfolio | Lower control for candidate selection |
| Best-of-N | Controls for brute-force multi-sampling |
| Uniform Monte Carlo + empirical Top-k | Exhaustive nonadaptive source ranking |
| Successive Halving | Strong fixed-budget allocation |
| CLUCB/Best-Set Top-k | Strong fixed-confidence allocation |
| Cost-aware racing | Controls for heterogeneous evaluation time without quantum search |
| XOXO Greedy Cayley Graph Search | Closest graph-search prior on the identical graph |
| Independent QAE + sort | Quantum estimator without the proposed joint structure |
| Q-GapSelect | Proposed selector |
| Exhaustive reward oracle | Diagnostic upper bound only; it is not a valid deployable baseline |

XOXO is the primary novelty blocker: its [ACL 2026 work](https://arxiv.org/abs/2503.14281)
already combines semantic code transformations, a Cayley graph, Greedy Cayley
Graph Search, and functional-vulnerability evaluation.  “Semantic transform
graph search” and “CWEval vulnerability induction” therefore cannot be claimed
as new.  The remaining application-level question is whether the new quantum
selection relation improves cost and multi-output transfer on the same graph.

### 3.3 End-to-end code-security attack baselines

The required primary methods are:

- clean/no attack;
- random/Best-of-N semantic variants;
- [INSEC, ICML 2025](https://proceedings.mlr.press/v267/jenko25a.html), using its
  official split and implementation;
- [XOXO-GCGS, ACL 2026 Main](https://arxiv.org/abs/2503.14281);
- [CodeLMSec](https://arxiv.org/abs/2302.04012);
- [DeceptPrompt](https://arxiv.org/abs/2312.04730), re-evaluated with the same
  dynamic functionality and vulnerability oracles;
- Q-GapAttack.

GCG, AutoDAN, PAIR, and TAP are a secondary general-jailbreak stress test.  They
use the code-security \(F\land V\) evaluator, disclose their different access
models, and cannot win the semantics-constrained primary column if their prompt
fails semantic validity.  TPIA and HACKODE are evaluated in a separate
dependency/RAG/external-reference threat table.

## 4. Benchmarks and split rules

The campaign uses four levels:

1. **INSEC official train/validation/test:** exact reproduction of the strongest
   ICML code-completion attack baseline.
2. **CWEval:** primary independent outcome-driven functionality/security test.
   The complete multilingual benchmark is used; a Python-only subset is not the
   primary result.
3. **BaxBench:** end-to-end backend functionality and exploit validation.
4. **SecRepoBench:** repository-level, out-of-distribution validation.

CodeLMSec remains a compatibility benchmark.  Static-analysis-only datasets can
appear as secondary cross-checks but never replace dynamic functionality and
target-vulnerability/exploit oracles.

Splits are group-disjoint across task, repository, CWE template, and model
family.  A global attack strategy/template is learned on source tasks and then
evaluated on unseen tasks; task-specific variants alone cannot establish attack
strategy transfer.  Deduplication hashes, repository ancestry, prompt/template
similarity, and benchmark contamination checks are stored in the dataset
manifest.  All victim records remain sealed until variants, selectors,
hyperparameters, budgets, and primary endpoints are frozen.

## 5. Model protocol

The source side contains at least three open-source code-model families.  The
victim side contains at least five held-out families; optional current commercial
models are classical transfer endpoints only.  Exact model choice is frozen by
an immutable manifest before the test split is opened.

Every run records model ID and revision, tokenizer revision, chat template,
system prompt, context length, precision/quantization, decoding temperature,
top-p, maximum output length, framework/container version, and hardware.  Source
and victim model families are disjoint for the strict transfer table.

Open models use paired clean/attack seeds where deterministic coupling is valid.
If a closed service cannot guarantee seed coupling, it reports marginal
\(\Delta\)-FV-ASR with task-cluster uncertainty and is not labeled paired
counterfactual.

## 6. Experiment panels

### Q1: fixed-confidence correctness

- \(n\in\{8,16,32,64,128,256\}\);
- \(k\in\{1,2,4,\lfloor n/4\rfloor,\lfloor n/2\rfloor\}\);
- \(\delta\in\{0.05,0.01\}\);
- equal, geometric, clustered, dyadic, endpoint, and natural-purification
  instances;
- at least 500 independently generated instances per grid cell.

The primary event is certified exact-set recovery.  Timeout, unresolved output,
or incomplete verification is a failure, regardless of whether a heuristic fill
happens to match the truth.

### Q2: composition and heterogeneous scaling

Vary \(n,k\), near-boundary crowding, minimum gap, reward variance, cost
coefficient of variation, heavy-tailed stopping times, gap-cost correlation, and
whether individual stopping times are known.  Include hard-but-cheap and
easy-but-expensive controls.  Report success against primitive-query budget and
weighted variable-time cost.  Finite slopes are diagnostics, not proofs.

### Q3: oracle and Layer-P audit

Separate canonical rotation, natural purification, compiled comparison, exact
value, and known-membership access.  Stress arm-dependent garbage and verify
full-workspace cleanup.  No result from stronger access appears in the primary
same-oracle win count.

### A1: search-algorithm isolation

Freeze one transformation graph and cached reward tensor.  Use

\[
N\in\{64,256,1024,4096,16384\},\qquad
k\in\{1,4,16,64\}.
\]

Vary boundary gap, reward variance, evaluation-time heterogeneity, success
sparsity, graph depth, and branching factor.  This panel can emulate quantum
selection on a classical reward table, but it cannot be described as an
end-to-end quantum run.

### A2: real source attack

Generate code on local source models, compile/run it in isolated benchmark
containers, and execute frozen functionality and vulnerability validators.
Compare all mandatory code-security attacks.  The primary endpoint is paired
clean-safe to attack-functional-vulnerable boundary crossing at a frozen budget.

### A3: zero-feedback transfer

After source selection, measure and freeze the Top-k portfolio.  Construct the
complete source-to-victim matrix

\[
M_{s,t}(Q,k)=\operatorname{Transfer\mbox{-}FV\mbox{-}ASR}_{s\to t}@Q,k.
\]

Rows are source families and columns are held-out victim families.  Also report
unseen task, CWE, repository, language, and decoding-seed transfer.

### A4: budget curves, ablations, and external validity

Report the full source/victim budget surface, all component ablations, BaxBench
exploits, SecRepoBench repository completions, and the separately stratified
TPIA/HACKODE context conditions.

## 7. Budget and resource accounting

The source-query grid is

\[
Q_S\in\{16,32,64,128,256,512,1024,2048,4096,8192\},
\]

the victim-query grid is

\[
Q_V\in\{1,2,4,8,16,32,64\},
\]

and \(k\in\{1,4,8,16,64\}\).  The large source budgets are necessary for a
fair XOXO-GCGS comparison.  The primary \(Q_S,Q_V,k\) triple is frozen before
test access; all other points form a preregistered curve.

One complete `(task, strategy, decoding seed)` generation counts as one model
query.  Returning multiple completions counts each completion.  Separate
ledgers record:

- source generation and source reward-oracle calls;
- quantum forward, inverse, controlled, and powered primitive calls;
- victim generation calls;
- attacker-model calls;
- functionality and security/exploit checker calls;
- input/output tokens;
- CPU validation seconds, GPU seconds, wall time, and monetary cost;
- logical gates, depth, T-count, qubits, QROM, history, and cleanup resources.

Quantum oracle calls are never equated with commercial API calls.  Query-matched
and compute/time-matched curves are both reported.  Every smaller budget is an
immutable prefix of the same raw stream.

## 8. Metrics

Let \(F(c)\) denote functional correctness and \(V_j(c)\) the target-CWE or
exploit oracle.  The primary attack event is

\[
B_g(z)=F(c_0)\land\neg V_j(c_0)\land F(c_g)\land V_j(c_g),
\]

using a shared seed when valid.  Primary outcomes are:

- paired counterfactual ASR@Q;
- targeted FV-ASR@Q and paired \(\Delta\)-FV-ASR@Q;
- held-out Transfer-FV-ASR@Q,\(k\);
- the full resource vector under a matched budget.

Secondary outcomes include FV-Precision@k, any-vulnerability rate, functionality
retention, semantic-validity rate, compile rate, query/time to first success,
AUC over the preregistered query grid, unique task/CWE coverage, timeouts, and
indeterminate validation.  Targeted and any-vulnerability success are never
pooled.  Repeated generations never enlarge the task-level denominator.

## 9. Statistical analysis

- Experimental cluster: task or repository, with all victim models and
  completions for that unit kept together.
- Repetition: at least five search seeds by five decoding seeds; the primary
  subset uses ten by ten.
- Intervals: 95% task-cluster bootstrap with 10,000 resamples, stratified by
  benchmark/CWE/language where preregistered.
- Fixed-budget paired binary comparison: exact McNemar test and absolute risk
  difference.
- AUC/resource comparison: paired cluster randomization or bootstrap difference.
- Time to success: Kaplan-Meier or another preregistered censored-time analysis;
  failures are censored at the budget, not dropped.
- Multiplicity: Holm family-wise correction inside each hypothesis family.
- Reporting: absolute effect, relative effect where defined, confidence interval,
  adjusted p-value, median/IQR resources, and all negative/null results.

Macro and micro aggregates are both shown, with macro summaries by CWE,
language, repository, benchmark, and model family.  A power calculation is run
before test access.  At least 10% of outputs receive stratified blinded human
audit by two raters, with Cohen's kappa or Krippendorff's alpha; human labels do
not replace the dynamic primary oracle.

## 10. Required ablations

Quantum-core ablations:

- fixed maximum precision;
- no adaptive precision;
- serial instead of variable-time execution;
- serial history rebuild;
- repeated single-output extraction;
- known boundary;
- no independent verification;
- Layer C versus Layer P;
- invalid free-history QRAM, visibly marked as an invalid negative control.

Application ablations:

- unpaired reward instead of paired counterfactual reward;
- no cost awareness;
- no activity history;
- no transfer regularization;
- candidate-pool size and graph depth/branching;
- portfolio size and source-model count;
- fresh task, fresh seed, fresh CWE, and fresh repository.

## 11. Reproducibility and artifact rules

Before execution, freeze dataset, model, validator, transformation-graph, split,
container, and hyperparameter manifests with hashes.  Each raw row stores the
commit, config hash, run ID, method, source/victim model revisions, task/split,
candidate/strategy ID, all seeds, budget ledgers, validator revisions, status,
and output reference.  Raw generations and sensitive exploit details remain in
access-controlled storage; paper artifacts contain redacted evidence and hashes.

Checkpoint/resume must preserve immutable query order.  Failed jobs, timeouts,
and indeterminate validators remain in the raw data and denominator.  At least
one independent reproduction of the primary table is required before a strong
empirical claim.

## 12. Claim gates

The empirical attack claim is eligible only after all mandatory baselines run on
the same frozen campaign and the primary effect survives clustered uncertainty.
The quantum-algorithm claim additionally requires a constructive upper bound,
strict separation from the strongest valid composition, and a matching
all-algorithms lower bound.  An LLM-domain quantum-advantage claim additionally
requires a credible reversible Layer-P source generator/checker with cleanup and
resource accounting.

Until all three gates pass, permitted wording is:

> We preregister a quantum-core and transferable code-LLM attack evaluation;
> current artifacts validate interfaces and finite simulations only.

Forbidden wording includes “CCF-A ready,” “state-of-the-art,” “quantum speedup,”
or “leading ASR” based on planned tables, synthetic fixtures, analytic proxies,
or classical emulation.
