# CCF-A-level algorithm experiment protocol

Version: preregistered algorithm-only protocol, 2026-07-15

## Scope and interpretation

CCF classifies publication venues, not datasets.  The
[seventh CCF directory](https://www.ccf.org.cn/Academic_Evaluation/TCS/)
lists STOC, SODA, CAV, FOCS, and LICS as class-A theory conferences; it does
not define a set of "CCF-A datasets."  Consequently, this project uses
`CCF-A-level dataset protocol` to mean a reproducible and non-adaptive
benchmark design suitable for a top algorithm paper.  It does not mean that a
public dataset can turn an unproved complexity separation into a theorem.

The paper's primary object remains exact Top-k selection under the canonical
coherent Bernoulli-oracle interface.  No LLM, attack target, commercial API,
or security claim belongs to this experiment stage.

## Evidence tiers

No single tier is sufficient on its own.

| tier | purpose | experimental unit | admissible claim |
|---|---|---|---|
| S: exact-state semantics | Check coherent index execution, controlled forward/inverse calls, stopping history, and cleanup on small instances. | One complete statevector fixture and seed. | Circuit/code semantics only. |
| H: preregistered hard families | Stress gap, active-count, stopping-time heterogeneity, and scaling under fixed logical-query caps. | One frozen non-isomorphic exact-count fixture; seeds are repeated within fixture. | Finite-family query/certificate evidence. |
| P: public-data transfer | Test whether the same selection contract behaves sensibly on independently sourced reward tables. | One frozen dataset split and prediction-matrix shard. | External validity only. |

Tier S must never be replaced by an analytic per-arm estimator.  Tier H may
use the scalable finite-state reference only when its backend is reported as
`analytic finite-state IR`, separately from actual coherent-index execution.
Tier P does not count as worst-case or asymptotic evidence.

## Public datasets and frozen reductions

The preregistered public-data suite is:

| dataset | official size | official split/protocol | role |
|---|---:|---|---|
| UCI Letter Recognition | 20,000 rows, 16 features, 26 classes | First 16,000 for training and last 4,000 for testing, as stated by UCI. | Medium multiclass classifier-selection table. |
| UCI Optical Recognition of Handwritten Digits | 5,620 rows, 64 features, 10 classes | Official `optdigits.tra` and `optdigits.tes`; contributors to train and test are disjoint. | Medium image-feature classifier-selection table. |
| UCI Covertype | 581,012 rows, 54 features, 7 classes | Deterministic stratified split whose row-index hash is committed before outcomes. | Large-table scalability/external-validity panel. |
| scikit-learn Digits | 1,797 rows, 64 features, 10 classes | Bundled offline copy of the UCI digits test set; deterministic stratified fit/selection split. | Environment-independent smoke/diagnostic dataset, not a substitute for the full official Optdigits panel. |

Official sources are the UCI pages for
[Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter%2Brecognition),
[Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits),
and [Covertype](https://archive.ics.uci.edu/dataset/31/covertype).  The offline
fallback is documented by
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
Every downloaded archive, extracted source file, split, classifier catalog,
and prediction matrix must carry a SHA-256 digest in the artifact.

For a dataset, fit a preregistered catalog of deterministic classifiers on the
fit split.  A candidate arm is one classifier/configuration ID.  For frozen
selection row `z`, define

```text
R(i,z) = 1[classifier_i(x_z) = y_z].
```

The resulting Boolean prediction-correctness matrix is the finite Bernoulli
reward table.  The trusted harness may compute empirical arm means only for
post-run scoring and strict-boundary validation.  The candidate and primary
baselines receive a fresh blind oracle, `k`, a failure budget, and one hard
logical-query cap.  They do not receive accuracies, rankings, gap floors,
thresholds, dataset names, test labels, or classifier metadata.

Classifier configurations are selected by a committed catalog and prediction
hash, never by their selection or test accuracy.  Duplicate prediction columns
are removed by hash before the fixture is frozen.  A shard with an exact tie at
the k/(k+1) boundary is retained as a declared negative control or rejected by
a preregistered tie rule; it is never silently replaced after observing method
outcomes.

## Fixed campaign grid

### Hard-family panel

- Families: equal-gap, dyadic-gap, and clustered-gap exact-count families.
- Frozen anchors: 10 per family selected from public structure only.
- Repetitions: 50 independent measurement seeds per fixed fixture.
- Query caps: 65,536; 131,072; 262,144; 524,288; 1,048,576.
- Primary methods: candidate activity-history reference, k-only independent
  adaptive estimation, coarse-partition plus BAI composition, repeated
  single-output selection, and unknown-time variable-time reference.
- Stronger-information control: known stopping schedule, reported separately.

### Public-data panel

- At least 24 distinct classifier prediction columns before choosing Top-k.
- At least 5 preregistered non-overlapping evaluation shards per dataset.
- Independent dimensions: dataset, shard, `k`, query cap, and algorithm seed.
- At least 20 seeds per fixed shard for diagnostics; the confirmatory campaign
  uses 100 or more seeds per shard after its manifest is locked.
- The full official Letter, Optdigits, and Covertype panels are mandatory for a
  final external-validity claim.  The bundled Digits panel is diagnostic only.

## Baselines and fairness contract

The primary quantum comparison panel must match all four items:

1. the same oracle unitary and controlled/inverse access;
2. the same public information (`k`, failure budget, and query cap);
3. the same hard atomic cap on total logical oracle calls;
4. the same output contract: an exact, independently verified Top-k
   certificate, with every timeout or unresolved output counted as failure.

Mandatory primary baselines are independent adaptive QAE, strongest valid
coarse-partition plus quantum BAI composition, repeated single-output
selection, and the best applicable unknown-time variable-time composition.
Known threshold, public gap, or stopping schedules are stronger-information
controls and must not be pooled with the primary comparison.  Classical
sampling baselines are reported under their own sample-access model; no
conversion from classical samples to coherent queries is inferred from wall
time.

## Outcomes and resource accounting

The primary outcome at each fixed cap is

```text
certified_exact = exact Top-k set
                  AND valid independent certificate
                  AND no timeout
                  AND no cap violation.
```

All attempts remain in the denominator.  Secondary outcomes are exact set
agreement without a certificate, timeout rate, incomplete output rate,
cleanup failure, and direct-output completeness.  Resource tables report
forward, inverse, controlled, verification, cleanup, and total logical oracle
calls.  Exact-state runs additionally report qubits, state dimension, gates,
and depth.  Simulator wall time is never called quantum runtime.

## Statistical analysis

The fixture, not the random seed, is the independent experimental unit.

- Report fixed fixture x multiple-seed success and exact one-sided
  Clopper--Pearson lower bounds with simultaneous correction.
- For candidate versus each matched baseline, compute paired risk differences
  within each fixture/cap block and exact McNemar tests.
- Form confidence intervals by cluster bootstrap over fixtures, retaining all
  within-fixture seeds as a cluster.
- Apply Holm family-wise correction across preregistered family x cap x
  baseline comparisons.
- Report the complete success--query-cap Pareto frontier.  A favorable cap may
  not be selected after seeing outcomes.
- Release raw attempt-level records, configuration, manifest hash, source
  hashes, environment, and exclusions.

## Fail-closed claim gate

An experimental quantum-advantage statement is blocked unless every item is
machine-readable as passed:

1. a new upper bound for the same oracle/output contract;
2. a composition frontier showing no known same-interface construction
   matches that bound;
3. a matching lower bound in the same weighted canonical oracle;
4. actual coherent-index execution plus complete resource accounting;
5. complete fixed-cap panels and preregistered fixture-level statistics;
6. independently verified fidelity for every baseline presented as a strongest
   published method (paper-informed stand-ins are insufficient);
7. a nontrivial improvement over every strongest matched baseline without
   post-selection.

The current theorem audit fails items 1--3: public-partition composition
matches or improves on the current candidate proxy, while the hidden-partition
upper bound and weighted matching lower bound remain open.  The exact-state
kernel executes coherent history but does not yet produce a complete direct
multi-output Top-k certificate.  Therefore stronger empirical rates can support
engineering and external-validity claims only; they cannot currently support
a CCF-A quantum-advantage claim.

## Reproduction and promotion policy

The small offline panel is allowed to run before official UCI archives are
available.  Its artifact must say `diagnostic`.  Promotion to `confirmatory`
requires all of the following before execution:

- official source files and hashes are present;
- the dataset/split/catalog/shard manifest is immutable;
- method IDs, caps, repetitions, metrics, exclusions, and statistical tests
  are locked;
- a clean independent rerun reproduces the artifact schema and all panel
  completeness checks;
- the theorem and implementation gates are evaluated without consulting
  empirical outcomes.
