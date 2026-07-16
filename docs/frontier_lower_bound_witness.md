# S3 composition-frontier and lower-bound witness

Status: finite falsification artifact; no matching lower-bound or composition
theorem.

The executable entry point is
`scripts/run_frontier_lower_bound_witness.py`. It produces three deliberately
separate witness types. Every JSON record contains `witness_type`,
`oracle_interface`, `computed_quantity`, `verified_local_statement`,
`explicit_non_theorem_boundary`, `composition_match`, and
`composition_kill_flag`.

## 1. Canonical-rotation paired hybrid

The repository oracle uses the real block

\[
B_\theta=
\begin{pmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{pmatrix}
=R_y(2\theta).
\]

The quantum-gate label `R_y(2 theta)` must not be confused with the planar
rotation angle. Direct singular-value decomposition and the analytic identity
both give

\[
\lVert B_\theta-B_\phi\rVert
=2\sin\frac{|\theta-\phi|}{2}.
\]

The same norm holds for inverse calls and for a controlled call because the
inactive control block is identical. The test suite includes the endpoint
`theta=pi/2`, `phi=0`: the norm is `sqrt(2)`, not `2`. This is a regression
guard against changing parameterizations silently.

For an explicit pair of instances whose unique Top-k sets differ, an algorithm
with error at most \(\varepsilon<1/2\) on both instances induces output
distributions with total variation at least \(1-2\varepsilon\). Replacing the
oracle in one query at a time gives the standard hybrid inequality

\[
D(\rho_x^T,\rho_y^T)
\le T\lVert B_x-B_y\rVert.
\]

Therefore the configured pair certifies only

\[
T\ge
\frac{1-2\varepsilon}
{2\sin(|\theta_{\rm high}-\theta_{\rm low}|/2)}.
\]

This statement covers arbitrary inter-query unitaries and forward, inverse,
or controlled canonical calls. It is nevertheless only a two-input local
barrier. It supplies neither an \(n\)-arm direct sum nor an activity-history or
multi-output factor. Belovs' general-oracle adversary framework confirms that
arbitrary-unitary input oracles require an explicitly general-oracle analysis;
the present code does not claim to instantiate that full framework
([arXiv:1504.06943v1](https://arxiv.org/abs/1504.06943v1)).

## 2. Fixed-weight Johnson adversary witness

For the standard discrete bit-query model, inputs are all weight-\(k\) strings
of length \(n\), and the output is the complete support. The finite adversary
matrix \(\Gamma\) is the adjacency matrix of the Johnson graph: two supports
are adjacent exactly when one selected and one unselected coordinate are
swapped.

The runner materializes the full matrix for small \(n\), computes

\[
\frac{\lVert\Gamma\rVert}
{\max_i\lVert\Gamma\circ\Delta_i\rVert},
\qquad
(\Delta_i)_{xy}=\mathbf 1[x_i\ne y_i],
\]

and checks symmetry, a zero diagonal, output-separating support, and every
filtered spectral norm. The configured cases numerically reproduce

\[
\lVert\Gamma\rVert=k(n-k),\qquad
\lVert\Gamma\circ\Delta_i\rVert=\sqrt{k(n-k)},
\]

so the finite objective is \(\sqrt{k(n-k)}\). The spectral-ratio adversary
formulation is standard; the pinned primary reference is
[Høyer--Lee--Špalek, arXiv:quant-ph/0611054v2](https://arxiv.org/abs/quant-ph/0611054v2).

The program verifies these equalities only on the materialized cases. More
importantly, this is a discrete symbol oracle, not the repository's continuous
rotation oracle. Multiplying the Johnson factor by an inverse-angle factor
without a general-oracle adversary construction would be an unsupported
composition. The JSON therefore sets `matching_lower_bound_claimable=false`.

## 3. Same-oracle-model finite comparator diagnostic

The composition checker independently executes (i) the tiny two-level S3
stopping-history circuit and (ii) the existing all-arm coherent-QPE,
rank-copy, and cleanup circuit on the same exact-grid fixture. The two results
come from distinct implementation classes and distinct oracle instances. The
harness computes `finite_fixture_query_dominance_verified` only if all of the
following hold:

1. both executions use the same canonical rotation-oracle model and the same
   harness-owned fixture hash;
2. the oracle objects and result implementation classes are distinct;
3. complete output, cleanup, and trusted-answer agreement for both executions;
4. separate exact reconciliations of forward, inverse, controlled-forward, and
   controlled-inverse query counts, with zero QRAM charge; and
5. the independently reconstructed comparator count is no larger than the
   executed candidate count under the declared tolerance.

On each default case, the true-coherent S3 candidate executes 176 canonical
queries while the all-arm comparator executes 28. However, neither
implementation exposes the registered `(n,k,delta,atomic_query_cap)` public
interface, and both explicitly issue no delta-sound correctness certificate.
Consequently `composition_match=false` and `composition_kill_flag=false` even
when the finite dominance diagnostic passes. The comparison does not verify
that the comparator faithfully implements Rall, Jeffery, Low--Su, or another
published strongest baseline, and it says nothing by itself about asymptotic
scaling. The strong-composition registry still has nine required uncovered
rows, so the global composition frontier remains open.

## Relation to hidden-frontier fixtures

`hidden_frontier_fixtures.py` correctly keeps private angles, gap, family,
ranking, schedule, and permutation out of the algorithm view. The S3 pair
angles and correct masks live in the trusted witness harness; they are not
algorithm inputs. The pair witness is a lower-bound adversary construction,
where defining the two possible private inputs is necessary and does not grant
that information to the evaluated algorithm.

The public-partition control remains a distinct stronger interface. It is
excluded from the blind S3 control aggregate because none of those methods
accepts the partition; counting it would duplicate the hidden-frontier orbit.

## What is and is not established

Established for the recorded finite cases:

- the exact canonical rotation difference norm, including inverse and control;
- a local two-input hybrid query lower bound;
- feasible finite Johnson positive-adversary matrices and their computed
  objectives; and
- a fail-closed, same-fixture finite query-dominance diagnostic with no
  composition match or kill.

Not established:

- a Johnson/direct-sum factor for the continuous canonical oracle;
- a variable-time activity-history adversary bound;
- a direct multi-output lower bound for arbitrary adaptive algorithms;
- a matching asymptotic upper and lower bound;
- fidelity-complete strongest published baselines;
- quantum advantage or CCF-A readiness.

The next proof target is not another proxy curve. It is a general-oracle
adversary or polynomial construction that combines continuous-angle
indistinguishability with the fixed-weight multi-hypothesis structure under the
same complete-output relation. A failed construction is informative: it means
the pair and Johnson factors cannot yet be multiplied.
