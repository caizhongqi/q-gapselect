# Coherent rank-copy cleanup lemma

Status: supporting circuit lemma; not a new query-complexity theorem.

## Setting

Let `E` be a coherent estimator acting on a clean transient register `W`:

\[
  E|0\rangle_W = |\psi\rangle_W
  = \sum_z \alpha_z |z\rangle_W.
\]

The label `z` contains every phase-estimate, sign, reward, and estimator-work
register that remains before output extraction.  Let `f(z)` be the complete
classical Top-k output mask computed from that label.  The durable-copy circuit
acts as

\[
  C_f|z\rangle_W|y\rangle_Y
  = |z\rangle_W|y\oplus f(z)\rangle_Y.
\]

For each possible output `y`, define the projector

\[
  \Pi_y = \sum_{z:f(z)=y}|z\rangle\!\langle z|
\]

and the estimator probability `p_y = <psi|Pi_y|psi>`.

## Exact identity

Starting from clean output and replaying the estimator gives

\[
  (E^\dagger\otimes I_Y) C_f
  (E\otimes I_Y)|0\rangle_W|0\rangle_Y
  = \sum_y E^\dagger\Pi_y|\psi\rangle_W|y\rangle_Y.
\]

The probability that `W` is exactly clean after replay is therefore

\[
  P_{\mathrm{clean}}
  = \sum_y
      |\langle 0|E^\dagger\Pi_y|\psi\rangle|^2
  = \sum_y p_y^2.
\]

Consequently,

\[
  P_{\mathrm{garbage}} = 1-\sum_y p_y^2.
\]

Exact cleanup is possible if and only if one `p_y` equals one, equivalently
`f` is constant on the support of the coherent estimator state.  This is a
circuit identity, not an asymptotic lower bound.

## Consequence for finite QPE

On an exact QPE grid with a strict rank margin, every phase component can map
to the same Top-k mask, so compute--copy--inverse-QPE may clean exactly.  For an
off-grid phase, finite QPE generally has support on several bins.  If those
bins induce different Top-k masks, the output becomes entangled with the
estimator and inverse QPE cannot erase all transient work.  Treating the rank
function as a classical post-processing step would hide this residual.

The final algorithm must therefore do at least one of the following and charge
it explicitly:

1. prove a rounding promise that makes the complete Top-k mask constant over
   the estimator support;
2. use a coherent robustification or randomized-shift construction and bound
   the nonconstant-output probability by the declared failure budget;
3. retain an explicit failure branch and include its garbage/error in the
   certificate; or
4. measure and reset, in which case the method is no longer the claimed
   end-to-end coherent frontier unitary.

## Executable check

The S2 exact-state artifact must report both sides of

\[
  P_{\mathrm{garbage}}
  = 1-\sum_y p_y^2
\]

for every fixture.  On-grid positive controls must have zero residual within
the frozen numerical tolerance.  Off-grid and rounding-boundary controls must
match the predicted residual and fail closed whenever it exceeds the allocated
coherent-estimation error budget.

This lemma closes only the copy/replay accounting question.  It does not prove
that a robust estimator exists at a new cost, that the resulting algorithm
beats composition baselines, or that a matching lower bound holds.
