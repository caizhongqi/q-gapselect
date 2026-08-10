# Neural collision topology: formal layer and claim boundary

This layer turns the phrase *collision topology* into a finite, falsifiable
object. It does not treat a new name as evidence of a new phenomenon.

## 1. Thresholded conditional-collision complex

For attack candidates `A`, benign anchors `B`, control distance `d_c`, payload
distance `d_p`, and functional divergence `d_g`, define

\[
G_{\epsilon,\Delta,\Gamma}=(A\sqcup B,E),
\]

with an edge `(a,b)` exactly when

\[
d_c(a,b)\le\epsilon,
\qquad d_p(a,b)\ge\Delta,
\qquad d_g(a,b)\ge\Gamma.
\]

Only edge-bearing vertices form the active collision complex. The graph is a
one-dimensional CW complex, so its first two Betti numbers are exact:

\[
\beta_0=\text{number of active connected components},
\]

\[
\beta_1=|E|-|V_{\rm active}|+\beta_0.
\]

Varying `epsilon` produces a monotone control-threshold filtration. A future
persistent analysis must use this filtration rather than a single favourable
threshold.

## 2. Frozen topology profile

The implemented profile is

\[
\mathfrak T_C=(\beta_0,\beta_1,H_C,D_C,K_C).
\]

- `H_C` is Shannon entropy of edge mass across active connected components.
  Disjoint equally sized collision basins have high entropy; one dominant
  basin has low entropy.
- `D_C` is not called a topological dimension. It is the entropy/stable rank of
  non-control hidden displacement vectors carried by collision edges.
- `K_C=nu(G)` is exact maximum bipartite matching capacity.
- `beta_1` records redundant collision cycles that neither raw edge count nor
  maximum matching can detect.

The descriptors are deliberately non-redundant. A star and a set of disjoint
claws can have the same number of edges but different entropy and capacity; a
complete `K_2,2` component has a non-zero cycle rank.

## 3. Retrospective evidence audit

The previously frozen Digits and Fashion-MNIST summaries identify only:

- matching-capacity fraction `K_C/min(|A|,|B|)`;
- a displacement spectral-rank proxy for `D_C`.

They did not persist graph adjacency, so `H_C`, `beta_0`, and `beta_1` cannot be
reconstructed honestly after the fact. The 16-cell retrospective audit finds:

- openness versus capacity correlation: `0.8122215`;
- displacement rank versus capacity correlation: `0.7756870`;
- openness versus displacement rank correlation: `0.9108010`.

These results support a partial capacity-dimension profile, not full neural
collision topology. All future visual runs must emit the graph-level metrics at
construction time.

## 4. Training dynamics

The repository now provides a two-line change-point diagnostic for topology
trajectories. A breakpoint is only a *transition candidate*. Phase-transition
language additionally requires multiple model sizes, replicated seeds,
threshold robustness, and finite-size scaling. The current project does not
claim that evidence yet.

## 5. Scientific consequence

The defensible upgraded question is:

> How do control bottlenecks organize functional collision edges into
> independent components, redundant cycles, hidden displacement dimensions,
> and maximum-matchable attack capacity, and which parts of that structure can
> a quantum collision algorithm recover more efficiently?

This is stronger than counting attacks, but narrower and more testable than a
blanket claim that every neural representation has a new global topology.
