# Persistent collision basins and restricted quantum topology sketches

This layer closes two gaps without claiming an arbitrary graph-reconstruction
speedup.

## 1. Truncated collision-basin persistence

For fixed attack candidates, payload threshold `Delta`, behaviour threshold
`Gamma`, and control-edge weight `d_c(a,b)`, define the nested graph filtration

\[
G_\epsilon = \{(a,b): d_c(a,b)\le \epsilon,
 d_p(a,b)\ge\Delta,
 d_g(a,b)\ge\Gamma\}.
\]

The implementation tracks zero-dimensional edge-bearing collision basins. A
basin is born when the first collision edge activates two previously unseen
endpoints. When two basins merge, the younger basin dies under the elder rule.
Basins surviving the preregistered threshold window are truncated at its upper
endpoint; they are not assigned an artificial finite global death time.

The reported quantities include total and normalized truncated persistence,
persistence entropy, essential basin count, half-window basin count, and
threshold-integrated capacity, basin density, cycle density, and component
entropy. This eliminates conclusions based on one favourable `epsilon`.

This is a specific H0 persistence construction on the conditional-collision
complex. It is not a claim that persistent homology of neural networks is new;
prior work has already used persistent homology to study network structure,
representations, training, and generalization.

## 2. Capacity factorization

If the active collision graph has edge-bearing components with matching sizes
`k_1,...,k_m`, then

\[
K_C=\sum_{j=1}^m k_j,
\qquad
\frac{K_C}{N}
=
\frac{m}{N}\frac{K_C}{m}
=B_C M_C.
\]

`B_C` is collision-basin density and `M_C` is within-basin matching
multiplicity. If every basin satisfies `k_j <= mu`, basin occupancy alone gives

\[
B_C\le \frac{K_C}{N}\le \mu B_C.
\]

For unit-multiplicity basins the occupancy sketch is exactly the capacity.

## 3. Prefix-addressable quantum basin sketch

The quantum statement requires a public basin partition, such as a control
signature prefix. It does **not** assume that arbitrary connected components
are known before evaluating collision edges.

Let basin `j` have a coherent occupied-basin decision/representative subroutine
of cost `H_j`, and let `m` of `L` basins be occupied. Variable-time search gives
one representative at analytic scale

\[
\widetilde O\!\left(
\sqrt{\frac{\sum_j H_j^2}{m}}
\right).
\]

Repeating with coherent exclusion gives the explicit all-representative upper
profile

\[
\sum_{r=1}^{m}
\sqrt{\frac{\sum_j H_j^2}{r}}
\le
2\sqrt{m\sum_j H_j^2}.
\]

For equal decision cost `H`, standard quantum enumeration has scale

\[
\Theta(H\sqrt{mL})
\]

up to logarithmic/error-management factors. The new project-specific content
is the connection between this enumeration problem and the empirically
observed near-unit-multiplicity collision basins, not Grover search itself.

## 4. Claim boundary

Supported by this layer:

- a nested conditional-collision filtration;
- truncated H0 collision-basin persistence;
- exact basin-density times multiplicity capacity factorization;
- a restricted prefix-addressable quantum representative-sketch profile.

Not claimed:

- novelty of persistent homology as a neural-network analysis tool;
- public access to arbitrary connected-component labels;
- a tight heterogeneous lower bound for all basin costs;
- reconstruction of every collision edge;
- coherent circuit, QRAM, or hardware runtime advantage.
