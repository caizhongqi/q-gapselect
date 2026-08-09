# Stage 10 — Real MNIST Response Codes in LPC / VT-FCW

## Scope

The attack predicate remains the locked external hard-label functional collision

\[
x\neq x',\quad y(x)\neq y(x'),\quad F(x)=F(x'),
\]

with the frozen F-v1 32-bit response code. This stage replaces the Stage-8 density-matched synthetic surrogate with the per-sample MNIST response matrices saved in Stage 9.

All quantum numbers remain theorem-normalized component-oracle proxies. They are **not** end-to-end wall-clock quantum speedups and require coherent component-oracle access.

## E44 — Real-code controlled one-claw

Each candidate domain is built entirely from real MNIST F-v1 codes. One real cross-semantic collision is retained and all other cross-domain exact code collisions are removed so that the marked-fraction formula is exactly aligned with the one-claw analysis:

\[
\varepsilon=(r/N)^2,\qquad \delta=\frac{N}{r(N-r)}.
\]

The classical comparator is stronger than the Stage-8 full-certificate baseline: it performs mixed-prefix refinement and stops as soon as the first exact claw is reached, visiting the revealed child with larger cross-side potential first.

Pilot over all eligible victim/seed/class-pair cells:

| N/side | strong classical early-stop | VT-FCW k=1 | VT-FCW k=2 | VT-FCW k=4 | median C/Q k=1 | median C/Q k=2 | median break-even k |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 478.6 | 490.4 | 831.9 | 1514.9 | 0.93 | 0.55 | 0.90 |
| 64 | 922.1 | 844.2 | 1440.0 | 2631.5 | 1.07 | 0.63 | 1.10 |
| 128 | 1941.6 | 1403.7 | 2397.2 | 4384.3 | 1.31 | 0.76 | 1.43 |
| 256 | 3912.4 | 2459.0 | 4216.3 | 7730.9 | 1.48 | 0.85 | 1.64 |

Finite-range fits for N>=32:

- strong classical early-stop: exponent 1.017;
- VT-FCW k=1: exponent 0.771;
- VT-FCW k=2: exponent 0.776;
- VT-FCW k=4: exponent 0.779.

At N=256, k=1 wins in 84% of the real one-claw cells, k=2 in 42%, and k=4 in 0%. Therefore the old Stage-8 statement that a fourfold cleanup envelope survives is **not** supported in the real one-claw regime.

## E45 — Real natural multi-claw, collision-bearing domains

A second experiment retains all additional real code collisions after seeding a candidate domain with one real collision endpoint pair. This is a collision-bearing natural-density regime, not an unconditional random-domain experiment.

After extending from the original Top-3 pilot to **all eligible collision-bearing class pairs**, the aggregate is:

| N/side | median natural claws | median empirical marked fraction | strong classical | VT-FCW k=1 | VT-FCW k=2 | median C/Q k=1 | median C/Q k=2 | median break-even k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.0 | 0.121 | 491.8 | 444.2 | 729.9 | 1.12 | 0.67 | 1.19 |
| 64 | 2.0 | 0.121 | 967.9 | 706.5 | 1145.2 | 1.37 | 0.85 | 1.59 |
| 128 | 5.0 | 0.156 | 1794.6 | 1042.6 | 1636.7 | 1.68 | 1.13 | 2.42 |
| 256 | 31.5 | 0.399 | 3244.3 | 1466.8 | 2110.8 | 2.14 | 1.50 | 3.70 |

At N=256, k=1 wins in 98.1% of cells, k=2 in 88.5%, and k=4 in 44.2%. Thus real multi-solution density materially improves the walk marked fraction, but k=4 is still not a robust real-data main claim.

The finite-range empirical slopes in this changing-density regime are approximately 0.906 for classical early-stop, 0.573 for VT-FCW k=1, and 0.511 for k=2. These are diagnostics only; the marked density itself grows with N, so they are not asymptotic one-claw exponents.

## E46 — Unconditioned random-domain hit rate

To separate cost conditional on a solution from solution prevalence, random N-by-N domains were sampled without forcing a collision endpoint. The following averages/medians are over **all class pairs that contain at least one collision in the full valid F-v1 pool and have enough samples at the requested N**:

| N/side | mean hit probability | median hit probability | evaluated cells |
|---:|---:|---:|---:|
| 32 | 0.326 | 0.188 | 184 |
| 64 | 0.496 | 0.445 | 152 |
| 128 | 0.747 | 0.878 | 105 |
| 256 | 0.944 | 1.000 | 52 |

At N=256, 98.1% of evaluated class-pair cells have hit probability at least 0.5 and 80.8% have hit probability at least 0.9. This does **not** imply that every one of the 45 semantic class pairs is vulnerable; the population here is explicitly restricted to class pairs with at least one full-pool exact collision.

## E47/E49 — Probe-order robustness and calibration stress

LPC cost depends on the order of the 32 response bits. On the same real one-claw domains, 12 random bit permutations were evaluated using identical subset/edge draws.

- N=128: median k=1 advantage across random orders 1.43; mean fraction of random orders that still win 82.7%.
- N=256: median k=1 advantage across random orders 1.66; mean fraction of random orders that still win 95.5%.
- Taking the least favorable common order among the fixed order plus 12 random permutations gives a median k=1 advantage about 1.02 at N=256.

A stronger calibration stress ranks bits by full-pool class-side marginal separation

\[
|P(b_j=1\mid A)-P(b_j=1\mid B)|.
\]

If that calibrated order is given to classical search for free while quantum is artificially kept on the original order, median k=1 advantage becomes 0.89 at N=128 and 1.08 at N=256. If both methods use the same calibrated order, median k=1 advantage is 1.37 at N=128 and 1.74 at N=256, with win rates 88.6% and 96.0% respectively.

Therefore probe-order calibration must be explicit in the resource model; it cannot be silently granted to one side.

## Main-line decision after Stage 10

The real-response-code rerun resolves the major Stage-7/8 surrogate gap.

The strongest defensible computational statement is now:

1. exact external functional collisions occur across multiple MNIST victim families and seeds;
2. collision pairs generally retain above-matched-random relational agreement on unseen probes, although abundance is architecture/seed dependent;
3. on real response geometry, controlled one-claw experiments retain a finite-range classical-vs-VT-FCW exponent separation;
4. the constant-factor margin is substantially tighter than on the old surrogate;
5. k=1 is the robust real-data regime; k=2 becomes robust mainly when natural multi-claw marked mass is high; k=4 is not a robust real-data result;
6. a normal classical remote API still does not instantiate the coherent component oracle needed by the quantum query model.

The next main-line gate is **explicit reversible LPC resource compilation**: logical gate/ancilla counts for the canonical sparse certificate and DELTA-STABILIZE transition, followed by a branch-clock/phase-detection implementation check. Query-level k=1 cannot be treated as physical overhead until this gate is passed.
