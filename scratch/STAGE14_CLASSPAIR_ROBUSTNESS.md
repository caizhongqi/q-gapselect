# Stage 14 addendum — Semantic-pair robustness

The Stage-14 secret-challenge impersonation result was additionally decomposed by victim, training seed, and semantic class pair to test whether the aggregate effect was driven by a few large collision buckets such as 6--9 or 1--4.

Each unit is a (victim, seed, semantic pair) triple with at least 20 strict public F-v1 collisions. The matched-random comparator is sampled **inside the same semantic pair**, so the result cannot be explained by class-pair difficulty.

At the 80% hidden-challenge acceptance threshold:

- evaluable victim-seed-class-pair units: **81**;
- distinct semantic class pairs represented: **29**;
- collision secret-pass rate: **40.71%**;
- class-pair-matched random secret-pass rate: **19.77%**;
- mean uplift: **+20.94 percentage points**;
- median uplift: **+21.11 percentage points**;
- positive-uplift units: **72/81**;
- significant units: **65/81**.

Representative semantic pairs appearing in at least three victim/seed units include:

- 3--9: mean uplift +31.77 pp, 3/3 positive;
- 4--7: +30.64 pp, 5/5 positive;
- 7--9: +24.39 pp, 8/9 positive;
- 6--9: +22.48 pp, 3/3 positive;
- 3--7: +22.14 pp, 5/5 positive;
- 5--7: +21.12 pp, 5/6 positive;
- 1--4: +9.79 pp, 4/4 positive.

Therefore the Stage-14 impersonation gap is not a consequence of one dominant semantic collision bucket. The effect appears across many distinct cross-semantic relationships, although its magnitude remains pair dependent.
