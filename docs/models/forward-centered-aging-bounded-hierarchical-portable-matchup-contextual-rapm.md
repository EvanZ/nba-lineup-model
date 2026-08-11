---
last_updated: "2026-08-09"
---

# Forward Centered Aging Bounded Hierarchical Portable-Matchup Contextual RAPM

This candidate tests whether an explicit, leakage-safe player-rating reference
improves the recursive age-informed portable contextual state. It is identical
to the aging bounded portable-matchup model except that every season's complete
player prior vector is centered at its prior-season possession-weighted mean.

Because every RAPM row contains five positive and five negative player
indicators, adding a common constant to every coefficient leaves lineup
predictions unchanged. The centering rule therefore turns an otherwise
arbitrary coefficient location into a documented pre-season reference:

\[
\widetilde{\mu}_{i,t} = \mu_{i,t} -
\frac{\sum_j p_{j,t-1}\mu_{j,t}}{\sum_j p_{j,t-1}}.
\]

The centered prior enters the same age-informed returner branch and
exposure-gated draft/replacement cold-start branch as the uncentered model.
Its contextual term remains:

\[
C(A,B)=h(x(A))-h(x(B))+q(x(A),x(B)).
\]

## Evaluation Status

The full through-2025-26 recursive evaluation is in progress. Its frozen
regular-season, playoff, full-game, and team-level metrics will be compared
with the uncentered aging candidate before this model is promoted.

Use [the training guide](../guides/train-forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md)
to reproduce the candidate.
