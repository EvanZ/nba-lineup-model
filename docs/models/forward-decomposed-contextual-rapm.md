---
last_updated: "2026-08-08"
---

# Forward Decomposed Contextual RAPM

Forward Decomposed Contextual RAPM is an interpretability ablation of
[Forward Contextual RAPM](forward-contextual-rapm.md). It retains the same
forward player-prior, cold-start, profiles, spline basis, Ridge penalty, and
season-by-season information boundary. Its only substantive change is the
contextual function: every lineup has an identifiable side score, and a matchup
context correction is exactly the home score minus the away score.

This is not the predictive default. On the frozen 2025-26 evaluation, the
original relative-context model has better regular-season game-margin, team
net-rating, and Pythagorean-win errors. The decomposed form remains useful as a
clean diagnostic and as a potential building block for an interactive lineup
tool because it exposes a per-unit context value.

## Same Inputs, Different Constraint

Each five-player unit receives the same leakage-safe context profile used by
the original model. It includes stabilized prior-season rates for shooting,
assists, turnovers, usage, offensive and defensive rebounds, steals, and
blocks, plus the same eleven composition summaries:

- Bottom-two shooting, credible-shooter count, top-two assists, and usage concentration.
- Square-root offensive and defensive rebounding totals.
- Imputed/replacement-profile counts and weights.
- Shooting-by-usage, shooter-by-passing, and rebounding-by-usage interactions.

The original model directly fits a function of the relative lineup feature
vector, \(g_t(\phi(H)-\phi(A))\). This model first maps a unit's own profile to
a scalar score \(h_t(\phi(U))\), then requires the matchup correction to be

\[
c_t(H,A) = h_t(\phi(H)) - h_t(\phi(A)).
\]

The side model uses the same quadratic spline basis with four knots and the
same possession-weighted Ridge alpha of \(10{,}000\). The spline transformer
and scaler are fit on the pooled home and away side-feature rows. Ridge is fit
without an intercept, so antisymmetry is a structural property rather than an
approximation:

\[
c_t(A,H) = -c_t(H,A).
\]

## Rolling State Transition

For regular-season stint \(s\) in season \(t\), the completed prior-season
side function is subtracted before the same prior-centered additive RAPM fit:

\[
y^{\mathrm{adj}}_{s,t} = y_{s,t} -
\left[h_{t-1}(\phi(H_s))-h_{t-1}(\phi(A_s))\right].
\]

The model then fits possession-weighted one-number RAPM with the existing
season-specific forward exposure-gated lambda. After the season completes, it
fits \(h_t\) to the residual lineup effects. The frozen 2025-26 evaluation uses
only \(h_{2024-25}\), player state through 2024-25, and 2025-26 oracle lineups
and exposure; \(h_{2025-26}\) is retained only for the next forecast season.

## Why The Constraint Matters

The decomposed design makes a unit's context contribution portable across
opponents: the difference between two home units is independent of the away
unit. That makes side-level displays and feature attribution straightforward.
It also removes patterns that can only be expressed as a relative matchup
effect. For example, a nonlinear response to a home-minus-away rebounding gap
need not equal the difference between two independent side scores. The frozen
results indicate that this lost flexibility matters in the current feature set.

| Frozen 2025-26 metric | Relative contextual RAPM | Decomposed contextual RAPM |
| --- | ---: | ---: |
| Regular possession RMSE | **1.199008** | 1.199259 |
| Regular eligible-game margin RMSE | **14.6525** | 14.9769 |
| Regular team NetRtg RMSE | **4.1572** | 4.7470 |
| Regular Pythagorean-win RMSE | **9.3153** | 11.0387 |
| Playoff eligible-game margin RMSE | **17.5493** | 18.4522 |

The full comparison is maintained in the [Frozen Preseason
Leaderboard](preseason-leaderboard.md).

## Artifact

The completed through-2025-26 run is
`forward-decomposed-contextual-rapm-2025-26-20260808T133655Z-41f61af6` in
`artifacts/models/forward_decomposed_contextual_rapm/2025-26/`. It stores the
per-season player state, profiles, decomposed context models, context metadata,
and frozen regular-season/playoff, team, and win evaluation tables.
