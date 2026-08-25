# State-Precision NAIL: No Forgetting

*Last updated: 2026-08-24*

This candidate tests a narrow question: does NAIL improve when its existing
forward player prior is regularized according to each player's estimated
uncertainty?

It is the first non-uniform State-Precision NAIL candidate. It received the
same strict three-season frozen evaluation as production NAIL and is not
promoted.

## Contract

The player-prior mean remains the current NAIL v1.2.1.2 forward prior. Only
the ridge precision changes by player.

After season \(t\), the RAPM fit supplies diagonal posterior variance
\(P^+_{i,t}\). For the first candidate, no offseason variance is added:

\[
P^-_{i,t+1}=P^+_{i,t}.
\]

The next fit rescales each player's prior penalty relative to the active
season's median variance:

\[
r_{i,t+1}=\frac{\operatorname{median}_j(P^-_{j,t+1})}{P^-_{i,t+1}}.
\]

Higher-confidence estimates receive \(r_i>1\), while uncertain estimates
receive \(r_i<1\). The published season lambda remains the global penalty
scale.

This differs from production NAIL, which uses \(r_i=1\) for every player.
It also differs from a decay experiment: this candidate is deliberately
equivalent to \(\rho=1\), so it isolates uncertainty-aware regularization
without offseason forgetting.

## Promotion Gate

Run the existing three-season frozen evaluation. Compare it with NAIL v1.2.1.2
using the bootstrap confidence intervals already used for promotion decisions.
Only if it clears that gate should we tune \(\rho<1\).

## Frozen Result

The completed through-2025-26 artifact was replayed directly for 2023-24,
2024-25, and 2025-26. The replay uses each stored target prior vector and its
matching prior-season context and schedule model; it does not retrain any
seasons.

| Cohort | Model | Poss. RMSE | Eligible game RMSE | Full-game RMSE |
| --- | --- | ---: | ---: | ---: |
| Regular | v1.2.1.2 production | **1.197946** | **14.0107** | **14.2330** |
| Regular | State-Precision, no forgetting | 1.197970 | 14.0255 | 14.2783 |
| Playoffs | v1.2.1.2 production | 1.192710 | 16.6032 | -- |
| Playoffs | State-Precision, no forgetting | **1.192641** | **16.5191** | -- |

The small playoff improvement does not offset weaker regular-season possession,
game, and team-level metrics. The candidate remains an evaluated, non-promoted
row on the [Three-Season Frozen Leaderboard](three-season-frozen-backtest.md).

Artifact:
`artifacts/models/nail_state_precision_no_forgetting_frozen_backtest/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260825T043905Z-da55e4b8`.
