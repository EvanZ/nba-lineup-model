---
last_updated: "2026-08-27"
---

# Withdrawn Split NAIL v0.1 Prototype

> This prototype is withdrawn. It added O/D specialization to scalar NAIL
> predictions while reusing scalar context and schedule terms, so it was not a
> complete O/D decomposition. It is retained only as a record of the rejected
> shortcut and is excluded from the leaderboard and model tree.

Split NAIL v0.1 is a constrained offense/defense extension of production
[NAIL-RAPM v1.2.1.2](nail-rapm-v1212-back-to-back.md). It fixes the total
player rating coordinate and adds a separately regularized specialization
coordinate, rather than fitting two unconstrained player totals.

## Parameterization

For player \(i\), let \(R_i\) be combined player value and \(s_i\) be
offense-versus-defense specialization:

\[
O_i = R_i + s_i, \qquad D_i = R_i - s_i.
\]

The predicted home net rating of home unit \(H\) against away unit \(A\) is:

\[
\widehat y(H,A) =
\sum_{i\in H} R_i - \sum_{j\in A} R_j
+ \sum_{i\in H} s_i + \sum_{j\in A} s_j
+ C_{t-1}(H,A) + B_{t-1}(H,A) + \alpha.
\]

\(C\) is the existing frozen scalar NAIL non-additive context correction and
\(B\) is its back-to-back schedule adjustment. The player state is the only
new component.

This makes scalar NAIL a nested case: setting every \(s_i=0\) recovers its
signed player design exactly. The displayed combined rating is \(R_i\), not
\(O_i+D_i\).

## Priors And Regularization

For completed season \(t\), \(R_i\) is centered on the exact scalar NAIL
v1.2.1.2 prior available before that season. Specialization is carried forward
from \(t-1\), with new players centered at zero. v0.1 uses a specialization
relative precision of 4.0: specialization deviations are penalized four times
as strongly as combined-rating deviations under the season's published player
lambda.

This avoids the old decomposition's effective de-regularization: two
independently penalized O/D coefficients can change their sum at only half the
scalar ridge cost.

## Frozen Evaluation

The model is being evaluated on 2023-24 through 2025-26. For each target
season, it adds only the prior completed season's \(s_i\) values to the exact
production NAIL v1.2.1.2 frozen prediction. Regular-season and playoff
possessions retain the same identifiers and realized outcomes as the production
artifact; full-game, team net-rating, and Pythagorean-win metrics are replayed
from that shared support.

## Three-Season Result

| Pooled metric | Split NAIL v0.1 | Production NAIL v1.2.1.2 |
| --- | ---: | ---: |
| Regular possession RMSE | 1.198015 | **1.197946** |
| Regular possession MAE | **1.141263** | 1.141313 |
| Eligible game RMSE | 14.1856 | **14.0107** |
| Full-game RMSE | 14.2995 | **14.2330** |
| Winner accuracy | 67.90% | **67.96%** |
| Team NetRtg RMSE | **3.2782** | 3.2847 |
| Pythagorean-win RMSE | **7.0471** | 7.0551 |
| Playoff possession RMSE | 1.192736 | **1.192710** |
| Playoff possession MAE | **1.137265** | 1.137604 |

v0.1 is retained as a non-promoted experiment. Its side specialization has
some useful aggregate signal, but it does not clear the primary regular-season
possession and game-margin gates against production NAIL.
