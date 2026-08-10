---
last_updated: "2026-08-09"
---

# Forward Bounded Hierarchical P-spline Contextual RAPM

This forward contextual RAPM candidate retains the original relative context
function, \(C(A,B)=g(x(A)-x(B))\), without portable-unit or matchup
decomposition. For every completed season it clips each relative feature to
its possession-weighted 5th--95th percentile interval before fitting and
prediction. It then applies Ridge level shrinkage, a second-difference
P-spline penalty, and a projected completed-prior-season function prior.

The completed 2024-25 bounded state is the only contextual information used
for the frozen 2025-26 forecast. The 2025-26 state is retained only for a
future forecast.

## Frozen Result

The model records a regular eligible-game margin RMSE of **14.5485**, a
full-game margin RMSE of **14.8706**, and a Pythagorean-win RMSE of
**9.2772**. The full comparison is maintained in the [Frozen Preseason
Leaderboard](preseason-leaderboard.md).

Artifact: `forward-bounded-hierarchical-pspline-contextual-rapm-2025-26-20260809T152421Z-658f4071`.
