---
last_updated: "2026-08-08"
---

# Student-t Talent-Prior Contextual RAPM

This controlled ablation combines the current Forward Contextual RAPM state
with the Student-t talent prior. It keeps Gaussian stint errors, the recursive
exposure-gated cold-start policy, the published per-season lambda schedule, and
the lagged contextual spline function unchanged.

## Seasonal Transition

For season \(t\), the completed context function \(g_{t-1}\) is subtracted
from the raw stint target before fitting player coefficients:

\[
y^{\mathrm{adj}}_{s,t}=y_{s,t}-g_{t-1}(z_{s,t}).
\]

The player adjustment around the forward prior \(\mu_{i,t}\) receives a
Student-t penalty instead of the Gaussian ridge penalty used by the base
contextual model:

\[
\beta_{i,t}-\mu_{i,t}
\sim \operatorname{StudentT}(\nu=3,0,s=3).
\]

IRLS turns that into a player-specific local shrinkage multiplier. The fitted
additive state then defines the raw residual used to fit \(g_t\), which is
carried into season \(t+1\). Thus the model changes only the player-prior
departure distribution; it does not alter cold starts, context features, or
the information boundary.

## Frozen 2025-26 Result

The run
`artifacts/models/student_t_talent_contextual_rapm/2025-26/student-t-talent-contextual-rapm-2025-26-20260808T012017Z-7bbeadf3/`
fits 1996-97 through 2025-26. Its frozen 2025-26 forecast uses the completed
2024-25 player state and \(g_{2024-25}\), then scores realized target-season
lineup exposure without a player refit.

| Cohort | Possession RMSE | Possession MAE | Eligible-game RMSE | Team NetRtg RMSE | Pythagorean win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular season | 1.199026 | **1.141556** | 14.6770 | 4.1771 | 9.3594 |
| Playoffs | 1.192959 | 1.135643 | 17.6146 | - | - |

It narrowly improves the base contextual model's regular-season possession MAE
(1.141571 to 1.141556), but is worse on its primary full-game margin metric
(15.0190 to 15.0487), team NetRtg RMSE, and Pythagorean win RMSE. It is
therefore retained as a transparent combination test, not promoted over
[Forward Contextual RAPM](forward-contextual-rapm.md).

Run it with [Train Student-t Talent-Prior Contextual RAPM](../guides/train-student-t-talent-contextual-rapm.md).

## Outputs

The immutable artifact includes the standard frozen possession, eligible-game,
full regular-game, team NetRtg, and Pythagorean-win outputs. It also retains
the per-season player priors, exposure-gated replacement tokens, cold-start
metadata, contextual models, and contextual fit metadata needed to reproduce
the state transition.
