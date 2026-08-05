# Box-Score RAPM Prior

## First Forecast: Returning Players

The first forecast is a possession-weighted ridge regression of next-season
canonical RAPM on a player's immediately preceding RAPM, possession-native box
profile, stabilized shooting rates, and preseason age, experience, draft, and
physical fields. It only includes players with a complete prior NBA season.

For player \(i\) entering season \(t\), the model forecasts

\[
\widehat{R}_{i,t} = \beta_0 + \beta^\top x_{i,t-1},
\]

where \(R_{i,t}\) is canonical RAPM fitted from the target season and
\(x_{i,t-1}\) contains only information known after season \(t-1\). Ridge
regularization is selected by expanding target-season validation folds. Each
row is weighted by target-season reconstructed on-court possessions, so the
selection criterion is possession-weighted mean squared error.

The required comparator is **persistence**:

\[
\widehat{R}^{\mathrm{persist}}_{i,t} = R_{i,t-1}.
\]

This is deliberately a player-season evaluation, not a replacement for the
locked lineup/stint Leaderboard evaluation. The forecast cannot yet become an
RAPM prior because it has no cold-start component.

## 2025-26 Returning-Player Holdout

The model was selected using 27 expanding validation folds from 1998-99 through
2024-25 and then fitted through 2024-25. The 2025-26 target outcomes remained
untouched until the final evaluation. The selected normalized ridge
regularization was \(0.1\), equivalent to scikit-learn `alpha = 0.1 \times n`.

| Cohort | Model | Players | Possession-weighted RMSE | Skill vs. persistence |
| --- | --- | ---: | ---: | ---: |
| All returning | Persistence | 462 | 2.049 | 0.0% |
| All returning | Box-score forecast | 462 | **1.670** | **33.6%** |
| Low exposure | Persistence | 77 | 1.442 | 0.0% |
| Low exposure | Box-score forecast | 77 | **1.307** | **17.8%** |
| Developing | Persistence | 109 | 1.776 | 0.0% |
| Developing | Box-score forecast | 109 | **1.594** | **19.5%** |
| Established | Persistence | 276 | 2.147 | 0.0% |
| Established | Box-score forecast | 276 | **1.713** | **36.3%** |

The 120 no-prior players are intentionally excluded from this result. This is
not a weakness hidden by the aggregate: the run manifest records the exclusion
and the later cold-start model must be evaluated separately.

The immutable run is
`artifacts/models/box_score_prior/2025-26/box-score-prior-2025-26-20260804T212413Z-7415d704/`.
It contains fold metrics, candidate summary, out-of-fold predictions, holdout
predictions, coefficients, serialized pipeline, hashes, and an MLflow-linked
manifest.

The focused model test perturbs the entire target holdout by a large constant
and verifies that the selected regularization and fitted coefficients do not
change. It also verifies that cold starts cannot enter this returning-player
run. This guards the temporal boundary independently of the published result.

## Next Component

A cold-start forecast will use only preseason profile fields for players with
no prior NBA season. A later exposure-based blend can then transition smoothly
between cold starts and the returning-player forecast, and only that complete
prior will be eligible for the locked 2025-26 regular-season and playoff
lineup evaluations.

## Cold-Start Result

The first profile-only cold-start model was fitted and evaluated on the 120
2025-26 no-prior players. Its selected normalized ridge regularization was
\(1.0\). It was better than zero RAPM but did **not** beat the forward,
possession-weighted training mean on the primary objective:

| Model | Possession-weighted RMSE | Skill vs. zero |
| --- | ---: | ---: |
| Zero RAPM | 1.712 | 0.0% |
| Forward training mean | **1.634** | **8.9%** |
| Preseason profile ridge | 1.647 | 7.4% |

Accordingly, the cold-start profile is not blended into the prior. The current
best cold-start default remains the forward training mean. Improving this
component will require more informative preseason data or a hierarchical model
rather than forcing static biography fields to add signal they do not have.

## Complete-Prior Ablation

For the first complete-prior test, the frozen 2025-26 components were joined
with a hard switch: 462 returning players used the box-score forecast and 120
cold starts used the preseason profile forecast. That complete table then
entered the unchanged prior-centered RAPM training and evaluation procedure.

The result did not improve the locked regular-season lineup target over the
forward-lagged RAPM prior, despite the strong returning-player forecast:

| Prior | Stint RMSE | Game-margin RMSE | Skill vs. mean |
| --- | ---: | ---: | ---: |
| Forward-lagged RAPM | **103.775** | **15.235** | **1.56%** |
| Combined box-score/cold-start | 103.825 | 15.327 | 1.47% |

It is therefore retained as a reproducible ablation, not added to the
Leaderboard. This points to a mismatch between player-season canonical RAPM
forecasting and the downstream held-out lineup target, and motivates tuning
the prior scale or blend weight using lineup-level chronological folds rather
than replacing the lagged prior wholesale.
