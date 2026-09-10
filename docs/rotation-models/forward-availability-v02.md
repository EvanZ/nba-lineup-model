---
last_updated: "2026-09-09"
---

# Forward Availability v0.2

**Forward Availability v0.2** corrects the first-state overconfidence found in
[v0.1](forward-availability-v01.md). It remains a separate availability model,
not yet an input to the minutes or win-projection pipeline.

## Corrected State Contract

The availability panel contains every known rostered player-game, including
players who never record an NBA minute. Age comes first from the player-season
panel, then from a locally cached roster birth date. A remaining row without an
age is retained and receives the pooled availability baseline rather than being
dropped from the fit.

For a player's first observed season, v0.1 used the raw observed share as the
state. A `0 / 82` rookie season therefore became nearly certain evidence of
zero future availability. v0.2 instead starts from the age baseline with a
tuned pseudo-game strength \(q_{\mathrm{init}}\):

\[
\widehat p^{\mathrm{post}}_{i,t} =
\frac{q_{\mathrm{init}} p^{\mathrm{age}}_{i,t} + A_{i,t}}
{q_{\mathrm{init}} + N_{i,t}}.
\]

This applies before the player-specific state is carried into the next season.
Later observations retain their separately tuned update strength
\(q_{\mathrm{update}}\).

## Frozen Replay

The corrected panel covers 2015-16 through 2025-26. Hyperparameters were
selected from the completed 2020-21 through 2022-23 targets, and the model was
then evaluated on frozen 2023-24 through 2025-26. The selected configuration
was:

| Parameter | Value |
| --- | ---: |
| State persistence \(\rho\) | 0.50 |
| Later-state update strength \(q_{\mathrm{update}}\) | 60 pseudo-games |
| Initial-state strength \(q_{\mathrm{init}}\) | 15 pseudo-games |
| Lagged workload weight | -0.25 |

| Frozen metric, mean of three seasons | Age-only control | Availability v0.2 |
| --- | ---: | ---: |
| Exposure-weighted Brier score | 0.06133 | **0.05816** |
| Binomial log loss | 0.56145 | **0.55168** |
| Player-share MAE | 0.19984 | **0.19145** |
| Player-share RMSE | 0.25888 | **0.25338** |

v0.2 wins every aggregate frozen metric. It improves probability calibration
without relying on games played as the target.

Artifact:
`artifacts/rotation/forward_availability/forward-availability-20260909T233439Z-e341ed5`.

## Returner Check

Chet Holmgren's missed 2022-23 rookie season is a direct regression check.
The raw panel records `0 / 82` availability for that season. v0.1 predicted
only 0.3% availability for 2023-24; v0.2 predicts 47.3% before his return and
the realized value was 100.0%. The estimate remains deliberately conservative,
but it is no longer locked near zero.

Thomas Sorber is also retained despite zero NBA minutes in 2025-26: the panel
records `1 / 82` availability at age 20 from the roster birth date. He is no
longer silently omitted from the model family.

## Integration Boundary

This is the first availability model that clears its age-only benchmark. The
next product decision is integration, not a replacement of the minutes model:

\[
E[\mathrm{minutes}_{i,t}] =
P(\mathrm{available}_{i,t})
\times E[\mathrm{minutes}_{i,t} \mid \mathrm{available}_{i,t},\ \mathrm{team}].
\]

The probability is player-specific. Conditional minutes remain a joint team
allocation, so availability should be wired in only with an explicit
reallocation contract.
