---
last_updated: "2026-09-12"
---

# Forward Conditional Minutes

Forward Conditional Minutes forecasts a player's regular-season NBA minutes
**per medically available game** before the team-level rotation allocation.
It is paired with Forward Availability in the preseason minutes stack.

## Current Contract

For a player-season with \(A_{i,t}\) medically available games and total NBA
minutes \(M_{i,t}\), the observed target is:

\[
m_{i,t}^{\mathrm{avail}} = \frac{M_{i,t}}{A_{i,t}}.
\]

A pooled quadratic age curve predicts \(\log(1+m_{i,t}^{\mathrm{avail}})\).
Returning players carry an exposure-shrunk deviation from that curve forward;
available games are measurement exposure, so a 10-game observation moves the
state less than a 70-game observation:

\[
z_{i,t}^{\mathrm{post}} =
\frac{q z_{i,t}^{\mathrm{prior}} + A_{i,t}\log(1+m_{i,t}^{\mathrm{avail}})}
{q + A_{i,t}}.
\]

The rookie branch adds a regularized, draft-informed residual to the age
baseline:

\[
\log(1+\widehat m_{i,t}^{\mathrm{avail}})
= b(\mathrm{age}_{i,t})
+ \gamma_0
+ \sum_{k=1}^{6}\gamma_k\,\widetilde{x}_{i,t,k}^{\mathrm{rookie}}.
\]

The six rookie inputs are draft capital, undrafted status, rookie age, and
listed guard/forward/center indicators. A non-rookie with no usable NBA
minutes state remains on the age baseline; the model does not invent a veteran
role from draft information.

### Incumbent Team-Strength Cold Start

FCM v0.3 adds one preseason feature to the rookie branch: the projected NAIL
strength of the returning rotation around the rookie. For team \(k\), let
\(I_{k,t}\) be its opening-roster players who are non-rookies with an observed
conditional-minutes state. Their strength is a raw expected-minute-weighted
average of frozen preseason NAIL forecasts:

\[
C_{k,t} =
\frac{\sum_{j\in I_{k,t}} p_{j,t} m_{j,t} R_{j,t}}
{\sum_{j\in I_{k,t}} p_{j,t} m_{j,t}}.
\]

Here \(p_{j,t}\) is Forward Availability, \(m_{j,t}\) is the incumbent's
pre-squash FCM estimate, and \(R_{j,t}\) is that player's target-season
preseason NAIL forecast. Rookies do not enter \(C_{k,t}\), so their own
prediction cannot feed back into the feature.

The cold-start equation receives a fitted linear correction:

\[
\log(1+\widehat m_{i,t}^{\mathrm{avail}})
= b(\mathrm{age}_{i,t})
+ \gamma_0
+ \sum_{k=1}^{6}\gamma_k\,\widetilde{x}_{i,t,k}^{\mathrm{rookie}}
+ \beta_C C_{\mathrm{team}(i),t}.
\]

The fitted unscaled value is \(\beta_C=-0.2071\). Holding the rookie's own
profile fixed, a stronger incumbent rotation therefore reduces the rookie's
projected conditional role; a weaker one increases it. This feature applies
only to the draft-informed rookie branch. Returning players retain the
exposure-shrunk player-state forecast above.

The two forward components form raw expected minutes:

\[
r_{i,t}=82\,P(\mathrm{available}_{i,t})
\,\widehat m_{i,t}^{\mathrm{avail}}.
\]

## Roster Squashing

The availability-times-conditional-minutes product is an independent player
forecast, so player raw totals do not normally add to the team's fixed
19,680-minute regular-season budget. The downstream Win Projections model
converts those raw totals into a feasible active-15 rotation before it builds
team strength. See [Roster Squashing In Win Projections](win-projections.md#roster-squashing).

## Current Production State: v0.3

The returner-state configuration is full one-season persistence, a 15-game
later-update strength, and a 30-game initial-state strength. The rookie ridge
penalty \(\alpha=0.01\) was selected on completed 2020-21 through 2022-23
targets. The v0.3 incumbent-strength feature uses the same tuning seasons and
was tested through the unchanged Availability, active-15, and 240-minute
roster-squash contract on frozen 2023-24 through 2025-26 seasons.

### v0.2 Component Validation

| Mean frozen metric | v0.1 | **v0.2** | Change |
| --- | ---: | ---: | ---: |
| Conditional minutes MAE | 6.14 | **5.60** | -0.53 |
| Conditional minutes RMSE | 8.03 | **7.55** | -0.48 |
| Available-game-weighted MAE | 5.44 | **5.06** | -0.38 |
| Available-game-weighted RMSE | 7.11 | **6.67** | -0.44 |
| Raw expected-total-minutes MAE | 505.9 | **455.0** | -50.9 |
| Raw expected-total-minutes RMSE | 632.4 | **583.4** | -49.0 |
| Rookie weighted RMSE | 10.09 | **7.63** | -2.45 |

v0.2 artifact:
`artifacts/rotation/forward_conditional_minutes/forward-conditional-minutes-20260910T163017Z-fe2dd1d`.

### End-To-End Roster Allocation Validation

The promotion comparison changes only the rookie FCM feature. Availability,
raw expected minutes, active-15 selection, and 240-minute normalization are
identical to the v0.2 control.

| Pooled frozen metric | FCM v0.2 control | **FCM v0.3** | Change |
| --- | ---: | ---: | ---: |
| Allocation total variation | 0.25756 | **0.25609** | -0.00147 |
| Allocation Brier score | 0.02277 | **0.02257** | -0.00020 |
| Player share MAE | 0.02268 | **0.02256** | -0.00012 |
| Player share RMSE | 0.03152 | **0.03140** | -0.00012 |
| Actual top-15 overlap | 0.81185 | 0.81185 | 0.00000 |

The paired team bootstrap favored v0.3 in 96.6% of draws for allocation total
variation and 91.8% for Brier score. Its 95% intervals narrowly include zero,
so the evidence is modest, but the candidate improves every continuous pooled
metric without changing the top-15 gate.

Artifact:
`artifacts/rotation/forward_conditional_team_strength_roster_squash/forward-conditional-team-strength-roster-squash-20260912T195229Z-0881199`.

## Update History

### v0.1: Age And Player-State Prior

The initial release established the age baseline, exposure-shrunk returner
state, availability-times-conditional-minutes combination, top-15 rotation
constraint, and exact team-minute conservation. It beat the age-only control
on every frozen aggregate metric, but a true rookie received only the pooled
age baseline.

### v0.2: Draft-Informed Rookie Prior

v0.2 keeps the returner model unchanged and adds the rookie residual above.
It improves every aggregate held-out metric and every individual held-out
season. The change is concentrated where intended, but also improves overall
allocation because a rookie's raw expected minutes are normalized jointly with
their teammates.

For current top prospects, the practical effect is substantial: AJ Dybantsa
changes from 9.8 to 21.5 conditional MPG, Darryn Peterson from 9.8 to 21.2,
and Cameron Boozer from 9.8 to 20.0.

### v0.3: Team-Strength Cold Starts

v0.3 retains the returner and draft-informed rookie contracts, then adds the
incumbent team-strength feature only to the rookie cold-start branch. It lets
the same rookie profile receive a different conditional-minute estimate when
joining a rotation with a different amount of established preseason value.
The feature is built exclusively from target-preseason forecasts and raw
expected minutes, before roster squashing.

## Win-Loss Envelope

Win Projections exposes projected available games and conditional MPG as
separate controls. It then simulates 82 games 10,000 times using the same
minute-weighted strengths, home-court term, back-to-back term, and logistic
game probabilities as the win-total table. The resulting Win-Loss Envelope
shows 1st--99th and 5th--95th percentile cumulative-win intervals, opponent
and back-to-back detail, toughest-game annotations, and a self-contained PNG
export.
