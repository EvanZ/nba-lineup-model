---
last_updated: "2026-09-10"
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

The current rookie branch adds a regularized, draft-informed residual to the
age baseline:

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

## Current Production State: v0.2

The returner-state configuration is full one-season persistence, a 15-game
later-update strength, and a 30-game initial-state strength. The rookie ridge
penalty \(\alpha=0.01\) was selected on completed 2020-21 through 2022-23
targets. The final comparison uses frozen 2023-24 through 2025-26 seasons.

| Mean frozen metric | v0.1 | **v0.2** | Change |
| --- | ---: | ---: | ---: |
| Conditional minutes MAE | 6.14 | **5.60** | -0.53 |
| Conditional minutes RMSE | 8.03 | **7.55** | -0.48 |
| Available-game-weighted MAE | 5.44 | **5.06** | -0.38 |
| Available-game-weighted RMSE | 7.11 | **6.67** | -0.44 |
| Raw expected-total-minutes MAE | 505.9 | **455.0** | -50.9 |
| Raw expected-total-minutes RMSE | 632.4 | **583.4** | -49.0 |
| Rookie weighted RMSE | 10.09 | **7.63** | -2.45 |

Artifact:
`artifacts/rotation/forward_conditional_minutes/forward-conditional-minutes-20260910T163017Z-fe2dd1d`.

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

## Win-Loss Envelope

Win Projections exposes projected available games and conditional MPG as
separate controls. It then simulates 82 games 10,000 times using the same
minute-weighted strengths, home-court term, back-to-back term, and logistic
game probabilities as the win-total table. The resulting Win-Loss Envelope
shows 1st--99th and 5th--95th percentile cumulative-win intervals, opponent
and back-to-back detail, toughest-game annotations, and a self-contained PNG
export.
