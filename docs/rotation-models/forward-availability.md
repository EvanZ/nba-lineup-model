---
last_updated: "2026-09-10"
---

# Forward Availability

Forward Availability forecasts a player's **medical availability share** for a
future season. It is intentionally separate from playing time: a player can be
medically available but receive no minutes because they are outside the
rotation.

The target is available games divided by known rostered player-games, not games
played. Its training panel and binary label are defined in the
[player-game availability contract](../data/player-availability.md).

## Current Contract

For target season \(t\), the model uses only completed earlier seasons. An
exposure-weighted quadratic age curve provides the pooled baseline. Each
player's deviation from that baseline is carried forward with persistence
\(\rho\), a gap adjustment, and a small standardized lagged-workload term:

\[
\operatorname{logit}(p_{i,t}) = a(\operatorname{age}_{i,t})
+ \rho^{g_{i,t}}\left(z_{i,t-1} + w\,\widetilde{m}_{i,t-1}\right).
\]

Observed availability updates the state with a beta-binomial-style
pseudo-game prior. The first observed season starts from the age baseline,
rather than treating a raw \(0/82\) observation as certainty:

\[
\widehat p^{\mathrm{post}}_{i,t} =
\frac{q\,p^{\mathrm{prior}}_{i,t} + A_{i,t}}
{q + N_{i,t}}.
\]

Here \(A\) is available games and \(N\) is known rostered player-games. The
promoted output is paired with Forward Conditional Minutes to form expected
season minutes:

\[
E[\mathrm{minutes}_{i,t}] =
P(\mathrm{available}_{i,t})
\times E[\mathrm{minutes}_{i,t}\mid\mathrm{available}_{i,t}].
\]

## Current Production State: v0.2

The promoted configuration is:

| Parameter | Value |
| --- | ---: |
| State persistence \(\rho\) | 0.50 |
| Later-state update strength \(q_{\mathrm{update}}\) | 60 pseudo-games |
| Initial-state strength \(q_{\mathrm{init}}\) | 15 pseudo-games |
| Lagged workload weight | -0.25 |

It was selected on completed 2020-21 through 2022-23 targets and evaluated on
frozen 2023-24 through 2025-26 seasons.

| Mean frozen metric | Age-only control | **v0.2** |
| --- | ---: | ---: |
| Exposure-weighted Brier score | 0.06133 | **0.05816** |
| Binomial log loss | 0.56145 | **0.55168** |
| Player-share MAE | 0.19984 | **0.19145** |
| Player-share RMSE | 0.25888 | **0.25338** |

Artifact:
`artifacts/rotation/forward_availability/forward-availability-20260909T233439Z-e341ed5`.

## Update History

### v0.1: Initial State Filter

The first release established the forward-only age curve, filtered player
state, pseudo-game update, and frozen evaluation contract. It improved player
share MAE but lost to the age-only control on probability-sensitive metrics,
which exposed overconfidence in first observed seasons. It was not promoted.

### v0.2: First-State Calibration

v0.2 changed only the first-state update: a player's first observed season now
starts from the age baseline with its own tuned pseudo-game strength. This fixes
the failure mode where a missed rookie season produced an implausibly near-zero
forecast. For example, Chet Holmgren's `0 / 82` rookie observation led v0.1 to
forecast 0.3% availability for 2023-24; v0.2 forecast 47.3% before his return,
with 100.0% realized availability.

The v0.2 result cleared every aggregate frozen metric and is the availability
component used by Win Projections.
