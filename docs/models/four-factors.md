---
last_updated: "2026-09-23"
---

# Four Factors RAPM

This is a descriptive player-attribution layer built from Dean Oliver's Four
Factors. It does not change NAIL-RAPM, the public rankings, or the frozen
leaderboard. Its purpose is to separate a player's estimated association with
the mechanisms that usually determine scoring efficiency and possession value.

## Definitions

For each lineup stint and team side, the audited event mart calculates:

\[
\begin{aligned}
\mathrm{eFG\%} &= \frac{\mathrm{FGM}+0.5\,\mathrm{3PM}}{\mathrm{FGA}},\\
\mathrm{TOV\%} &= \frac{\mathrm{TOV}}{\mathrm{FGA}+0.44\,\mathrm{FTA}+\mathrm{TOV}},\\
\mathrm{ORB\%} &= \frac{\mathrm{ORB}}{\mathrm{ORB}+\mathrm{Opponent\ DRB}},\\
\mathrm{FTR} &= \frac{\mathrm{FTA}}{\mathrm{FGA}}.
\end{aligned}
\]

FTR is deliberately `FTA/FGA`, the historical Four Factors convention. The
mart is aligned to the canonical lineup-stint event boundaries and only admits
games whose play-by-play event points exactly reconcile to the final score.

## First Fit Contract

The initial fit estimates a separate **net** signed player effect for each
factor. For a factor rate \(r\), the target is \(r_H-r_A\). The player design
has `+1` for the home unit and `-1` for the away unit. Turnovers are presented
as **turnover avoidance**, so its fitted target is \(-\mathrm{TOV\%}_H -
(-\mathrm{TOV\%}_A)\); larger values are favorable for every displayed
component.

Each row is weighted by the harmonic opportunity of the two sides:

\[
w=\frac{n_H n_A}{n_H+n_A}.
\]

This is factor-specific: \(n\) is FGA for eFG% and FTR, usage opportunities
for TOV%, and offensive-rebound opportunities for ORB%. The weight reflects
the precision of a difference between two rates, rather than treating generic
possessions as the denominator for every mechanism.

Home-court advantage is the unpenalized intercept, and an unpenalized
home-minus-away back-to-back control is fitted jointly. Player coefficients
receive a sample-size-normalized ridge penalty. Lambda is selected separately
for each factor through the existing expanding chronological game folds; no
NAIL lambda, player prior, or context coefficient is reused.

Effects are published in percentage points of the factor rate. A positive
eFG%, turnover-avoidance, ORB%, or FTR effect is favorable. This stage is
completed-season descriptive attribution. Separate offensive and defensive
factor components, stability charts, and any test as a NAIL prior remain
pending Issue #18.

## First Completed-Season Fit

The initial 2025-26 fit selected independent ridge values of `0.03` for eFG%,
turnover avoidance, and FTR, and `0.01` for ORB%. All are interior selections
from the registered `0` through `10` screening grid. The early leaders are
directionally sensible: Victor Wembanyama led the eFG component (+2.99 pp),
Shai Gilgeous-Alexander led turnover avoidance (+1.46 pp), Donovan Clingan led
ORB% (+7.48 pp), and Nikola Jokić led FTR (+4.13 pp). These are descriptive
completed-season estimates, not claims that a player will repeat the result or
that the four components should be summed into total NAIL.

The immutable 2025-26 artifact is
`artifacts/models/analysis/four_factor_rapm/2025-26/four-factor-rapm-2025-26-20260923T144522Z-0f16fedd`.
The identical historical pass through 2024-25 is
`artifacts/models/analysis/four_factor_rapm_history/four-factor-rapm-history-1996-97-to-2024-25-20260923T145204Z-18a98d54`.

## Audited Source

The current mart covers 1996-97 through 2025-26. Of 34,610 regular-season
games, 34,508 (99.705%) reconcile exactly and are eligible. The remaining 102
games are explicitly excluded rather than assigning their missing or
nonreconciling events to player coefficients.
