---
last_updated: "2026-09-09"
---

# Forward Conditional Minutes v0.1

This is the first preseason minutes component. It predicts a player's expected
NBA minutes **per medically available game**, before any team-level rotation
allocation is imposed.

## Forward Player State

For a player-season with \(A_{i,t}\) medically available games and total NBA
minutes \(M_{i,t}\), the observed conditional target is:

\[
m_{i,t}^{\mathrm{avail}} = \frac{M_{i,t}}{A_{i,t}}.
\]

The model fits a pooled quadratic age baseline on
\(\log(1+m_{i,t}^{\mathrm{avail}})\), then carries an exposure-shrunk player
state forward. Available games are measurement exposure: a 10-game observation
updates the state less than a 70-game observation. First observed seasons
shrink to the age baseline; gap years decay the player-specific deviation back
toward that baseline.

\[
z_{i,t}^{\mathrm{post}} =
\frac{q z_{i,t}^{\mathrm{prior}} + A_{i,t}\log(1+m_{i,t}^{\mathrm{avail}})}
{q + A_{i,t}}.
\]

There are no target-season box-score, minutes, or availability outcomes in the
forward prediction. A cold start receives only the age baseline.

## Combine With Availability

Forward Availability v0.2 supplies \(p_{i,t}^{\mathrm{avail}}\). The two
independent components first form raw expected totals:

\[
r_{i,t}=82\,p_{i,t}^{\mathrm{avail}}\,
\widehat m_{i,t}^{\mathrm{avail}}.
\]

## Roster Normalization

Raw player totals do not necessarily sum to a team's 19,680 regulation
minutes. For each opening roster, the 15 highest raw totals form the rotation,
and the model normalizes those totals as:

\[
w_{i,t}=\operatorname{softmax}(\log r_{i,t})
=\frac{r_{i,t}}{\sum_{j\in R_t}r_{j,t}},
\qquad
\widehat M_{i,t}=19{,}680\,w_{i,t}.
\]

Zero raw weights remain zero. This makes roster minutes conserve exactly while
automatically reallocating a manually unavailable player's minutes to
teammates.

## Planning Controls

The Win Projections page exposes both forward inputs for every player:

- \(+/-\), the default NAIL-RAPM projection, which users may override for a
  scenario;
- \(G_{\mathrm{available}}\), projected medically available games out of 82;
- (E[\mathrm{MPG}\mid\mathrm{available}]), conditional playing time.

Internally, the page converts the whole-game control to
\(P(\mathrm{available}) = G_{\mathrm{available}}/82\). Users may override
any of these inputs before recalculating. A \(+/-\) override affects only the
local scenario's minute-weighted team strength; it does not alter the minutes
allocation or stored NAIL release.
The page re-ranks the full roster after each edit, then applies the same
top-15 normalization. Thus, setting a current rotation player's availability
to zero drops them from the rotation and promotes the next-highest raw
projection; a player outside the rotation enters only by rising into the top
15. Every change reallocates the fixed 240 regulation minutes per game across
that team rather than adding or removing minutes from the league schedule.

## Evaluation

Hyperparameters are selected from completed 2020-21 through 2022-23 seasons,
then tested on frozen 2023-24 through 2025-26 seasons. The replay reports
conditional-minutes error, combined raw expected-total-minutes error, and an
exact team-minute-conservation audit for every historical opening roster.

The selected player-state configuration is full one-season persistence
\(\rho=1.0\), a 15 available-game later-update strength, and a 30 available-game
initial-state strength.

| Frozen metric, mean of three seasons | Age-only control | Forward Conditional Minutes v0.1 |
| --- | ---: | ---: |
| Conditional minutes MAE | 9.85 | **6.14** |
| Conditional minutes RMSE | 11.25 | **8.04** |
| Available-game-weighted conditional MAE | 9.62 | **5.45** |
| Available-game-weighted conditional RMSE | 11.09 | **7.11** |
| Raw expected-total-minutes MAE | 695.1 | **505.9** |
| Raw expected-total-minutes RMSE | 806.5 | **632.4** |

The control uses the same Forward Availability v0.2 probability but replaces
the player state with the age baseline. The state model wins every aggregate
metric. Every one of the 90 frozen opening-roster team rows sums to exactly
19,680 regulation minutes after normalization.

Artifact:
`artifacts/rotation/forward_conditional_minutes/forward-conditional-minutes-20260910T020256Z-31d03d6`.

This is a prerequisite for the later game-level joint availability and rotation
model in Issue #7.

## Win-Loss Envelope

The Win Projections page also shows a browser-side Win-Loss Envelope above the
player controls. It simulates all 82 games for the selected team 10,000 times
using the same minute-weighted team strengths, home-court term, back-to-back
term, and logistic game probabilities as the win-total table. The chart shows
both the 1st--99th and 5th--95th percentile intervals around cumulative wins.

The 80 published games retain their actual opponents, dates, home/away status,
and back-to-back flags. The two NBA Cup-dependent regular-season slots are
modeled as one home and one away game against a neutral-strength opponent, to
match the win-total table's existing assumption. They are inserted in the Dec.
4--10 Cup window and labeled `NBA CUP` until the actual matchup details are
known.

The chart can be downloaded as a self-contained PNG. The export adds the NBA
GESTALT title, season and simulation subtitle, percentile legend, team
watermark, and the opponent-logo annotations for the five toughest modeled
games.
