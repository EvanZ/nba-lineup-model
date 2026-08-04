# Forward RAPM Calibration

## Purpose

This experiment converts the sequential, forward lagged-prior RAPM signal into
a team regular-season win estimate. It is intentionally narrow: **prior RAPM
only**. Age, draft status, physical measurements, box-score features, and
target-season RAPM adjustments are excluded so the result is a clean first
test of whether the player rating scale can be carried across eras.

It is not part of the possession-level [Leaderboard](leaderboard.md). Its
target is one team-season win percentage rather than a possession outcome.

## Information Boundary

For team \(j\) in season \(t\), player \(i\)'s input \(r_{i,t}^{-}\) is the
RAPM estimate frozen at the end of season \(t-1\). A player with no prior gets
the established cold-start value of zero.

Prior-centered RAPM has an arbitrary league-wide location and scale: adding a
constant to every player coefficient does not change any lineup prediction.
Before comparing seasons, each frozen prior is therefore normalized using the
previous season's player-exposure-weighted mean \(\mu_{t-1}\) and standard
deviation \(\sigma_{t-1}\):

\[
z_{i,t}^{-} = \frac{r_{i,t}^{-} - \mu_{t-1}}{\sigma_{t-1}}.
\]

Both reference quantities are fully known before season \(t\). This does not
add player side information; it only fixes the otherwise unidentified RAPM
coordinate system. The raw prior average remains in the artifact for audit,
but the standardized score is the calibration predictor.

The team signal is the realized-stint-seconds-weighted player average:

\[
S_{j,t} =
\frac{\sum_i q_{i,j,t} z_{i,t}^{-}}
     {\sum_i q_{i,j,t}},
\]

where \(q_{i,j,t}\) is the number of seconds player \(i\) appeared in team
\(j\)'s reconstructed regular-season lineup stints. Overtime seconds are
included naturally. This makes the first version **usage-conditional**: the
rating is forward-looking, while the season's eventual playing-time allocation
is observed retrospectively.

The calibration then fits, using only completed seasons before \(t\),

\[
\widehat{W}_{j,t} =
\operatorname{clip}_{[0,1]}(\alpha_t + \beta_t S_{j,t}),
\]

where \(W_{j,t}\) is regular-season win percentage. Parameters are estimated
by weighted least squares, with a team-season weighted by its game count. The
clip only prevents impossible probabilities at extreme extrapolations.

The 2025-26 estimate is fit on 1997-98 through 2024-25 team outcomes and
evaluated on 2025-26. No 2025-26 win, possession result, or fitted RAPM
adjustment enters its calibration slope or intercept.

## Evaluation

Every season after a four-season warmup receives an expanding-window forecast.
Reported metrics are team-level, with all teams equally weighted:

- win-percentage RMSE and MAE;
- wins RMSE and MAE, retaining the varying season lengths;
- skill relative to predicting a .500 win percentage for every team;
- Spearman rank correlation between predicted and observed team win percentage.

The persistent artifacts include `team_season_inputs.parquet`, all
`forward_predictions.parquet`, season-level metrics, the 2025-26 target
predictions, and the final calibration coefficients. The team input table also
records prior-exposure fraction, making cold-start reliance visible.

## Interpretation

A +1 RAPM point does not receive an assumed, fixed win value. The expanding
historical calibration estimates the empirical relationship at each target
date. That is useful for comparing team strength across eras, but it is not yet
a deployable preseason forecast because it uses realized season minutes.

A later allocation model can replace \(q_{i,j,t}\) with preseason projected
minutes or observed-to-date minutes without changing the rating or calibration
contracts.

## Initial Results

Run `forward-calibration-2025-26-20260804T132355Z-7ec43aa9` uses the regular-only
forward lagged-prior RAPM run
`forward-lagged-rapm-2025-26-20260803T203054Z-c627d89d`.

The target-season calibration was fit on 833 team-seasons from 1997-98 through
2024-25 and then evaluated on all 30 teams and 2,460 games in 2025-26. Median
2025-26 exposure to a player with a prior was 92.8%; the minimum was 62.5%.

| Measure | 2025-26 result |
| --- | ---: |
| Calibration intercept | 0.5136 |
| Win percentage per one standardized team RAPM unit | 0.3254 |
| Win-percentage RMSE | 0.1303 |
| Win-percentage MAE | 0.1091 |
| Wins RMSE | 10.68 |
| Wins MAE | 8.94 |
| Skill vs .500 prediction | 38.4% |
| Spearman team-rank correlation | 0.639 |

The expanding-window backtest covers 25 target seasons from 2001-02 through
2025-26. Its unweighted mean season-level results are 0.0994 win-percentage
RMSE, 7.29 wins RMSE, 56.2% skill versus .500, and 0.754 rank correlation.
This is a useful historical signal, but it should not yet be compared to Vegas
or presented as a preseason model: realized player allocation is doing real
work here.
