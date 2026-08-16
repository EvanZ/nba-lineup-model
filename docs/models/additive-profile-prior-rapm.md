---
last_updated: "2026-08-15"
---

# Additive Profile-Prior RAPM

This controlled experiment moves the **additive** player-profile terms from
HPM x3's former lineup-context layer into the forward player prior, with no
lineup-composition or matchup term.

For player \(i\) entering season \(t\), the base centered value-conditioned
aging and exposure-gated prior is \(\mu^{\mathrm{base}}_{i,t}\). A
possession-weighted ridge model learns a strictly lagged adjustment:

\[
\mu_{i,t}=\mu^{\mathrm{base}}_{i,t}+\widehat{b}_{i,t}.
\]

\(\widehat{b}_{i,t}\) uses exactly these player-level counterparts of the
additive HPM x3 signals:

- three-point attempts and makes per 100 possessions;
- assists, turnovers, and usage events per 100;
- offensive-rebound claim percentage; and
- steals and blocks per 100.

Those profiles use the same pre-season smoothing and rebound-claim construction
as HPM x3. The model excludes all non-additive composition terms: shooter
depth, top-two or bottom-two summaries, concentration, interactions, imputed
profile counts, and replacement weights.

The possession model is then ordinary forward RAPM:

\[
y_{H,A}=\sum_{i\in H}r_i-\sum_{j\in A}r_j+\alpha_{\mathrm{home}}+\epsilon.
\]

This is the clean test of whether the predictive signal previously found by
the full linear context model can be recovered as an individual player prior.
Each annual ridge penalty is selected using expanding folds over earlier
completed seasons only. Returning players with a prior profile receive the
residual adjustment; cold starts remain on the existing exposure-gated branch.

## Frozen Three-Season Result

The immutable replay covers 2023-24 through 2025-26: 584,970 regular-season
possessions from 3,284 eligible games, 3,511 full games, and 39,967 playoff
possessions. The artifact is:

```text
artifacts/models/analysis/additive_profile_prior_frozen/
frozen_multiseason_backtest/2023-24_to_2025-26/
frozen_multiseason_backtest-2023-24-to-2025-26-20260816T000315Z-3d0fd094
```

| Metric | Additive profile-prior RAPM | Full linear HPM x3 context |
| --- | ---: | ---: |
| Regular possession RMSE | 1.198042 | 1.197978 |
| Regular eligible-game RMSE | 14.2457 | 14.1051 |
| Regular full-game RMSE | 14.4902 | 14.3529 |
| Winner accuracy | 68.07% | 68.13% |
| Team NetRtg RMSE | 3.5860 | 3.4348 |
| Pythagorean-win RMSE | 7.6646 | 7.3731 |
| Playoff possession RMSE | **1.192751** | 1.192766 |
| Playoff eligible-game RMSE | 16.6886 | **16.6817** |

The upstream prior retains a small amount of the additive signal relative to
the complete no-context player-prior control (regular possession RMSE
1.198061), but it does **not** recover the full linear-context model's
regular-season game and team-prediction gains. This rejects the hypothesis that
the additive context features were merely a better-implemented player prior.
At least some useful signal depends on fitting the terms against realized
five-player combinations, even though they are algebraically additive within a
fixed profile representation.

## Paired Bootstrap

The following uses 10,000 paired complete-game resamples, stratified by season,
with seed `20260815`. Deltas are additive profile-prior RAPM minus the reference;
negative RMSE/MAE values favor the additive prior.

| Reference | Metric | Delta | Paired 95% interval | P(additive prior better) |
| --- | --- | ---: | --- | ---: |
| Complete no-context player prior | Full-game RMSE | +0.0146 | [-0.0259, +0.0552] | 24.79% |
| Complete no-context player prior | Winner accuracy | +0.43 pp | [-0.28 pp, +1.14 pp] | 86.52% |
| Complete no-context player prior | Possession RMSE | -0.000019 | [-0.000042, +0.000003] | 95.30% |
| Complete no-context player prior | Possession MAE | -0.000246 | [-0.000270, -0.000222] | 100.00% |
| Full linear HPM x3 context | Full-game RMSE | +0.1374 | [+0.0631, +0.2129] | 0.02% |
| Full linear HPM x3 context | Possession RMSE | +0.000064 | [+0.000024, +0.000104] | 0.10% |
| Full linear HPM x3 context | Possession MAE | +0.000080 | [+0.000036, +0.000123] | 0.00% |

The immutable bootstrap report is:

```text
artifacts/models/analysis/additive_profile_prior_bootstrap/2023-24_to_2025-26/
additive-profile-prior-bootstrap-20260816T003707Z-e9be1af7
```
