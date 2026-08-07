---
last_updated: "2026-07-27"
---

# Baseline Methodology

## Null model

The null prediction is the possession-weighted training mean. Because every
target is expressed from the home perspective, its intercept includes average
home-court advantage.

## Team model

\[
\widehat{y} =
\text{HCA} + T_{home} - T_{away}
\]

The signed sparse matrix has one column per team. It provides the main
schedule-adjusted predictive baseline for RAPM.

## One-number RAPM

\[
\widehat{y} =
\text{HCA}
+ \sum_{p \in home} \beta_p
- \sum_{p \in away} \beta_p
\]

There is one coefficient per player and no offensive/defensive split. The
initial model includes no bio features, draft prior, score context, or
interaction terms.

## Selection and evaluation

Team and player penalties are evaluated on expanding chronological folds. The
final test is the last 15% of regular-season games and is not used for lambda
selection. After test metrics are recorded, an all-season refit produces the
ranking artifact.

Reported metrics include:

- possession-weighted RMSE and MAE;
- game-margin RMSE after aggregating stint predictions;
- skill relative to the mean baseline;
- skill relative to the team baseline.

Raw on-court net rating is included beside RAPM as a descriptive comparison,
not as a predictive model.
