# Train the RAPM Aging Model

The aging trainer consumes the validated multi-season player transition panel
and produces forward-only priors for its latest target season.

## Prerequisites

Each source season needs curated regular-season data, player bios, and a
canonical RAPM run. Build the panel after those runs complete:

```bash
uv run nba-build-player-season-panel \
  1996-97 1997-98 1998-99 1999-00 2000-01 2001-02 2002-03 2003-04 \
  2004-05 2005-06 2006-07 2007-08 2008-09 2009-10 2010-11 2011-12 \
  2012-13 2013-14 2014-15 2015-16 2016-17 2017-18 2018-19 2019-20 \
  2020-21 2021-22 2022-23 2023-24 2024-25 2025-26
```

The aging model requires at least three target seasons: an initial training
season, one expanding validation season, and one untouched holdout.

## Train

Use the latest transition target as the holdout:

```bash
uv run nba-train-aging-model
```

Pin an earlier target for a historical walk-forward experiment:

```bash
uv run nba-train-aging-model --holdout-season 2024-25
```

Rows after an explicitly selected holdout are excluded from fitting,
preprocessing, and hyperparameter selection.

Customize the normalized ridge grid or age basis:

```bash
uv run nba-train-aging-model \
  --regularizations 0.0001,0.001,0.01,0.1,1,10 \
  --age-spline-knots 5 \
  --age-spline-degree 2
```

## Outputs

Immutable runs are written under:

```text
artifacts/models/aging/{holdout_season}/{run_id}/
```

| File | Purpose |
| --- | --- |
| `fold_metrics.parquet` | Candidate metrics for every expanding season fold |
| `hyperparameter_summary.parquet` | Pooled candidate selection evidence |
| `cv_predictions.parquet` | Out-of-time predictions from the selected model |
| `holdout_predictions.parquet` | Holdout labels and all baseline predictions |
| `holdout_metrics.parquet` | Overall, eligible, returning, and cold-start metrics |
| `player_priors.parquet` | Label-free priors consumable by RAPM and neural models |
| `uncertainty_scales.parquet` | Returning and cold-start predictive error scales |
| `feature_coefficients.parquet` | Fitted transformed-feature coefficients |
| `model_parameters.json` | Resolved features, alpha, split, and intercept |
| `model.joblib` | Fitted preprocessing and ridge pipeline |
| `manifest.json` | Source, code, configuration, row, and artifact hashes |

`latest.json` is updated only after the complete run validates. The CLI then
indexes the immutable run in MLflow.

## Interpret the result

The first decision boundary is whether aging ridge improves exposure-weighted
holdout RMSE over persistence. Review returning and cold-start cohorts
separately; a useful returning-player curve can coexist with weak cold-start
performance.

Do not join `holdout_predictions.parquet` into another model. It contains
evaluation outcomes. `player_priors.parquet` is the sole predictive handoff.
