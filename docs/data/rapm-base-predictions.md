---
last_updated: "2026-07-30"
---

# RAPM Base Predictions

The RAPM + Transformer model does not estimate its additive baseline inside
the network. It consumes frozen possession predictions produced from the
canonical stint-weighted ridge RAPM model. The analytical mart makes every
base-model fit and prediction boundary explicit.

## Location

```text
data/analytical/rapm_base_predictions/<season>/regular/
  _manifest.json
  part-00000.parquet
  rapm_player_coefficients.parquet
  stage_parameters.parquet
```

The 2025-26 mart contains 929,003 stage-possession rows for 218,810 distinct
eligible possessions. Rows repeat across stages because each model-selection
or refit stage has its own RAPM training cutoff.

## Stages

| Stage | RAPM training games | Prediction roles |
| --- | ---: | --- |
| `cv_0` | 677 | Train and validation |
| `cv_1` | 798 | Train and validation |
| `cv_2` | 920 | Train and validation |
| `final` | 1,044 | Train and untouched test |
| `all_season` | 1,230 | Train |

Every stage reuses the lambda selected by the source one-year RAPM run. It
refits coefficients and the home-court intercept on only that stage's training
stints.

## Prediction conversion

For stage \(k\), RAPM predicts home net rating

\[
\widehat r_{i,k}
= \widehat\alpha_k
+ \sum_{p\in H_i}\widehat\beta_{p,k}
- \sum_{p\in A_i}\widehat\beta_{p,k}.
\]

Let \(s_i=+1\) for home offense and \(-1\) for away offense. The
offense-oriented possession prediction is

\[
\widehat y^{RAPM}_{i,k}
= \overline y_k
+ s_i\frac{\widehat r_{i,k}}{200},
\]

where \(\overline y_k\) is mean offense margin among the stage's training
possessions. The residual target is

\[
e_{i,k}=y_i-\widehat y^{RAPM}_{i,k}.
\]

## Leakage boundary

Each row records `stage`, `role`, `base_train_game_count`, and
`base_is_out_of_sample`.

- Training rows use predictions from the RAPM fitted on that same stage's
  training games. This is ordinary residual fitting, analogous to fitting a
  second learner to a first learner's training residuals.
- Validation and test rows are predicted only by a RAPM fitted on earlier
  games. Their `base_is_out_of_sample` value must be `true`.
- The final 186 regular-season games and all playoff games remain absent from
  every model-selection fit.

The Transformer therefore never receives a base prediction from a RAPM model
that used that evaluation game's outcome. This does not make training-row base
predictions cross-fitted; that stricter stacking variant remains a future
ablation.

## Prediction rows

`part-00000.parquet` contains:

| Column | Meaning |
| --- | --- |
| `stage` | `cv_0`, `cv_1`, `cv_2`, `final`, or `all_season` |
| `role` | `train`, `validation`, or `test` |
| `base_is_out_of_sample` | Whether the game's outcome was excluded from RAPM fitting |
| `base_train_game_count` | Number of games available to that RAPM state |
| `rapm_home_net_rating` | Raw home-oriented RAPM prediction |
| `prediction_rapm` | Converted offense-margin prediction |
| `prediction_rapm_home_margin` | Converted home-margin prediction |
| `residual_target` | Actual offense margin minus `prediction_rapm` |

The remaining columns preserve the possession key, time, target, and
home-offense sign.

## Fitted states

`rapm_player_coefficients.parquet` contains one coefficient per stage and
player. `stage_parameters.parquet` records lambda, scikit-learn alpha,
intercept, training mean, train and prediction counts, and the exact training
cutoff.

The manifest hashes the source RAPM run, current RAPM stints, neural
possessions, all three Parquet files, and the builder code. Existing
RAPM-stint and neural-possession partitions are validated and reused rather
than rewritten merely because a model command was invoked.

## Correctness tests

`tests/test_transformer_modeling.py` perturbs validation outcomes and verifies
that their fold-specific RAPM predictions do not change. Runtime validation
also requires:

- identical chronological game splits between stints and possessions;
- prediction keys unique within stage;
- every non-training row marked out of sample;
- every evaluation game strictly after its stage's training cutoff;
- exact residual arithmetic;
- one retained coefficient per stage and player.

Run the focused checks with:

```bash
uv run pytest -q tests/test_transformer_modeling.py
```
