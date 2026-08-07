---
last_updated: "2026-08-04"
---

# Leaderboard

This page is the cross-model scoreboard. Every row within a cohort uses the
same eligible possessions, target, and game boundaries, with no information
after the cohort's training cutoff. Training objectives remain model-specific.
**Bold values are best at the displayed precision.** Lower error is better;
higher skill is better.

This board measures **in-season** prediction after fitting target-season
games. The [Frozen Preseason Leaderboard](preseason-leaderboard.md) separately
holds out the entire regular season and playoffs with player values fixed
before opening night.

Last generated: **2026-08-03 20:32 UTC** from `evaluation-2025-26-20260803T203258Z-84bd43a0`.

## Evaluation cohorts

| Cohort | Games | Source possessions | Eligible possessions | Excluded multi-lineup | Eligible share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular-season holdout | 186 | 37,123 | 33,172 | 3,951 | 89.357% |
| Playoffs | 85 | 16,329 | 14,253 | 2,076 | 87.286% |

The regular holdout is the final 186 regular-season games and is untouched by
model selection. The playoff cohort contains all games in the `playoffs`
partition and excludes play-in games. Every evaluated model is frozen before
its cohort begins:

- regular holdout predictions use models fit on the first 1,044 regular-season
  games;
- playoff predictions use models refit on all 1,230 regular-season games;
- no playoff outcomes are used for fitting, calibration, or model selection.

## Possession target

For eligible possession \(i\), define the offense-oriented point margin

\[
y_i = P_{offense,i} - P_{defense,i}.
\]

Most possessions have zero defense points, but retaining the second term
handles unusual opponent scoring without changing the target definition.

Only possessions with exactly one lineup segment are eligible. A possession
with a substitution boundary is excluded in full rather than assigned to a
starting or terminal lineup.

## Possession metrics

For \(N\) eligible possessions:

\[
\operatorname{MSE}_{poss}
= \frac{1}{N}\sum_{i=1}^N(y_i-\widehat{y}_i)^2,
\]

\[
\operatorname{RMSE}_{poss}
= \sqrt{\operatorname{MSE}_{poss}},
\]

\[
\operatorname{MAE}_{poss}
= \frac{1}{N}\sum_{i=1}^N
\left|y_i-\widehat{y}_i\right|.
\]

RMSE and MAE are measured in points per possession. RMSE penalizes large
errors more strongly; MAE gives the average absolute miss.

The mean reference predicts the offense-margin mean from the model's training
window. Possession skill is

\[
\operatorname{Skill}_{poss}
= 1 - \frac{\operatorname{MSE}_{model}}
{\operatorname{MSE}_{mean}}.
\]

Positive skill beats the training mean; zero ties it; negative skill is worse.

## Eligible-possession game margin

Let \(s_i=+1\) when the home team is on offense and \(s_i=-1\) when the away
team is on offense. For game \(g\), aggregate only its eligible possessions:

\[
M_g^{eligible} = \sum_{i \in g}s_i y_i,
\qquad
\widehat{M}_g^{eligible} = \sum_{i \in g}s_i\widehat{y}_i.
\]

Across \(G\) games:

\[
\operatorname{RMSE}_{game}
= \sqrt{
\frac{1}{G}\sum_{g=1}^G
\left(
M_g^{eligible}-\widehat{M}_g^{eligible}
\right)^2
}.
\]

This is deliberately named **eligible-possession game-margin RMSE**. It is not
the error against the official final margin because points from excluded
multi-lineup possessions are absent from both actual and predicted totals.

Using the corresponding mean-reference game predictions, game-margin skill is

\[
\operatorname{Skill}_{game}
= 1 - \frac{\operatorname{MSE}_{game,model}}
{\operatorname{MSE}_{game,mean}}
= 1 - \frac{\operatorname{RMSE}_{game,model}^2}
{\operatorname{RMSE}_{game,mean}^2}.
\]

Possession skill and game-margin skill therefore normalize improvement at
different aggregation levels. Each skill score must be interpreted alongside
the RMSE from the same level.

## RAPM conversion

Ridge and Bayesian RAPM predict home net rating, while the common target is
offense margin per possession. Their predictions are translated as

\[
\widehat{y}_i
= \overline{y}_{train}
+ s_i\frac{\widehat{r}_i}{200},
\]

where \(\widehat{r}_i\) is the predicted home net rating for the possession's
lineup and \(\overline{y}_{train}\) is the regular-training offense-margin
mean. The factor 200 follows from comparing the same two lineups over two
role-swapped possessions. If their signed lineup effect is \(\delta_i\), then

\[
\widehat{r}_i
= 100\left[
(\overline{y}_{train}+\delta_i)
-
(\overline{y}_{train}-\delta_i)
\right]
= 200\delta_i.
\]

## Regular-season holdout

Mean reference: possession RMSE
**1.200988** and
eligible-possession game-margin RMSE
**17.4781**.

| Model | Possession RMSE | Possession MAE | Possession skill vs mean | Eligible-possession game-margin RMSE | Game-margin skill vs mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-year ridge RAPM | 1.199460 | 1.141670 | 0.2542% | 14.7107 | 29.1594% |
| Forward lagged-prior RAPM | **1.199147** | **1.140798** | **0.3063%** | **14.2396** | **33.6240%** |
| One-year Bayesian RAPM | 1.199460 | 1.141670 | 0.2542% | 14.7107 | 29.1594% |
| One-year additive neural | 1.199453 | 1.141746 | 0.2555% | 14.7181 | 29.0884% |
| One-year Deep Sets | 1.199759 | 1.142093 | 0.2046% | 15.1073 | 25.2890% |
| One-year categorical CatBoost | 1.199911 | 1.141991 | 0.1792% | 15.8389 | 17.8770% |
| One-year RAPM + Transformer | 1.199526 | 1.141563 | 0.2434% | 14.7182 | 29.0881% |

## Playoffs

Mean reference: possession RMSE
**1.192560** and
eligible-possession game-margin RMSE
**16.7534**.

| Model | Possession RMSE | Possession MAE | Possession skill vs mean | Eligible-possession game-margin RMSE | Game-margin skill vs mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-year ridge RAPM | **1.191688** | 1.135678 | **0.1462%** | 15.2254 | 17.4097% |
| Forward lagged-prior RAPM | 1.191755 | **1.135572** | 0.1348% | 15.3103 | 16.4860% |
| One-year Bayesian RAPM | **1.191688** | 1.135678 | **0.1462%** | 15.2254 | 17.4097% |
| One-year additive neural | 1.191717 | 1.135770 | 0.1413% | 15.2386 | 17.2667% |
| One-year Deep Sets | 1.191919 | 1.136305 | 0.1074% | 15.2321 | 17.3372% |
| One-year categorical CatBoost | 1.192320 | 1.136179 | 0.0402% | 15.9439 | 9.4310% |
| One-year RAPM + Transformer | 1.191857 | 1.136214 | 0.1178% | **15.2162** | **17.5098%** |

## Interpretation

Bayesian RAPM uses the same Gaussian prior and lambda corresponding to ridge,
so its posterior mean is the ridge point estimate. Equal point-prediction
metrics are expected. Bayesian value appears in uncertainty, interval
calibration, and rank probabilities rather than lower posterior-mean RMSE.

The additive neural and Deep Sets exemplars select learning rate and AdamW
weight decay by validation-possession-weighted MSE across expanding
regular-season folds. CatBoost uses its resolved defaults and chooses its tree
count from the latest chronological validation fold. RAPM + Transformer keeps
the ridge prediction frozen and learns only a position-free attention
residual, using a RAPM fit that excludes every validation or test game it
predicts. Regular holdout and playoff outcomes remain outside every selection
process.

Forward lagged-prior RAPM selects lambda independently within each historical
season, carries the completed prior season's coefficient estimate forward, and
uses zero for players without a prior-season estimate. Its regular-holdout
state is fit only on the first 1,044 games; its playoff state is refit on all
1,230 regular-season games after selection.

## Paired model comparisons

To preserve correlation among possessions from the same game, uncertainty is
estimated by resampling complete games with replacement. Each row identifies
its candidate and reference model. For bootstrap draw \(b\),

\[
\Delta_b =
\operatorname{RMSE}_{candidate,b}
-
\operatorname{RMSE}_{reference,b}.
\]

Negative differences favor the candidate. The interval is the 2.5th through
97.5th percentile of 2,000 paired game-cluster bootstrap draws. The final
column is the share of draws where \(\Delta_b < 0\).

| Cohort | Candidate | Reference | Metric | Difference | 95% interval | P(candidate better) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Regular-season holdout | Forward lagged-prior RAPM | One-year ridge RAPM | Possession RMSE | -0.000313 | [-0.000485, -0.000137] | 100.0% |
| Regular-season holdout | Forward lagged-prior RAPM | One-year ridge RAPM | Eligible-possession game-margin RMSE | -0.471102 | [-0.790552, -0.140570] | 99.6% |
| Playoffs | Forward lagged-prior RAPM | One-year ridge RAPM | Possession RMSE | 0.000068 | [-0.000209, 0.000359] | 32.0% |
| Playoffs | Forward lagged-prior RAPM | One-year ridge RAPM | Eligible-possession game-margin RMSE | 0.084904 | [-0.307806, 0.471020] | 31.9% |
| Regular-season holdout | One-year Deep Sets | One-year additive neural | Possession RMSE | 0.000306 | [0.000170, 0.000452] | 0.0% |
| Regular-season holdout | One-year Deep Sets | One-year additive neural | Eligible-possession game-margin RMSE | 0.389153 | [0.173284, 0.637469] | 0.1% |
| Playoffs | One-year Deep Sets | One-year additive neural | Possession RMSE | 0.000203 | [0.000009, 0.000388] | 2.1% |
| Playoffs | One-year Deep Sets | One-year additive neural | Eligible-possession game-margin RMSE | -0.006494 | [-0.317953, 0.326210] | 52.1% |
| Regular-season holdout | One-year categorical CatBoost | One-year additive neural | Possession RMSE | 0.000459 | [0.000065, 0.000881] | 1.4% |
| Regular-season holdout | One-year categorical CatBoost | One-year additive neural | Eligible-possession game-margin RMSE | 1.120823 | [0.729666, 1.529960] | 0.0% |
| Playoffs | One-year categorical CatBoost | One-year additive neural | Possession RMSE | 0.000603 | [0.000015, 0.001194] | 2.2% |
| Playoffs | One-year categorical CatBoost | One-year additive neural | Eligible-possession game-margin RMSE | 0.705302 | [0.198476, 1.204177] | 0.4% |
| Regular-season holdout | One-year RAPM + Transformer | One-year ridge RAPM | Possession RMSE | 0.000065 | [0.000031, 0.000099] | 0.1% |
| Regular-season holdout | One-year RAPM + Transformer | One-year ridge RAPM | Eligible-possession game-margin RMSE | 0.007408 | [-0.002541, 0.016340] | 7.3% |
| Playoffs | One-year RAPM + Transformer | One-year ridge RAPM | Possession RMSE | 0.000169 | [0.000051, 0.000285] | 0.2% |
| Playoffs | One-year RAPM + Transformer | One-year ridge RAPM | Eligible-possession game-margin RMSE | -0.009226 | [-0.042714, 0.026947] | 71.2% |

## Correctness checks

`tests/test_model_evaluation.py` verifies offense-to-home aggregation,
identical row counts and keys across models, playoff possession construction,
metric calculations, and bolded-winner rendering. The evaluator additionally
requires:

- validated source model manifests and exact artifact hashes;
- Bayesian and Transformer runs derived from the selected ridge run;
- matching regular-holdout game IDs across every model;
- matching possession, game, and player counts across neural-model sources;
- exact held-out possession keys for every stored prediction set;
- Bayesian and ridge posterior-mean equivalence within tolerance.

Run the focused checks with:

```bash
uv run pytest -q tests/test_model_evaluation.py
```

## Reproduce

```bash
uv run nba-evaluate-models 2025-26 \
  --ridge-run-id baseline-2025-26-20260727T230533Z-72eac627 \
  --prior-rapm-run-id forward-lagged-rapm-2025-26-20260803T203054Z-c627d89d \
  --bayesian-run-id bayesian-2025-26-20260729T043953Z-b50cc2f7 \
  --neural-run-id neural-2025-26-20260729T173539Z-51bc0264 \
  --deep-sets-run-id deep-sets-2025-26-20260729T215128Z-dc12dd11 \
  --catboost-run-id catboost-2025-26-20260729T225755Z-9d5251ad \
  --rapm-transformer-run-id rapm-transformer-2025-26-20260729T233233Z-e316a73e
```

| Provenance | Value |
| --- | --- |
| Evaluation run | `evaluation-2025-26-20260803T203258Z-84bd43a0` |
| Ridge run | `baseline-2025-26-20260727T230533Z-72eac627` |
| Forward prior RAPM run | `forward-lagged-rapm-2025-26-20260803T203054Z-c627d89d` |
| Bayesian run | `bayesian-2025-26-20260729T043953Z-b50cc2f7` |
| Neural run | `neural-2025-26-20260729T173539Z-51bc0264` |
| Neural selection | `learning_rate=0.0003`, `weight_decay=0.001`, `epochs=3` |
| Deep Sets run | `deep-sets-2025-26-20260729T215128Z-dc12dd11` |
| Deep Sets selection | `learning_rate=0.001`, `weight_decay=0`, `epochs=1`, `seed=17` |
| CatBoost run | `catboost-2025-26-20260729T225755Z-9d5251ad` |
| CatBoost selection | `max_iterations=1000`, `best_iteration=117`, `trees=118`, `learning_rate=0.113375` |
| RAPM + Transformer run | `rapm-transformer-2025-26-20260729T233233Z-e316a73e` |
| RAPM + Transformer selection | `learning_rate=0.0003`, `weight_decay=0.01`, `epochs=1`, `seed=17` |
| Evaluation code | `sha256:670f2a4facd459d0dd9d960623ad90bdae8e2abcd38179c539841246d3502d0e` |
| Evaluation manifest SHA-256 | `bf490caee5e8004f85a66ad818ddd3be775a303352abd242fa56f4f86bea7b57` |

The underlying `metrics.parquet`, possession predictions, cohort summary, and
source metadata are stored under
`artifacts/reports/model_evaluation/2025-26/evaluation-2025-26-20260803T203258Z-84bd43a0/`.

| Artifact | Contents |
| --- | --- |
| `metrics.parquet` | One row per cohort and model |
| `predictions.parquet` | Every model prediction on every eligible possession |
| `cohorts.parquet` | Inclusion counts, dates, and training cutoffs |
| `comparisons.parquet` | Paired game-cluster bootstrap intervals |
| `model_sources.json` | Model states, translation, means, and unknown exposures |
| `manifest.json` | Source hashes, code fingerprint, and artifact integrity |
