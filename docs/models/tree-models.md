---
last_updated: "2026-07-29"
---

# Tree Models

CatBoost provides an orthogonal nonlinear baseline for the lineup problem. It
does not share the additive coefficient structure of RAPM or the differentiable
representations of the neural models. The first specification asks a narrow
question: can boosted trees extract useful lineup interactions when player
state is represented entirely by categorical features?

The implementation follows the CatBoost paper and native categorical-feature
interface rather than preprocessing categories outside the model:

- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [Categorical feature transformation](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)
- [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/)

## Feature contract

For \(P\) players in the regular-season vocabulary, possession \(i\) becomes a
\((P+1)\)-element vector:

\[
x_i =
\left[
x_{i,1},\ldots,x_{i,P},h_i
\right].
\]

Each player column is categorical:

\[
x_{i,p} =
\begin{cases}
0 & \text{player }p\text{ is absent},\\
1 & \text{player }p\text{ is on offense},\\
2 & \text{player }p\text{ is on defense}.
\end{cases}
\]

The final categorical feature is

\[
h_i =
\begin{cases}
1 & \text{home team is on offense},\\
0 & \text{away team is on offense}.
\end{cases}
\]

For 2025-26 this produces 582 player-state columns plus one home-offense
column, or 583 features for each of 218,810 eligible possessions. Player order
inside either lineup cannot affect the vector. A player first seen outside the
training vocabulary remains absent in every known player column; the evaluator
records that exposure separately.

All columns are passed to a CatBoost `Pool` as categorical. Setting
`one_hot_max_size=3` means every player-state feature is handled by CatBoost's
native one-hot path because its complete known domain has three values.
CatBoost does not calculate target statistics for features that meet this
threshold.

## Prediction function

The model predicts offense points minus defense points directly:

\[
\widehat{m_i}
= b + \sum_{t=1}^{T} f_t(x_i),
\]

where each \(f_t\) is a fitted symmetric regression tree. Tree paths can split
on several player states, so the ensemble can represent lineup interactions.
Unlike an embedding model, it does not learn a compact player representation
that can transfer similarity between players.

## Defaults-first protocol

The initial experiment deliberately avoids a manual parameter search. It
explicitly fixes only the experiment boundary:

| Control | Value |
| --- | ---: |
| Loss and evaluation metric | RMSE |
| Maximum iterations | 1,000 |
| `one_hot_max_size` | 3 |
| `has_time` | `true` |
| `use_best_model` | `true` |
| Random seed | 17 |

`has_time=true` preserves the chronological input order rather than applying a
random object permutation. Each of the three expanding validation folds trains
to the 1,000-iteration ceiling and saves trees only through its best validation
iteration. The latest fold determines the tree count for the untouched
regular-season holdout refit and the all-regular-season playoff refit.

This is **not** automatic hyperparameter optimization. CatBoost resolves
unspecified defaults, some dynamically, and `get_all_params()` records their
fitted values. Only the number of retained trees is selected from validation.
The distinction follows the CatBoost documentation for
[`get_all_params()`](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_all_params)
and
[`use_best_model`](https://catboost.ai/docs/en/references/training-parameters/common).

## 2025-26 exemplar

Run `catboost-2025-26-20260729T225755Z-9d5251ad` completed in about 122 seconds
on CPU. The latest validation fold selected zero-based iteration 117, so the
stored holdout and full-season models contain 118 trees.

Important resolved values were:

| Parameter | Resolved value |
| --- | ---: |
| Learning rate | `0.113375` |
| Depth | 6 |
| L2 leaf regularization | 3 |
| Bootstrap | MVS |
| Subsample | `0.8` |
| Random strength | 1 |
| Feature subsampling (`rsm`) | 1 |
| Grow policy | SymmetricTree |
| Boosting type | Plain |

The fold-specific best iterations were 130, 106, and 117. The untouched
regular-season holdout result was:

| Model | Possession RMSE | Possession skill | Game-margin RMSE | Game-margin skill |
| --- | ---: | ---: | ---: | ---: |
| Mean reference | 1.200988 | 0.0000% | 17.4781 | 0.0000% |
| CatBoost | 1.199911 | 0.1792% | 15.8389 | 17.8770% |

CatBoost therefore learns signal, but does not lead the current scoreboard:

| Cohort | Metric | CatBoost - additive neural | 95% interval |
| --- | --- | ---: | ---: |
| Regular holdout | Possession RMSE | +0.000459 | [+0.000065, +0.000881] |
| Regular holdout | Game-margin RMSE | +1.120823 | [+0.729666, +1.529960] |
| Playoffs | Possession RMSE | +0.000603 | [+0.000015, +0.001194] |
| Playoffs | Game-margin RMSE | +0.705302 | [+0.198476, +1.204177] |

All four paired game-cluster intervals favor the additive neural model. That
makes CatBoost a credible nonlinear control rather than a candidate for a
larger search yet. Future experiments can add pre-possession context or test a
small, predeclared tuning budget without replacing this defaults-first result.

## Interpretation

`feature_importance.parquet` stores CatBoost's default
PredictionValuesChange importance for each player-state feature and the
home-offense feature. These values show how much a feature changes predictions
within this fitted ensemble. They are not signed player ratings, causal
effects, or a substitute for RAPM.

Direction and context require prediction-based analysis:

- one-player replacement counterfactuals;
- role-swapped lineup predictions;
- CatBoost SHAP values for specified possession rows;
- interaction importance for selected player pairs;
- stability across seasons and seeds.

CatBoost documents the available
[feature-importance methods](https://catboost.ai/docs/en/features/feature-importances-calculation).
Any public player analysis should identify the background cohort and preserve
the lineup context used for the calculation.

## Correctness tests

`tests/test_catboost_modeling.py` verifies:

- exact offense and defense category codes;
- invariance to player order within each lineup;
- one feature per player plus home-offense context;
- best-iteration and retained-tree consistency;
- complete recording of CatBoost's resolved parameters;
- finite predictions when playoff rows contain unknown players;
- explicit counting of unknown player exposures.

Run the focused checks with:

```bash
uv run pytest -q tests/test_catboost_modeling.py
```

Operational commands and artifact definitions are in
[Train CatBoost](../guides/train-catboost.md). Common holdout and playoff
results are maintained on the [Leaderboard](leaderboard.md).
