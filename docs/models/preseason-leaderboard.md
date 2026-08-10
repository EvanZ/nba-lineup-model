---
last_updated: "2026-08-09"
---

# Frozen Preseason Leaderboard

This leaderboard evaluates models as true preseason forecasts. Every player
value is frozen before the target season begins. Target-season lineups and
exposure are supplied by an oracle, but no target-season score, possession
outcome, fitted player adjustment, or playoff result can change the model.

[Model Evolution](index.md#model-evolution) visualizes these experiments as an
interactive primary-parent tree. Its default ordering uses frozen regular-season
game-margin RMSE, with selectors for every leaderboard metric it records.

This is distinct from the [in-season Leaderboard](leaderboard.md), where models
fit the first 1,044 regular-season games before predicting the final 186.

## Information Boundary

The initial baseline predicts 2025-26 from the completed 2024-25 regular-only
forward RAPM state:

| Component | Frozen source |
| --- | --- |
| Player values | 2024-25 regular-season forward RAPM |
| Cold starts | Zero RAPM |
| Offense-margin mean | 2024-25 regular season |
| Home-court effect | Recovered from the completed 2024-25 RAPM state |
| Lineups and exposure | Realized 2025-26 oracle allocation |
| Player refit | None |

Prior-season playoffs are intentionally excluded from this first baseline.

For eligible target-season possession $i$,

\[
\widehat y_i =
\overline y_{2024-25}
+ \frac{\sum_{p\in O_i}r_{p,2024-25}
- \sum_{p\in D_i}r_{p,2024-25}}{200}
+ s_i\frac{h_{2024-25}}{200},
\]

where $s_i=+1$ for home offense and $-1$ for away offense. The factor 200
uses the same one-number RAPM-to-possession conversion as the in-season
Leaderboard.

Every model name in the evaluation tables links to its technical guide. Hover
over a linked name for a short description of the model change.

## Metric Definitions

All error metrics are evaluated after the player values and every other fitted
component have been frozen before 2025-26. Lower RMSE and MAE are better; higher
skill, accuracy, and correlation are better. Regular season and playoffs are
always scored separately.

Let \(i\) denote a possession, \(g\) a game, \(t\) a team, and \(E_g\) the
set of possession rows in game \(g\) with exactly one reconstructed lineup.
For each eligible possession, \(y_i\) is its actual offense-oriented point
margin, \(\widehat y_i\) is the forecast, and \(s_i\in\{-1,+1\}\) converts
the value into the home-team frame. Let \(G\) be the games in the cohort and
\(N\) its eligible possessions.

### Possession Errors And Skill

The possession columns use each row in \(\bigcup_g E_g\):

\[
\operatorname{RMSE}_{\mathrm{poss}} =
\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat y_i)^2},
\qquad
\operatorname{MAE}_{\mathrm{poss}} =
\frac{1}{N}\sum_{i=1}^{N}\lvert y_i-\widehat y_i\rvert.
\]

The frozen-mean reference predicts the completed source season's league-wide
offense-margin mean for every possession. Skill reports the relative reduction
in mean squared error against that fixed reference:

\[
\operatorname{Skill}_{\mathrm{poss}} =
1 - \frac{\operatorname{MSE}_{\mathrm{model}}}
{\operatorname{MSE}_{\mathrm{frozen\ mean}}}.
\]

Thus, zero means no improvement over the frozen mean and a negative value means
the model is worse. The `Games` and `Possessions` columns are simply
\(\lvert G\rvert\) and \(N\), respectively; they define the scoring coverage.

### Eligible-Possession Game Margins

The `Game-margin RMSE` and `Game skill vs frozen mean` columns in the
Possession And Game Results table do **not** use final official margins. They
first sum only eligible rows in the home-team frame:

\[
M_g^{(E)} = \sum_{i\in E_g}s_i y_i,
\qquad
\widehat M_g^{(E)} = \sum_{i\in E_g}s_i\widehat y_i.
\]

They then score the per-game aggregate:

\[
\operatorname{RMSE}_{\mathrm{eligible\ game}} =
\sqrt{\frac{1}{\lvert G\rvert}
\sum_{g\in G}\left(M_g^{(E)}-\widehat M_g^{(E)}\right)^2}.
\]

`Game skill vs frozen mean` applies the same MSE-ratio skill formula to these
eligible-game margins. This definition is available for both regular season and
playoffs because it is based on the common possession-reconstruction boundary.

### Full Regular-Season Game Outcomes

The Full-Game Outcomes table is a separate, stricter game-level regular-season
evaluation. It uses every allocated lineup stint, including rows outside
\(E_g\). With \(n_s\) possessions and home-net-rating forecast
\(\widehat R_s\) for stint \(s\), the model's final home margin is

\[
\widehat M_g = \sum_{s\in g}\frac{n_s\widehat R_s}{100}.
\]

Let \(M_g\) be the official final home margin. `Full-game margin RMSE` and
`Full-game margin MAE` are

\[
\operatorname{RMSE}_{\mathrm{full}} =
\sqrt{\frac{1}{\lvert G\rvert}\sum_{g\in G}(M_g-\widehat M_g)^2},
\qquad
\operatorname{MAE}_{\mathrm{full}} =
\frac{1}{\lvert G\rvert}\sum_{g\in G}\lvert M_g-\widehat M_g\rvert.
\]

`Winner accuracy` compares the signs of \(M_g\) and \(\widehat M_g\). A
forecast of exactly zero receives one-half credit, rather than being
arbitrarily treated as a road win:

\[
\operatorname{Accuracy} = \frac{1}{\lvert G\rvert}\sum_{g\in G}
\begin{cases}
1, & \operatorname{sign}(\widehat M_g)=\operatorname{sign}(M_g),\\
\tfrac{1}{2}, & \widehat M_g=0,\\
0, & \text{otherwise.}
\end{cases}
\]

`Predicted ties` counts games with \(\widehat M_g=0\). Full-game outcomes are
currently regular season only; the stored frozen playoff artifacts retain only
eligible-possession game aggregates, not the all-stint prediction needed for
this definition.

### Team Net Rating And Win Totals

For each regular-season team, actual and predicted net rating are calculated
from all allocated RAPM stints. If \(d_{ts}\) and \(\widehat d_{ts}\) are the
actual and predicted team margins in stint \(s\), then

\[
\operatorname{NetRtg}_t =
100\frac{\sum_s d_{ts}}{\sum_s n_s},
\qquad
\widehat{\operatorname{NetRtg}}_t =
100\frac{\sum_s \widehat d_{ts}}{\sum_s n_s}.
\]

`Net-rating RMSE` and `Net-rating MAE` apply the usual formulas above across
the 30 teams. Pearson correlation is the linear correlation between predicted
and actual team NetRtg; Spearman correlation is their rank correlation.

`Pythagorean wins` are the forecasted win totals obtained by applying the
frozen historical NetRtg-to-win-percentage calibration to each predicted team
NetRtg. For team \(t\) with \(G_t\) games,

\[
\widehat W_t = G_t\,
\operatorname{clip}\left(a+b\widehat{\operatorname{NetRtg}}_t,0,1\right).
\]

The calibrated coefficients \(a\) and \(b\) are reported below. `Win-total
RMSE` and `Win-total MAE` compare \(\widehat W_t\) with actual wins \(W_t\)
across teams. `Win-percentage RMSE` uses \(\widehat W_t/G_t\) and \(W_t/G_t\).
The final Spearman column is the rank correlation between predicted and actual
team win totals.

## Possession And Game Results

Regular season and playoffs are evaluated separately. Possession metrics use
only possessions with one reconstructed lineup, matching the existing neural
evaluation boundary. Game-margin RMSE aggregates those eligible possessions in
the home-team frame.

| Model | Cohort | Games | Possessions | Possession RMSE | Possession MAE | Game-margin RMSE | Possession skill vs frozen mean | Game skill vs frozen mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | Regular season | 1,230 | 218,810 | 1.199061 (#17) | 1.142303 (#16) | 15.0694 (#19) | 0.0704% (#17) | 9.4406% (#19) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | Regular season | 1,230 | 218,810 | 1.198919 (#7) | 1.142004 (#11) | 14.7820 (#10) | 0.0941% (#7) | 12.8614% (#10) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | Regular season | 1,230 | 218,810 | 1.199000 (#12) | 1.142154 (#14) | 14.8894 (#14) | 0.0805% (#12) | 11.5905% (#14) |
| [Frozen aging prior][frozen-aging-prior] | Regular season | 1,230 | 218,810 | 1.199062 (#18) | 1.142736 (#19) | 15.0203 (#18) | 0.0702% (#18) | 10.0297% (#18) |
| [Frozen O/D RAPM][frozen-od-rapm] | Regular season | 1,230 | 218,810 | 1.198853 (#3) | 1.142664 (#18) | 14.8901 (#15) | 0.1092% (#2) | 11.5692% (#16) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | Regular season | 1,230 | 218,810 | 1.198986 (#9) | 1.142088 (#13) | 14.8590 (#13) | 0.0829% (#9) | 11.9512% (#13) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | Regular season | 1,230 | 218,810 | 1.198952 (#8) | 1.141943 (#9) | 14.7413 (#8) | 0.0886% (#8) | 13.3410% (#8) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | Regular season | 1,230 | 218,810 | **1.198792 (#1)** | 1.142422 (#17) | 14.7631 (#9) | **0.1193% (#1)** | 13.0714% (#9) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | Regular season | 1,230 | 218,810 | 1.198993 (#11) | 1.142055 (#12) | 14.8225 (#12) | 0.0817% (#11) | 12.3830% (#12) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | Regular season | 1,230 | 218,810 | 1.199014 (#14) | 1.142242 (#15) | 14.8908 (#16) | 0.0782% (#14) | 11.5747% (#15) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | Regular season | 1,230 | 218,810 | 1.198989 (#10) | 1.141996 (#10) | 14.7993 (#11) | 0.0825% (#10) | 12.6578% (#11) |
| [Forward contextual RAPM][forward-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199008 (#13) | 1.141571 (#2) | 14.6525 (#6) | 0.0793% (#13) | 14.3817% (#6) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199259 (#19) | 1.141636 (#4) | 14.9769 (#17) | 0.0373% (#19) | 10.5488% (#17) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199030 (#16) | 1.141625 (#3) | 14.6349 (#5) | 0.0755% (#16) | 14.5870% (#5) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198859 (#4) | 1.141648 (#5) | 14.5410 (#2) | 0.1041% (#4) | 15.6803% (#2) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198894 (#5) | 1.141695 (#6) | 14.5485 (#3) | 0.0982% (#5) | 15.5932% (#3) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198917 (#6) | 1.141735 (#7) | 14.5953 (#4) | 0.0945% (#6) | 15.0483% (#4) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198831 (#2) | 1.141770 (#8) | **14.3654 (#1)** | 0.1088% (#3) | **17.7044% (#1)** |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199026 (#15) | **1.141556 (#1)** | 14.6770 (#7) | 0.0762% (#15) | 14.0952% (#7) |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | Playoffs | 85 | 14,253 | 1.192713 (#5) | 1.136139 (#11) | 17.1867 (#5) | -0.0468% (#5) | -5.2607% (#5) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | Playoffs | 85 | 14,253 | 1.192614 (#3) | 1.135787 (#4) | 16.9988 (#3) | -0.0302% (#3) | -2.9720% (#3) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | Playoffs | 85 | 14,253 | 1.192895 (#9) | 1.136163 (#13) | 17.5409 (#12) | -0.0774% (#9) | -9.6446% (#12) |
| [Frozen aging prior][frozen-aging-prior] | Playoffs | 85 | 14,253 | **1.192332 (#1)** | 1.136455 (#17) | **16.4946 (#1)** | **0.0170% (#1)** | **3.0460% (#1)** |
| [Frozen O/D RAPM][frozen-od-rapm] | Playoffs | 85 | 14,253 | 1.194642 (#17) | 1.139203 (#18) | 17.8037 (#17) | -0.3924% (#17) | -12.9804% (#17) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | Playoffs | 85 | 14,253 | 1.192971 (#13) | 1.136187 (#15) | 17.6339 (#15) | -0.0901% (#13) | -10.8093% (#15) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | Playoffs | 85 | 14,253 | 1.192980 (#14) | 1.136174 (#14) | 17.6725 (#16) | -0.0917% (#14) | -11.2952% (#16) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | Playoffs | 85 | 14,253 | 1.194678 (#18) | 1.139209 (#19) | 17.8562 (#18) | -0.3984% (#18) | -13.6482% (#18) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | Playoffs | 85 | 14,253 | 1.192821 (#6) | 1.136096 (#8) | 17.4188 (#7) | -0.0649% (#6) | -8.1231% (#7) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | Playoffs | 85 | 14,253 | 1.192621 (#4) | 1.136193 (#16) | 17.1300 (#4) | -0.0314% (#4) | -4.5672% (#4) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | Playoffs | 85 | 14,253 | 1.192895 (#9) | 1.136105 (#9) | 17.4713 (#9) | -0.0774% (#9) | -8.7752% (#9) |
| [Forward contextual RAPM][forward-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192898 (#10) | **1.135625 (#1)** | 17.5493 (#13) | -0.0778% (#10) | -9.7486% (#13) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | Playoffs | 85 | 14,253 | 1.193395 (#16) | 1.136131 (#10) | 18.4522 (#19) | -0.1613% (#16) | -21.3327% (#19) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192994 (#15) | 1.135701 (#3) | 17.5261 (#11) | -0.0939% (#15) | -9.4596% (#11) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192857 (#7) | 1.136067 (#7) | 17.4003 (#6) | -0.0710% (#7) | -7.8941% (#6) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192872 (#8) | 1.136002 (#5) | 17.4604 (#8) | -0.0736% (#8) | -8.6406% (#8) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192935 (#11) | 1.136059 (#6) | 17.4742 (#10) | -0.0840% (#11) | -8.8115% (#10) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192544 (#2) | 1.136145 (#12) | 16.5388 (#2) | -0.0185% (#2) | 2.5259% (#2) |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192959 (#12) | 1.135643 (#2) | 17.6146 (#14) | -0.0881% (#12) | -10.5673% (#14) |

Bolding marks the better value within each cohort and metric. Future frozen
priors must use these exact cohorts.

## Full-Game Outcomes

<!-- frozen-full-game-outcomes:start -->
Full-game outcomes aggregate every allocated regular-season stint to the official final home margin. This is deliberately distinct from the eligible-possession game
metric above, which excludes possession rows without one reconstructed lineup. Winner accuracy calls the sign of that full-game margin; an exactly zero forecast
receives half credit rather than being arbitrarily assigned to the away team.

| Model | Games | Full-game margin RMSE | Full-game margin MAE | Winner accuracy | Predicted ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | 1,230 | 15.5605 (#19) | 12.3241 (#19) | 63.17% (#16) | 0 |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | 1,230 | 15.1944 (#11) | 12.0436 (#11) | 65.20% (#13) | 0 |
| [Frozen lagged RAPM][frozen-lagged-rapm] | 1,230 | 15.3211 (#15) | 12.1338 (#16) | 65.69% (#11) | 0 |
| [Frozen aging prior][frozen-aging-prior] | 1,230 | 15.5090 (#18) | 12.2839 (#18) | 65.04% (#15) | 0 |
| [Frozen O/D RAPM][frozen-od-rapm] | 1,230 | 15.4015 (#17) | 12.2653 (#17) | 65.04% (#15) | 0 |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | 1,230 | 15.3025 (#14) | 12.1092 (#13) | 65.12% (#14) | 0 |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | 1,230 | 15.1594 (#9) | 11.9812 (#8) | 65.77% (#10) | 0 |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | 1,230 | 15.2474 (#12) | 12.1289 (#15) | 65.28% (#12) | 0 |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | 1,230 | 15.1802 (#10) | 12.0162 (#10) | 66.67% (#9) | 0 |
| [Student-t recursive RAPM][student-t-recursive-rapm] | 1,230 | 15.2758 (#13) | 12.1226 (#14) | 65.69% (#11) | 0 |
| [Student-t talent-prior RAPM][student-t-talent-prior] | 1,230 | 15.1499 (#8) | 11.9941 (#9) | 67.56% (#5) | 0 |
| [Forward contextual RAPM][forward-contextual-rapm] | 1,230 | 15.0190 (#6) | 11.8474 (#6) | **67.89% (#1)** | 0 |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | 1,230 | 15.3555 (#16) | 12.1046 (#12) | 67.48% (#6) | 0 |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | 1,230 | 14.9602 (#5) | 11.7780 (#4) | 67.72% (#3) | 0 |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | 1,230 | 14.8516 (#2) | 11.7498 (#2) | 67.64% (#4) | 0 |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | 1,230 | 14.8706 (#3) | 11.7602 (#3) | 67.80% (#2) | 0 |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | 14.9224 (#4) | 11.8112 (#5) | 67.40% (#7) | 0 |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | **14.6265 (#1)** | **11.5301 (#1)** | 67.40% (#7) | 0 |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | 1,230 | 15.0487 (#7) | 11.8634 (#7) | 67.32% (#8) | 0 |

Source report: `artifacts/models/frozen_game_outcomes/2025-26/frozen_game_outcomes-2025-26-20260810T024501Z-0825db96`. The retained `game_outcome_predictions.parquet`
contains one final-margin prediction per model and game, and `sources.parquet`
pins every upstream immutable manifest.
<!-- frozen-full-game-outcomes:end -->

## Team Net Rating

Team net rating uses all regular-season RAPM stints, including allocated
multi-lineup possessions. For stint (s), the frozen lineup prediction is
converted to a predicted margin using the stint's oracle possession exposure;
team margins are then summed and divided by team possessions.

| Model | Teams | Net-rating RMSE | Net-rating MAE | Pearson correlation | Spearman correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | 30 | 4.9792 (#18) | 4.2342 (#18) | 0.5890 (#18) | 0.5444 (#19) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | 30 | 4.6618 (#9) | 4.0620 (#17) | 0.6490 (#10) | 0.6271 (#10) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | 30 | 4.8538 (#16) | 3.9606 (#15) | 0.6113 (#17) | 0.5493 (#18) |
| [Frozen aging prior][frozen-aging-prior] | 30 | 5.0366 (#19) | 4.2395 (#19) | 0.6484 (#11) | 0.6111 (#13) |
| [Frozen O/D RAPM][frozen-od-rapm] | 30 | 4.8740 (#17) | 3.9380 (#14) | 0.6113 (#17) | 0.5626 (#16) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | 30 | 4.8473 (#15) | 3.8653 (#13) | 0.6163 (#15) | 0.5528 (#17) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | 30 | 4.6989 (#11) | 3.6020 (#8) | 0.6480 (#12) | 0.5844 (#15) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | 30 | 4.7113 (#12) | 3.6999 (#10) | 0.6454 (#14) | 0.6018 (#14) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | 30 | 4.6680 (#10) | 3.8199 (#12) | 0.6471 (#13) | 0.6236 (#11) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | 30 | 4.8263 (#14) | 3.9928 (#16) | 0.6159 (#16) | 0.6147 (#12) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | 30 | 4.6069 (#8) | 3.7344 (#11) | 0.6587 (#9) | 0.6383 (#8) |
| [Forward contextual RAPM][forward-contextual-rapm] | 30 | 4.1572 (#2) | 3.1821 (#3) | 0.7427 (#2) | 0.7219 (#2) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | 30 | 4.7470 (#13) | 3.6765 (#9) | 0.6822 (#8) | 0.6320 (#9) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | 30 | 4.1864 (#5) | 3.1994 (#4) | 0.7394 (#4) | 0.7201 (#3) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | 30 | 4.1883 (#6) | 3.2466 (#5) | 0.7354 (#5) | 0.7041 (#5) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | 30 | 4.1799 (#4) | 3.3077 (#6) | 0.7338 (#6) | 0.6974 (#6) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 4.2546 (#7) | 3.3952 (#7) | 0.7222 (#7) | 0.6774 (#7) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | **3.7737 (#1)** | **3.0395 (#1)** | **0.7920 (#1)** | **0.7562 (#1)** |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | 30 | 4.1771 (#3) | 3.1699 (#2) | 0.7412 (#3) | 0.7077 (#4) |

## Team Win Totals

The primary win estimate is called **Pythagorean wins** in this project. It is
a forward-safe historical mapping from predicted team net rating to expected
win percentage, rather than the traditional points-for/points-against
Pythagorean exponent. The mapping is fit by game-weighted least squares on 862
regular-season team-seasons from 1996-97 through 2024-25:

\[
\widehat{\operatorname{WinPct}}_t =
\operatorname{clip}\left(
0.499583 + 0.030250\,\widehat{\operatorname{NetRtg}}_t,
0,
1
\right).
\]

For an 82-game season, the un-clipped form is approximately

\[
\widehat{\operatorname{Wins}}_t =
40.97 + 2.4805\,\widehat{\operatorname{NetRtg}}_t.
\]

The historical calibration's in-sample team win-total RMSE is 2.7446 wins.
That value measures only the NetRtg-to-wins relationship; the leaderboard
error below also includes error in the preseason NetRtg prediction itself.

| Model | Teams | Win-total RMSE | Win-total MAE | Win-percentage RMSE | Spearman correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | 30 | 11.5828 (#19) | 9.9905 (#19) | 0.1413 (#17) | 0.5274 (#18) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | 30 | 10.7999 (#15) | 9.2025 (#17) | 0.1317 (#14) | 0.6394 (#14) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | 30 | 10.7006 (#14) | 8.9333 (#15) | 0.1305 (#13) | 0.6238 (#16) |
| [Frozen aging prior][frozen-aging-prior] | 30 | 10.9632 (#16) | 9.6337 (#18) | 0.1337 (#15) | 0.6394 (#14) |
| [Frozen O/D RAPM][frozen-od-rapm] | 30 | 10.9640 (#17) | 8.9734 (#16) | 0.1337 (#15) | 0.6078 (#17) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | 30 | 10.6718 (#13) | 8.7037 (#13) | 0.1301 (#12) | 0.6245 (#15) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | 30 | 10.2466 (#9) | 8.0307 (#8) | 0.1250 (#8) | 0.6586 (#11) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | 30 | 10.5096 (#11) | 8.4409 (#11) | 0.1282 (#10) | 0.6483 (#13) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | 30 | 10.2778 (#10) | 8.4776 (#12) | 0.1253 (#9) | 0.6829 (#9) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | 30 | 10.6358 (#12) | 8.7390 (#14) | 0.1297 (#11) | 0.6608 (#10) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | 30 | 10.1176 (#8) | 8.3473 (#10) | 0.1234 (#7) | 0.6935 (#8) |
| [Forward contextual RAPM][forward-contextual-rapm] | 30 | 9.3153 (#3) | 7.0364 (#3) | 0.1136 (#3) | 0.7604 (#2) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | 30 | 11.0387 (#18) | 8.2281 (#9) | 0.1346 (#16) | 0.6548 (#12) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | 30 | 9.3999 (#6) | 7.1117 (#4) | 0.1146 (#5) | 0.7535 (#3) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | 30 | 9.3959 (#5) | 7.1510 (#5) | 0.1146 (#5) | 0.7361 (#6) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | 30 | 9.2772 (#2) | 7.3575 (#6) | 0.1131 (#2) | 0.7430 (#5) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 9.4360 (#7) | 7.5588 (#7) | 0.1151 (#6) | 0.7171 (#7) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | **7.7289 (#1)** | **6.3771 (#1)** | **0.0943 (#1)** | **0.8129 (#1)** |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | 30 | 9.3594 (#4) | 7.0082 (#2) | 0.1141 (#4) | 0.7521 (#4) |

As a diagnostic, the artifact also retains the raw count obtained by awarding
each game to the team with the positive predicted margin. That deterministic
rule has 14.5258 RMSE and 10.7333 MAE, confirming that it turns small predicted
edges into unrealistically extreme records. The raw count conserves exactly
1,230 league wins. Independently calibrated Pythagorean expectations are not
normalized to the target schedule after fitting: they total 1,230.9 wins for
the lagged baseline and 1,229.8 for the aging prior.

The age/draft/physical profile does capture part of the young-team signal. For
example, it moves San Antonio from 32.3 to 39.6 Pythagorean wins, but the actual
result was 62 wins. A smooth historical player-development prior cannot forecast
the largest discontinuous breakouts.

### Predicted Standings And Actual Results

The table is sorted by Pythagorean expected wins. Ranks are league-wide rather
than conference-specific because conference is not part of the current
team-season data contract.

| Pythagorean rank | Team | Pythagorean wins | Predicted NetRtg | Actual rank | Actual W-L | Actual NetRtg | Win error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | OKC | 65.6 | +9.92 | 1 | 64-18 | +11.11 | +1.6 |
| 2 | CLE | 59.8 | +7.60 | 8 | 52-30 | +4.11 | +7.8 |
| 3 | NYK | 56.1 | +6.11 | 6 | 53-29 | +6.50 | +3.1 |
| 4 | LAC | 55.5 | +5.86 | 18 | 42-40 | +1.17 | +13.5 |
| 5 | GSW | 55.0 | +5.66 | 20 | 37-45 | -0.56 | +18.0 |
| 6 | DEN | 49.5 | +3.43 | 5 | 54-28 | +5.15 | -4.5 |
| 7 | LAL | 48.3 | +2.96 | 6 | 53-29 | +1.78 | -4.7 |
| 8 | HOU | 47.0 | +2.42 | 8 | 52-30 | +5.36 | -5.0 |
| 9 | MIN | 46.6 | +2.28 | 10 | 49-33 | +3.32 | -2.4 |
| 10 | BOS | 46.3 | +2.16 | 4 | 56-26 | +8.10 | -9.7 |
| 11 | DET | 45.8 | +1.95 | 3 | 60-22 | +8.18 | -14.2 |
| 12 | ORL | 41.1 | +0.05 | 13 | 45-37 | +0.63 | -3.9 |
| 13 | POR | 40.7 | -0.09 | 18 | 42-40 | -0.29 | -1.3 |
| 14 | ATL | 40.1 | -0.33 | 11 | 46-36 | +2.37 | -5.9 |
| 15 | MIL | 39.2 | -0.73 | 21 | 32-50 | -6.34 | +7.2 |
| 16 | IND | 37.8 | -1.29 | 29 | 19-63 | -7.88 | +18.8 |
| 17 | PHX | 37.4 | -1.43 | 13 | 45-37 | +1.50 | -7.6 |
| 18 | CHI | 37.1 | -1.56 | 22 | 31-51 | -5.07 | +6.1 |
| 19 | TOR | 37.0 | -1.61 | 11 | 46-36 | +2.86 | -9.0 |
| 20 | PHI | 36.0 | -2.01 | 13 | 45-37 | -0.18 | -9.0 |
| 21 | MEM | 35.3 | -2.28 | 25 | 25-57 | -5.94 | +10.3 |
| 22 | SAC | 32.9 | -3.25 | 26 | 22-60 | -10.04 | +10.9 |
| 23 | MIA | 32.7 | -3.32 | 17 | 43-39 | +2.25 | -10.3 |
| 24 | SAS | 32.3 | -3.49 | 2 | 62-20 | +8.29 | -29.7 |
| 25 | NOP | 32.1 | -3.58 | 23 | 26-56 | -4.45 | +6.1 |
| 26 | DAL | 31.0 | -4.02 | 23 | 26-56 | -5.37 | +5.0 |
| 27 | UTA | 29.8 | -4.50 | 26 | 22-60 | -8.15 | +7.8 |
| 28 | BKN | 28.7 | -4.93 | 28 | 20-62 | -10.28 | +8.7 |
| 29 | CHA | 27.5 | -5.42 | 16 | 44-38 | +4.97 | -16.5 |
| 30 | WAS | 26.6 | -5.80 | 30 | 17-65 | -11.76 | +9.6 |

## Artifact

The promoted baseline run is
`frozen-lagged-prior-2025-26-20260805T011238Z-9ac7c011` under
`artifacts/models/frozen_prior_evaluation/2025-26/`. It contains player priors,
source-state declarations, possession and game predictions, team net-rating
and win tables, the historical Pythagorean calibration panel, all metric
tables, file hashes, and an MLflow index.

The frozen no-prior window controls are documented in
[Frozen No-Prior Window RAPM](frozen-window-rapm.md). Their one-year and
three-year fit/evaluation artifact IDs are recorded there with their selected
lambdas and training windows.

The evaluated age/draft/physical candidate is
`frozen-aging-prior-2025-26-20260805T013515Z-2fb4c418` in the same directory.
Its source state records both the 2024-25 reference lagged-RAPM run used for
the mean/home-court terms and the 2025-26 aging-prior artifact.

The O/D candidate is
`frozen-offense-defense-rapm-2025-26-20260805T050511Z-62b718bd` under
`artifacts/models/frozen_offense_defense_rapm/2025-26/`.

The completed combined box-score/cold-start audit is
`frozen-combined-box-score-prior-2025-26-20260805T163038Z-8d11217e` in the
frozen-prior artifact directory. Its source state pins combined run
`combined-box-score-prior-rapm-2025-26-20260804T215544Z-1b4294e3`, records the
selected zero box-score weight, and declares `prior_vector_equals_lagged: true`.

The draft-cold-start ablation is
`frozen-draft-cold-start-prior-2025-26-20260806T001516Z-bd7be959` in the same
frozen-prior directory. Its source state pins the draft study, records the
strict 2024-25 cutoff, and declares 462 `lagged_rapm`, 100
`draft_cold_start`, and 20 `zero_cold_start` player branches.

The exposure-gated cold-start evaluation is
`frozen-exposure-gated-cold-start-prior-2025-26-20260806T015348Z-089d2a36` in
the same directory. Its source state records the immutable draft-rate,
exposure-gate, and 2024-25-cutoff replacement-token inputs, and declares 462
`lagged_rapm`, 100 `exposure_gated_cold_start`, and 20 `zero_cold_start`
player branches.

The exposure-gated O/D cold-start evaluation is
`exposure-gated-od-2025-26-20260806T023559Z-4ffeb2bc` under
`artifacts/models/exposure_gated_offense_defense/2025-26/`. Its source state
pins the 2024-25 O/D source and exposure-gate artifacts, separately records
the 29-season offense and defense replacement-token means, and declares that
no 2025-26 outcomes formed the prior.

The Student-t recursive artifact is
`student-t-forward-rapm-2025-26-20260806T131834Z-b1cc6592` under
`artifacts/models/student_t_forward_rapm/2025-26/`. It records the frozen
player vector, all season-level IRLS diagnostics, and the same possession,
game, team NetRtg, and Pythagorean-win files as the Gaussian recursive model.

The Student-t talent-prior artifact is
`student-t-talent-forward-rapm-2025-26-20260806T203508Z-8e8a2698` under
`artifacts/models/student_t_talent_forward_rapm/2025-26/`. It records the
Gaussian observation contract, Student-t coefficient-prior settings,
season-level convergence diagnostics, and the full frozen evaluation output.

[frozen-one-year-no-prior-rapm]: frozen-window-rapm.md "Zero-centered 2024-25 ridge RAPM with no player prior."
[frozen-three-year-no-prior-rapm]: frozen-window-rapm.md "One zero-centered ridge coefficient fitted across 2022-23 through 2024-25."
[frozen-lagged-rapm]: prior-rapm.md "Completed prior-season RAPM used as the frozen player prior."
[frozen-aging-prior]: aging-model.md "Lagged RAPM augmented by a forward age, experience, draft, and physical-profile prior."
[frozen-od-rapm]: offense-defense-rapm.md "Separate frozen offensive and defensive player ratings."
[frozen-draft-cold-start]: draft-prior.md "Draft profile replaces the zero prior for first-year players."
[frozen-exposure-gated-cold-start]: exposure-gated-cold-start.md "Draft-rate and replacement-token cold starts blended by predicted exposure risk."
[frozen-exposure-gated-od]: exposure-gated-offense-defense.md "Separate offense and defense cold starts with exposure-gated replacement tokens."
[recursive-exposure-gated-rapm]: forward-exposure-gated-rapm.md "Leakage-safe exposure-gated player state rebuilt one season at a time."
[student-t-recursive-rapm]: student-t-forward-rapm.md "Recursive exposure-gated RAPM with Student-t stint errors."
[student-t-talent-prior]: student-t-talent-forward-rapm.md "Recursive RAPM with a heavy-tailed player-prior departure penalty."
[forward-contextual-rapm]: forward-contextual-rapm.md "Recursive RAPM plus a lagged nonlinear lineup-composition offset."
[forward-decomposed-contextual-rapm]: forward-decomposed-contextual-rapm.md "Interpretability ablation with an identifiable home-side minus away-side contextual offset."
[forward-portable-matchup-contextual-rapm]: forward-portable-matchup-contextual-rapm.md "Antisymmetric context decomposed into portable unit scores and opponent-specific matchup residuals."
[forward-hierarchical-pspline-contextual-rapm]: forward-hierarchical-pspline-contextual-rapm.md "Portable-matchup context with P-spline curvature regularization and a projected prior-season function hierarchy."
[forward-bounded-hierarchical-pspline-contextual-rapm]: forward-bounded-hierarchical-pspline-contextual-rapm.md "Relative context with 5th-95th percentile bounds, P-spline curvature regularization, and a projected prior-season function hierarchy."
[forward-bounded-hierarchical-portable-matchup-contextual-rapm]: forward-bounded-hierarchical-portable-matchup-contextual-rapm.md "Portable composition and matchup context with bounded support, P-spline curvature regularization, and a projected prior-season function hierarchy."
[forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm]: forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md "Bounded portable context with a recursively fit age-informed returning-player prior and exposure-gated cold starts."
[student-t-talent-contextual-rapm]: student-t-talent-contextual-rapm.md "Contextual RAPM with heavy-tailed player-prior departures."
