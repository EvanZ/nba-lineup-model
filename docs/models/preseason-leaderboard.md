---
last_updated: "2026-08-13"
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

The recovered-coverage, multi-target RAPM controls are maintained separately in
the [Three-Season Frozen Leaderboard](three-season-frozen-backtest.md). They are
the current rigorous forward evaluation baseline; this page remains a
single-2025-26 snapshot while the remaining model families are rerun against
that shared contract.

> **Rebuild status (2026-08-13):** the historical raw-to-stint recovery
> changed the source data contract for the forward priors. The tables below are
> retained as the pre-recovery comparison snapshot until every listed model is
> rerun. The rebuilt Value-Conditioned Aging HPM, HPM v2 depth-aware shooting
> ablation, and Box-Score Residual HPM are documented on their model pages;
> their numbers are not ranked against these stale-chain rows.

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
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | Regular season | 1,230 | 218,810 | 1.199061 (#23) | 1.142303 (#23) | 15.0694 (#26) | 0.0704% (#23) | 9.4406% (#26) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | Regular season | 1,230 | 218,810 | 1.198919 (#13) | 1.142004 (#18) | 14.7820 (#17) | 0.0941% (#13) | 12.8614% (#17) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | Regular season | 1,230 | 218,810 | 1.199000 (#18) | 1.142154 (#21) | 14.8894 (#21) | 0.0805% (#18) | 11.5905% (#21) |
| [Frozen aging prior][frozen-aging-prior] | Regular season | 1,230 | 218,810 | 1.199062 (#24) | 1.142736 (#26) | 15.0203 (#25) | 0.0702% (#24) | 10.0297% (#25) |
| [Frozen O/D RAPM][frozen-od-rapm] | Regular season | 1,230 | 218,810 | 1.198853 (#9) | 1.142664 (#25) | 14.8901 (#22) | 0.1092% (#8) | 11.5692% (#23) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | Regular season | 1,230 | 218,810 | 1.198986 (#15) | 1.142088 (#20) | 14.8590 (#20) | 0.0829% (#15) | 11.9512% (#20) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | Regular season | 1,230 | 218,810 | 1.198952 (#14) | 1.141943 (#15) | 14.7413 (#15) | 0.0886% (#14) | 13.3410% (#15) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | Regular season | 1,230 | 218,810 | 1.198792 (#5) | 1.142422 (#24) | 14.7631 (#16) | 0.1193% (#2) | 13.0714% (#16) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | Regular season | 1,230 | 218,810 | 1.198993 (#17) | 1.142055 (#19) | 14.8225 (#19) | 0.0817% (#17) | 12.3830% (#19) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | Regular season | 1,230 | 218,810 | 1.199014 (#20) | 1.142242 (#22) | 14.8908 (#23) | 0.0782% (#20) | 11.5747% (#22) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | Regular season | 1,230 | 218,810 | 1.198989 (#16) | 1.141996 (#17) | 14.7993 (#18) | 0.0825% (#16) | 12.6578% (#18) |
| [Forward contextual RAPM][forward-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199008 (#19) | 1.141571 (#2) | 14.6525 (#13) | 0.0793% (#19) | 14.3817% (#13) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199259 (#25) | 1.141636 (#4) | 14.9769 (#24) | 0.0373% (#25) | 10.5488% (#24) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199030 (#22) | 1.141625 (#3) | 14.6349 (#12) | 0.0755% (#22) | 14.5870% (#12) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198859 (#10) | 1.141648 (#5) | 14.5410 (#9) | 0.1041% (#10) | 15.6803% (#9) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198894 (#11) | 1.141695 (#8) | 14.5485 (#10) | 0.0982% (#11) | 15.5932% (#10) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198917 (#12) | 1.141735 (#11) | 14.5953 (#11) | 0.0945% (#12) | 15.0483% (#11) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198831 (#8) | 1.141770 (#12) | 14.3654 (#6) | 0.1088% (#9) | 17.7044% (#6) |
| [Forward centered aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198792 (#5) | 1.141805 (#13) | 14.3615 (#5) | 0.1153% (#5) | 17.7490% (#5) |
| [Value-Conditioned Aging HPM][forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | **1.198763 (#1)** | 1.141698 (#9) | **14.3469 (#1)** | **0.1201% (#1)** | **17.9158% (#1)** |
| [Controlled no-context value-conditioned RAPM][forward-centered-value-conditioned-aging-no-context-rapm] | Regular season | 1,230 | 218,810 | 1.198800 (#7) | 1.141973 (#16) | 14.4991 (#8) | 0.1139% (#7) | 16.1647% (#8) |
| [Forward context-reattributed HPM (rho=0.1)][forward-context-reattributed-hpm] | Regular season | 1,230 | 218,810 | 1.198768 (#2) | 1.141706 (#10) | 14.3543 (#2) | 0.1193% (#2) | 17.8308% (#2) |
| [Forward box-score residual HPM][forward-box-score-residual-hpm] | Regular season | 1,230 | 218,810 | 1.198770 (#3) | 1.141668 (#7) | 14.3548 (#3) | 0.1189% (#3) | 17.8248% (#3) |
| [Forward box-score interaction HPM][forward-box-score-interaction-hpm] | Regular season | 1,230 | 218,810 | 1.198771 (#4) | 1.141664 (#6) | 14.3577 (#4) | 0.1187% (#4) | 17.7920% (#4) |
| [Era-Conditioned Aging HPM][forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.198799 (#6) | 1.141806 (#14) | 14.4398 (#7) | 0.1141% (#6) | 16.8492% (#7) |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | Regular season | 1,230 | 218,810 | 1.199026 (#21) | **1.141556 (#1)** | 14.6770 (#14) | 0.0762% (#21) | 14.0952% (#14) |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | Playoffs | 85 | 14,253 | 1.192713 (#11) | 1.136139 (#17) | 17.1867 (#12) | -0.0468% (#12) | -5.2607% (#12) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | Playoffs | 85 | 14,253 | 1.192614 (#9) | 1.135787 (#4) | 16.9988 (#10) | -0.0302% (#10) | -2.9720% (#10) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | Playoffs | 85 | 14,253 | 1.192895 (#15) | 1.136163 (#19) | 17.5409 (#19) | -0.0774% (#16) | -9.6446% (#19) |
| [Frozen aging prior][frozen-aging-prior] | Playoffs | 85 | 14,253 | **1.192332 (#1)** | 1.136455 (#23) | 16.4946 (#4) | 0.0170% (#2) | 3.0460% (#4) |
| [Frozen O/D RAPM][frozen-od-rapm] | Playoffs | 85 | 14,253 | 1.194642 (#23) | 1.139203 (#24) | 17.8037 (#24) | -0.3924% (#24) | -12.9804% (#24) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | Playoffs | 85 | 14,253 | 1.192971 (#19) | 1.136187 (#21) | 17.6339 (#22) | -0.0901% (#20) | -10.8093% (#22) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | Playoffs | 85 | 14,253 | 1.192980 (#20) | 1.136174 (#20) | 17.6725 (#23) | -0.0917% (#21) | -11.2952% (#23) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | Playoffs | 85 | 14,253 | 1.194678 (#24) | 1.139209 (#25) | 17.8562 (#25) | -0.3984% (#25) | -13.6482% (#25) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | Playoffs | 85 | 14,253 | 1.192821 (#12) | 1.136096 (#13) | 17.4188 (#14) | -0.0649% (#13) | -8.1231% (#14) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | Playoffs | 85 | 14,253 | 1.192621 (#10) | 1.136193 (#22) | 17.1300 (#11) | -0.0314% (#11) | -4.5672% (#11) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | Playoffs | 85 | 14,253 | 1.192895 (#15) | 1.136105 (#15) | 17.4713 (#16) | -0.0774% (#16) | -8.7752% (#16) |
| [Forward contextual RAPM][forward-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192898 (#16) | **1.135625 (#1)** | 17.5493 (#20) | -0.0778% (#17) | -9.7486% (#20) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | Playoffs | 85 | 14,253 | 1.193395 (#22) | 1.136131 (#16) | 18.4522 (#26) | -0.1613% (#23) | -21.3327% (#26) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192994 (#21) | 1.135701 (#3) | 17.5261 (#18) | -0.0939% (#22) | -9.4596% (#18) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192857 (#13) | 1.136067 (#11) | 17.4003 (#13) | -0.0710% (#14) | -7.8941% (#13) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192872 (#14) | 1.136002 (#9) | 17.4604 (#15) | -0.0736% (#15) | -8.6406% (#15) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192935 (#17) | 1.136059 (#10) | 17.4742 (#17) | -0.0840% (#18) | -8.8115% (#17) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192544 (#8) | 1.136145 (#18) | 16.5388 (#8) | -0.0185% (#9) | 2.5259% (#8) |
| [Forward centered aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192464 (#3) | 1.136093 (#12) | 16.5018 (#5) | -0.0051% (#4) | 2.9615% (#5) |
| [Value-Conditioned Aging HPM][forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192460 (#2) | 1.135968 (#7) | 16.4605 (#2) | -0.0043% (#3) | 3.4461% (#2) |
| [Controlled no-context value-conditioned RAPM][forward-centered-value-conditioned-aging-no-context-rapm] | Playoffs | 85 | 14,253 | **1.192332 (#1)** | 1.135983 (#8) | **16.2971 (#1)** | **0.0172% (#1)** | **5.3543% (#1)** |
| [Forward context-reattributed HPM (rho=0.1)][forward-context-reattributed-hpm] | Playoffs | 85 | 14,253 | 1.192470 (#4) | 1.135963 (#6) | 16.4701 (#3) | -0.0060% (#5) | 3.3333% (#3) |
| [Forward box-score residual HPM][forward-box-score-residual-hpm] | Playoffs | 85 | 14,253 | 1.192508 (#6) | 1.135960 (#5) | 16.5027 (#6) | -0.0124% (#7) | 2.9510% (#6) |
| [Forward box-score interaction HPM][forward-box-score-interaction-hpm] | Playoffs | 85 | 14,253 | 1.192521 (#7) | 1.135963 (#6) | 16.5138 (#7) | -0.0146% (#8) | 2.8199% (#7) |
| [Era-Conditioned Aging HPM][forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192475 (#5) | 1.136097 (#14) | 16.5525 (#9) | -0.0069% (#6) | 2.3637% (#9) |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | Playoffs | 85 | 14,253 | 1.192959 (#18) | 1.135643 (#2) | 17.6146 (#21) | -0.0881% (#19) | -10.5673% (#21) |

Bolding marks the better value within each cohort and metric. Future frozen
priors must use these exact cohorts.

## Full-Game Outcomes

<!-- frozen-full-game-outcomes:start -->
Full-game outcomes aggregate every allocated regular-season stint to the official final home margin. This is deliberately distinct from the eligible-possession game
metric above, which excludes possession rows without one reconstructed lineup. Winner accuracy calls the sign of that full-game margin; an exactly zero forecast
receives half credit rather than being arbitrarily assigned to the away team.

| Model | Games | Full-game margin RMSE | Full-game margin MAE | Winner accuracy | Predicted ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | 1,230 | 15.5605 | 12.3241 | 63.17% | 0 |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | 1,230 | 15.1944 | 12.0436 | 65.20% | 0 |
| [Frozen lagged RAPM][frozen-lagged-rapm] | 1,230 | 15.3211 | 12.1338 | 65.69% | 0 |
| [Frozen aging prior][frozen-aging-prior] | 1,230 | 15.5090 | 12.2839 | 65.04% | 0 |
| [Frozen O/D RAPM][frozen-od-rapm] | 1,230 | 15.4015 | 12.2653 | 65.04% | 0 |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | 1,230 | 15.3025 | 12.1092 | 65.12% | 0 |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | 1,230 | 15.1594 | 11.9812 | 65.77% | 0 |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | 1,230 | 15.2474 | 12.1289 | 65.28% | 0 |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | 1,230 | 15.1802 | 12.0162 | 66.67% | 0 |
| [Student-t recursive RAPM][student-t-recursive-rapm] | 1,230 | 15.2758 | 12.1226 | 65.69% | 0 |
| [Student-t talent-prior RAPM][student-t-talent-prior] | 1,230 | 15.1499 | 11.9941 | 67.56% | 0 |
| [Forward contextual RAPM][forward-contextual-rapm] | 1,230 | 15.0190 | 11.8474 | 67.89% | 0 |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | 1,230 | 15.3555 | 12.1046 | 67.48% | 0 |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | 1,230 | 14.9602 | 11.7780 | 67.72% | 0 |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | 1,230 | 14.8516 | 11.7498 | 67.64% | 0 |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | 1,230 | 14.8706 | 11.7602 | 67.80% | 0 |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | 14.9224 | 11.8112 | 67.40% | 0 |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | 14.6265 | 11.5301 | 67.40% | 0 |
| [Forward centered aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | 14.6334 | 11.5535 | 68.13% | 0 |
| [Forward centered value-conditioned aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | **14.5898** | **11.5110** | 68.46% | 0 |
| [HIPSTER PM v2: Depth-Aware Shooting][hpm-v2-depth-aware-shooting] | 1,230 | 14.6164 | 11.5359 | 67.97% | 0 |
| [Controlled no-context value-conditioned aging RAPM][forward-centered-value-conditioned-aging-no-context-rapm] | 1,230 | 14.8026 | 11.6727 | 68.21% | 0 |
| [Forward context-reattributed HPM (rho=0.1)][forward-context-reattributed-hpm] | 1,230 | 14.6344 | 11.5522 | 68.21% | 0 |
| [Forward box-score residual HPM][forward-box-score-residual-hpm] | 1,230 | 14.5910 | 11.5112 | **68.78%** | 0 |
| [Forward box-score interaction HPM][forward-box-score-interaction-hpm] | 1,230 | 14.6309 | 11.5534 | 68.05% | 0 |
| [Forward centered era-conditioned aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 1,230 | 14.7481 | 11.6446 | 67.97% | 0 |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | 1,230 | 15.0487 | 11.8634 | 67.32% | 0 |

Source report: `artifacts/models/frozen_game_outcomes/2025-26/frozen_game_outcomes-2025-26-20260813T175418Z-d9649330`. The retained `game_outcome_predictions.parquet`
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
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | 30 | 4.9792 (#25) | 4.2342 (#25) | 0.5890 (#25) | 0.5444 (#26) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | 30 | 4.6618 (#16) | 4.0620 (#24) | 0.6490 (#17) | 0.6271 (#17) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | 30 | 4.8538 (#23) | 3.9606 (#22) | 0.6113 (#24) | 0.5493 (#25) |
| [Frozen aging prior][frozen-aging-prior] | 30 | 5.0366 (#26) | 4.2395 (#26) | 0.6484 (#18) | 0.6111 (#20) |
| [Frozen O/D RAPM][frozen-od-rapm] | 30 | 4.8740 (#24) | 3.9380 (#21) | 0.6113 (#24) | 0.5626 (#23) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | 30 | 4.8473 (#22) | 3.8653 (#20) | 0.6163 (#22) | 0.5528 (#24) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | 30 | 4.6989 (#18) | 3.6020 (#15) | 0.6480 (#19) | 0.5844 (#22) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | 30 | 4.7113 (#19) | 3.6999 (#17) | 0.6454 (#21) | 0.6018 (#21) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | 30 | 4.6680 (#17) | 3.8199 (#19) | 0.6471 (#20) | 0.6236 (#18) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | 30 | 4.8263 (#21) | 3.9928 (#23) | 0.6159 (#23) | 0.6147 (#19) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | 30 | 4.6069 (#15) | 3.7344 (#18) | 0.6587 (#16) | 0.6383 (#15) |
| [Forward contextual RAPM][forward-contextual-rapm] | 30 | 4.1572 (#9) | 3.1821 (#8) | 0.7427 (#9) | 0.7219 (#9) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | 30 | 4.7470 (#20) | 3.6765 (#16) | 0.6822 (#15) | 0.6320 (#16) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | 30 | 4.1864 (#12) | 3.1994 (#9) | 0.7394 (#11) | 0.7201 (#10) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | 30 | 4.1883 (#13) | 3.2466 (#11) | 0.7354 (#12) | 0.7041 (#12) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | 30 | 4.1799 (#11) | 3.3077 (#13) | 0.7338 (#13) | 0.6974 (#13) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 4.2546 (#14) | 3.3952 (#14) | 0.7222 (#14) | 0.6774 (#14) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 3.7737 (#4) | 3.0395 (#3) | 0.7920 (#2) | 0.7562 (#6) |
| [Forward centered aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 3.7826 (#5) | 3.0953 (#6) | **0.7921 (#1)** | 0.7682 (#2) |
| [Value-Conditioned Aging HPM][forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 3.7568 (#2) | 3.0401 (#4) | 0.7912 (#3) | **0.7744 (#1)** |
| [Controlled no-context value-conditioned RAPM][forward-centered-value-conditioned-aging-no-context-rapm] | 30 | 4.0654 (#8) | 3.2645 (#12) | 0.7664 (#8) | 0.7428 (#8) |
| [Forward context-reattributed HPM (rho=0.1)][forward-context-reattributed-hpm] | 30 | 3.7827 (#6) | 3.0521 (#5) | 0.7879 (#6) | 0.7615 (#4) |
| [Forward box-score residual HPM][forward-box-score-residual-hpm] | 30 | **3.7536 (#1)** | **3.0236 (#1)** | 0.7910 (#4) | 0.7633 (#3) |
| [Forward box-score interaction HPM][forward-box-score-interaction-hpm] | 30 | 3.7595 (#3) | 3.0240 (#2) | 0.7901 (#5) | 0.7531 (#7) |
| [Era-Conditioned Aging HPM][forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 3.9402 (#7) | 3.2381 (#10) | 0.7677 (#7) | 0.7584 (#5) |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | 30 | 4.1771 (#10) | 3.1699 (#7) | 0.7412 (#10) | 0.7077 (#11) |

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
| [Frozen 1-year no-prior RAPM][frozen-one-year-no-prior-rapm] | 30 | 11.5828 (#26) | 9.9905 (#26) | 0.1413 (#23) | 0.5274 (#25) |
| [Frozen pooled 3-year no-prior RAPM][frozen-three-year-no-prior-rapm] | 30 | 10.7999 (#22) | 9.2025 (#24) | 0.1317 (#20) | 0.6394 (#21) |
| [Frozen lagged RAPM][frozen-lagged-rapm] | 30 | 10.7006 (#21) | 8.9333 (#22) | 0.1305 (#19) | 0.6238 (#23) |
| [Frozen aging prior][frozen-aging-prior] | 30 | 10.9632 (#23) | 9.6337 (#25) | 0.1337 (#21) | 0.6394 (#21) |
| [Frozen O/D RAPM][frozen-od-rapm] | 30 | 10.9640 (#24) | 8.9734 (#23) | 0.1337 (#21) | 0.6078 (#24) |
| [Frozen draft cold-start prior][frozen-draft-cold-start] | 30 | 10.6718 (#20) | 8.7037 (#20) | 0.1301 (#18) | 0.6245 (#22) |
| [Frozen exposure-gated cold-start prior][frozen-exposure-gated-cold-start] | 30 | 10.2466 (#16) | 8.0307 (#15) | 0.1250 (#14) | 0.6586 (#18) |
| [Frozen exposure-gated O/D cold-start prior][frozen-exposure-gated-od] | 30 | 10.5096 (#18) | 8.4409 (#18) | 0.1282 (#16) | 0.6483 (#20) |
| [Recursive exposure-gated RAPM][recursive-exposure-gated-rapm] | 30 | 10.2778 (#17) | 8.4776 (#19) | 0.1253 (#15) | 0.6829 (#16) |
| [Student-t recursive RAPM][student-t-recursive-rapm] | 30 | 10.6358 (#19) | 8.7390 (#21) | 0.1297 (#17) | 0.6608 (#17) |
| [Student-t talent-prior RAPM][student-t-talent-prior] | 30 | 10.1176 (#15) | 8.3473 (#17) | 0.1234 (#13) | 0.6935 (#15) |
| [Forward contextual RAPM][forward-contextual-rapm] | 30 | 9.3153 (#10) | 7.0364 (#10) | 0.1136 (#9) | 0.7604 (#9) |
| [Forward decomposed contextual RAPM][forward-decomposed-contextual-rapm] | 30 | 11.0387 (#25) | 8.2281 (#16) | 0.1346 (#22) | 0.6548 (#19) |
| [Forward portable-matchup contextual RAPM][forward-portable-matchup-contextual-rapm] | 30 | 9.3999 (#13) | 7.1117 (#11) | 0.1146 (#11) | 0.7535 (#10) |
| [Forward hierarchical P-spline contextual RAPM][forward-hierarchical-pspline-contextual-rapm] | 30 | 9.3959 (#12) | 7.1510 (#12) | 0.1146 (#11) | 0.7361 (#13) |
| [Forward bounded hierarchical P-spline contextual RAPM][forward-bounded-hierarchical-pspline-contextual-rapm] | 30 | 9.2772 (#9) | 7.3575 (#13) | 0.1131 (#8) | 0.7430 (#12) |
| [Forward bounded hierarchical portable-matchup contextual RAPM][forward-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 9.4360 (#14) | 7.5588 (#14) | 0.1151 (#12) | 0.7171 (#14) |
| [Forward aging bounded hierarchical portable-matchup contextual RAPM][forward-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | **7.7289 (#1)** | **6.3771 (#1)** | **0.0943 (#1)** | **0.8129 (#1)** |
| [Forward centered aging bounded hierarchical portable-matchup contextual RAPM][forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 7.9356 (#2) | 6.4760 (#6) | 0.0968 (#2) | 0.8109 (#3) |
| [Value-Conditioned Aging HPM][forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 8.0346 (#5) | 6.4224 (#4) | 0.0980 (#4) | 0.8123 (#2) |
| [Controlled no-context value-conditioned RAPM][forward-centered-value-conditioned-aging-no-context-rapm] | 30 | 8.6225 (#8) | 7.0174 (#9) | 0.1052 (#7) | 0.7797 (#8) |
| [Forward context-reattributed HPM (rho=0.1)][forward-context-reattributed-hpm] | 30 | 8.0940 (#6) | 6.4157 (#3) | 0.0987 (#5) | 0.8033 (#5) |
| [Forward box-score residual HPM][forward-box-score-residual-hpm] | 30 | 7.9943 (#4) | 6.4144 (#2) | 0.0975 (#3) | 0.8056 (#4) |
| [Forward box-score interaction HPM][forward-box-score-interaction-hpm] | 30 | 7.9928 (#3) | 6.4281 (#5) | 0.0975 (#3) | 0.8013 (#6) |
| [Era-Conditioned Aging HPM][forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm] | 30 | 8.3940 (#7) | 6.9362 (#7) | 0.1024 (#6) | 0.7969 (#7) |
| [Student-t talent-prior contextual RAPM][student-t-talent-contextual-rapm] | 30 | 9.3594 (#11) | 7.0082 (#8) | 0.1141 (#10) | 0.7521 (#11) |

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

The centered aging portable candidate is
`forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm-2025-26-20260810T060030Z-eff5e2d3`.
The Value-Conditioned Aging HPM candidate is
`forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm-2025-26-20260810T142633Z-3e4dd5da`.
The Era-Conditioned Aging HPM candidate is
`forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm-2025-26-20260810T154501Z-76a5fad1`.
All three artifacts are under their respective `artifacts/models/` model
directories and use only prior-season inputs for every target-season forecast.

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
[forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm]: forward-centered-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md "Age-informed portable context with a prior-season possession-weighted player-prior reference."
[forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm]: forward-centered-value-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md "Centered aging HPM with a learned, ridge-penalized age-by-prior-RAPM interaction."
[hpm-v2-depth-aware-shooting]: hpm-v2.md "A controlled HPM shooting-context ablation replacing raw lineup totals with depth, capped capacity, and concentration features."
[forward-centered-value-conditioned-aging-no-context-rapm]: forward-centered-value-conditioned-aging-no-context-rapm.md "Controlled ablation of Value-Conditioned Aging HPM with the recursive context state fixed at zero."
[forward-context-reattributed-hpm]: forward-context-reattributed-hpm.md "Experimental HPM candidate that transfers rho=0.1 of the prior season's player-projectable context into the next player prior and leaves the remainder contextual."
[forward-box-score-residual-hpm]: forward-box-score-residual-hpm.md "Value-Conditioned Aging HPM plus a strictly lagged, possession-native box-score residual for returning players."
[forward-box-score-interaction-hpm]: forward-box-score-interaction-hpm.md "The box-score residual HPM augmented with six predeclared, lagged player-profile interactions."
[forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm]: forward-centered-era-conditioned-aging-bounded-hierarchical-portable-matchup-contextual-rapm.md "Centered value-conditioned aging HPM with a ridge-penalized age-spline-by-era interaction."
[student-t-talent-contextual-rapm]: student-t-talent-contextual-rapm.md "Contextual RAPM with heavy-tailed player-prior departures."
