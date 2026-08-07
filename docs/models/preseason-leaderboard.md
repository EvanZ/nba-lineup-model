# Frozen Preseason Leaderboard

This leaderboard evaluates models as true preseason forecasts. Every player
value is frozen before the target season begins. Target-season lineups and
exposure are supplied by an oracle, but no target-season score, possession
outcome, fitted player adjustment, or playoff result can change the model.

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

### Frozen Aging Candidate

The first candidate keeps the same scoring equation, league mean, home-court
term, target rows, and oracle lineup exposure. It replaces only the frozen
player-prior vector with the forward aging ridge model trained through 2024-25.
That model uses preseason age, NBA experience, prior RAPM and exposure, draft
profile, height, weight, and age-by-profile interactions. It produces the same
582-player 2025-26 coverage as the lagged-RAPM baseline and has no 2025-26
outcome fields.

### Frozen Offense/Defense Candidate

The O/D candidate uses two weighted offensive-rating rows per lineup stint:
the offense lineup enters offensive columns and the opponent lineup enters
defensive columns. It fits forward regular-only O/D states from 1996-97 through
2024-25, then freezes the completed 2024-25 state before scoring 2025-26.
See [Offense/Defense RAPM](offense-defense-rapm.md) for the exact equations and
identification convention.

### Frozen Combined Box-Score Candidate

The existing complete box-score/cold-start prior was also evaluated under this
strict frozen contract. Its earlier in-season lineup-level selection chose a
box-score blend weight of zero. Consequently, its 582 published player priors
are exactly equal to the frozen lagged-RAPM vector. The dedicated frozen run
confirms exact equality across regular-season and playoff possession metrics,
team NetRtg, and Pythagorean wins. It is therefore not repeated in the primary
metric tables as a distinct candidate.

### Frozen Draft Cold-Start Candidate

The draft-cold-start candidate keeps the completed 2024-25 lagged-RAPM vector
for 462 returning players and the neutral zero prior for 20 other no-prior
players. It replaces only 100 first-NBA-season zero priors with the
draft-profile ridge estimates fit through 2024-25. This is a direct frozen
ablation, not a recursive historical RAPM refit. It therefore isolates whether
the draft profile improves the zero cold-start default on the exact same oracle
lineups and exposure. The current specification includes a normalized-pick by
historical-median-centered draft-age interaction.

### Frozen Exposure-Gated Cold-Start Candidate

The exposure-gated candidate keeps the same 462 returning-player lagged RAPM
values and 20 zero-prior players. For the 100 first-NBA-season players, it
continuously blends the frozen draft-rate estimate with the pooled 1996-97 to
2024-25 replacement-token estimate using the player-specific probability of
finishing below 5% of team possession opportunities. See
[Exposure-Gated Cold-Start Prior](exposure-gated-cold-start.md) for the
component contract and revised rookie rankings.

### Frozen Exposure-Gated O/D Cold-Start Candidate

This candidate keeps the frozen 2024-25 O/D state for returning players. For
the same 100 first-NBA-season players, it replaces missing O/D values with
separately blended offense and defense draft-rate/replacement-token priors.
The first-year exposure-gate probability is shared across sides, but the rate
and replacement components are independently estimated by side. See
[Exposure-Gated O/D Cold Starts](exposure-gated-offense-defense.md).

### Recursive Exposure-Gated Candidate

This candidate rebuilds the one-number exposure-gated state season by season,
using only earlier regular seasons for every historical cold-start component.
It is the production-state candidate, while the simpler frozen blend remains
the regular-season metric leader on this single target holdout. See [Forward
Exposure-Gated RAPM](forward-exposure-gated-rapm.md).

### Student-t Recursive Candidate

The Student-t candidate preserves the recursive exposure-gated forward state,
cold-start policy, and per-season ridge lambdas. It changes only the
regular-season stint-error likelihood from Gaussian to Student-t with five
degrees of freedom, solved by IRLS. This is a controlled robustness ablation;
see [Student-t Forward RAPM](student-t-forward-rapm.md).

### Student-t Talent-Prior Candidate

This candidate restores Gaussian stint errors and keeps the recursive
exposure-gated state and lambda schedule fixed. Only player departures from
their forward prior receive a Student-t \(\nu=3\) penalty, with a three-point
RAPM tail scale. See [Student-t Talent-Prior RAPM](student-t-talent-forward-rapm.md).

## Possession And Game Results

Regular season and playoffs are evaluated separately. Possession metrics use
only possessions with one reconstructed lineup, matching the existing neural
evaluation boundary. Game-margin RMSE aggregates those eligible possessions in
the home-team frame.

| Model | Cohort | Games | Possessions | Possession RMSE | Possession MAE | Game-margin RMSE | Possession skill vs frozen mean | Game skill vs frozen mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen lagged RAPM | Regular season | 1,230 | 218,810 | 1.199000 | 1.142154 | 14.8894 | 0.0805% | 11.5905% |
| Frozen aging prior | Regular season | 1,230 | 218,810 | 1.199062 | 1.142736 | 15.0203 | 0.0702% | 10.0297% |
| Frozen O/D RAPM | Regular season | 1,230 | 218,810 | **1.198853** | 1.142664 | 14.8901 | **0.1092%** | 11.5692% |
| Frozen draft cold-start prior | Regular season | 1,230 | 218,810 | 1.198986 | 1.142088 | 14.8590 | 0.0829% | 11.9512% |
| Frozen exposure-gated cold-start prior | Regular season | 1,230 | 218,810 | 1.198952 | **1.141943** | **14.7413** | 0.0886% | **13.3410%** |
| Frozen exposure-gated O/D cold-start prior | Regular season | 1,230 | 218,810 | **1.198792** | 1.142422 | 14.7631 | **0.1193%** | 13.0714% |
| Recursive exposure-gated RAPM | Regular season | 1,230 | 218,810 | 1.198993 | 1.142055 | 14.8225 | 0.0817% | 12.3830% |
| Student-t recursive RAPM | Regular season | 1,230 | 218,810 | 1.199014 | 1.142242 | 14.8908 | 0.0782% | 11.5747% |
| Student-t talent-prior RAPM | Regular season | 1,230 | 218,810 | 1.198989 | 1.141996 | 14.7993 | 0.0825% | 12.6578% |
| Frozen lagged RAPM | Playoffs | 85 | 14,253 | 1.192895 | **1.136163** | 17.5409 | -0.0774% | -9.6446% |
| Frozen aging prior | Playoffs | 85 | 14,253 | **1.192332** | 1.136455 | **16.4946** | **0.0170%** | **3.0460%** |
| Frozen O/D RAPM | Playoffs | 85 | 14,253 | 1.194642 | 1.139203 | 17.8037 | -0.3924% | -12.9804% |
| Frozen draft cold-start prior | Playoffs | 85 | 14,253 | 1.192971 | 1.136187 | 17.6339 | -0.0901% | -10.8093% |
| Frozen exposure-gated cold-start prior | Playoffs | 85 | 14,253 | 1.192980 | 1.136174 | 17.6725 | -0.0917% | -11.2952% |
| Frozen exposure-gated O/D cold-start prior | Playoffs | 85 | 14,253 | 1.194678 | 1.139209 | 17.8562 | -0.3984% | -13.6482% |
| Recursive exposure-gated RAPM | Playoffs | 85 | 14,253 | 1.192821 | **1.136096** | 17.4188 | -0.0649% | -8.1231% |
| Student-t recursive RAPM | Playoffs | 85 | 14,253 | 1.192621 | 1.136193 | 17.1300 | -0.0314% | -4.5672% |
| Student-t talent-prior RAPM | Playoffs | 85 | 14,253 | 1.192895 | 1.136105 | 17.4713 | -0.0774% | -8.7752% |

Bolding marks the better value within each cohort and metric. Future frozen
priors must use these exact cohorts.

## Team Net Rating

Team net rating uses all regular-season RAPM stints, including allocated
multi-lineup possessions. For stint (s), the frozen lineup prediction is
converted to a predicted margin using the stint's oracle possession exposure;
team margins are then summed and divided by team possessions.

| Model | Teams | Net-rating RMSE | Net-rating MAE | Pearson correlation | Spearman correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen lagged RAPM | 30 | 4.8538 | 3.9606 | 0.6113 | 0.5493 |
| Frozen aging prior | 30 | 5.0366 | 4.2395 | 0.6484 | 0.6111 |
| Frozen O/D RAPM | 30 | 4.8740 | 3.9380 | 0.6113 | 0.5626 |
| Frozen draft cold-start prior | 30 | 4.8473 | 3.8653 | 0.6163 | 0.5528 |
| Frozen exposure-gated cold-start prior | 30 | 4.6989 | **3.6020** | 0.6480 | 0.5844 |
| Frozen exposure-gated O/D cold-start prior | 30 | 4.7113 | 3.6999 | 0.6454 | 0.6018 |
| Recursive exposure-gated RAPM | 30 | 4.6680 | 3.8199 | 0.6471 | 0.6236 |
| Student-t recursive RAPM | 30 | 4.8263 | 3.9928 | 0.6159 | 0.6147 |
| Student-t talent-prior RAPM | 30 | **4.6069** | 3.7344 | **0.6587** | **0.6383** |

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
| Frozen lagged RAPM | 30 | 10.7006 | 8.9333 | 0.1305 | 0.6238 |
| Frozen aging prior | 30 | 10.9632 | 9.6337 | 0.1337 | 0.6394 |
| Frozen O/D RAPM | 30 | 10.9640 | 8.9734 | 0.1337 | 0.6078 |
| Frozen draft cold-start prior | 30 | 10.6718 | 8.7037 | 0.1301 | 0.6245 |
| Frozen exposure-gated cold-start prior | 30 | 10.2466 | **8.0307** | 0.1250 | 0.6586 |
| Frozen exposure-gated O/D cold-start prior | 30 | 10.5096 | 8.4409 | 0.1282 | 0.6483 |
| Recursive exposure-gated RAPM | 30 | 10.2778 | 8.4776 | 0.1253 | 0.6829 |
| Student-t recursive RAPM | 30 | 10.6358 | 8.7390 | 0.1297 | 0.6608 |
| Student-t talent-prior RAPM | 30 | **10.1176** | 8.3473 | **0.1234** | **0.6935** |

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
