<p class="project-kicker">Forward priors / regularized adjusted plus-minus</p>

# Prior-Centered RAPM

<p class="project-lead">
Prior-centered RAPM keeps the canonical one-number lineup design but shrinks
each player toward a forward-looking estimate instead of toward zero. The
first exemplar uses only prior-season RAPM; a second, directly comparable
ablation uses the frozen output of the RAPM aging model.
</p>

## Estimand

For target-season lineup stint (j), let (X_j) contain `+1` for home players
and `-1` for away players, (y_j) be home net rating, and (mu_i) be player
(i)'s prior-season RAPM. The estimator is

\[
\underset{\beta,b}{\operatorname{argmin}}
\quad
\sum_j w_j\left(y_j-b-X_j\beta\right)^2
+
N\lambda\sum_i\left(\beta_i-\mu_i\right)^2.
\]

The intercept (b) remains an unpenalized home-court term. Stint possessions
are the weights (w_j), normalized to mean one before fitting, consistent with
canonical Ridge RAPM.

## First Exemplar

The prior for season (t) is the completed RAPM estimate from season (t-1).
1996-97 is fitted as ordinary zero-centered Ridge RAPM; its player estimates
become the 1997-98 priors, then the procedure repeats forward through 2025-26.
The target season's games, box-score outcomes, and RAPM estimates are not
inputs to its prior. Each historical season uses only its audited pass/warning
curated-game subset; unresolved legacy identity placeholders are excluded from
the separate player-history panel, not silently mapped to real players.

Players absent from the frozen prior table are explicit cold starts and receive
a zero prior. The output records `prior_available` so
missing prior coverage cannot be mistaken for an observed zero estimate.

This first ablation intentionally excludes age, experience, box-score, and
draft information. The age-informed extension below evaluates whether a
pre-season aging forecast improves this simple lagged prior.

## Implementation

The sparse solver fits the equivalent residualized problem:

\[
y'_j = y_j - X_j\mu,
\qquad
\widehat{\delta}
=
\underset{\delta}{\operatorname{argmin}}
\sum_j w_j\left(y'_j-b-X_j\delta\right)^2
+N\lambda\lVert\delta\rVert_2^2,
\qquad
\widehat{\beta}=\mu+\widehat{\delta}.
\]

This is exactly the original objective, not an approximation. It retains the
project's SciPy CSR representation and scikit-learn `lsqr` Ridge solver. The
published player table will include the final RAPM estimate, prior mean, and
the fitted adjustment from the prior.

## Selection And Evaluation

The historical pass/warning panel from 1996-97 through 2024-25 is used only
to construct frozen lagged-RAPM priors. It does not change the Leaderboard
evaluation split: the final model trains on the same first 1,044 2025-26
regular-season games used by the other exemplars and predicts the same final
186 regular-season holdout games. The prior is frozen before every target
season fold; lambda is selected only by chronological validation within those
1,044 training games.

The Leaderboard row uses the same possession-level and eligible-game-margin
metrics as canonical Ridge RAPM, on exactly the same holdout game IDs.

## Age-Informed Prior Exemplar

The second exemplar replaces the completed prior-season coefficient with the
frozen player-specific forecast from the [RAPM Aging Model](aging-model.md):

\[
\mu_{i,2025\text{-}26}
=
f\left(
  \widehat{\beta}^{0}_{i,2024\text{-}25},
  \operatorname{age}_{i,2025\text{-}26},
  \operatorname{experience}_{i,2025\text{-}26},
  \operatorname{exposure}_{i,2024\text{-}25},
  \operatorname{returning}_i,
  \operatorname{rookie}_i
\right).
\]

The aging run is trained only through 2024-25 and publishes its 2025-26
`player_priors.parquet` before any 2025-26 RAPM fitting. The current run uses
the full 1996-97 through 2024-25 history, the same chronological lambda
selection, and the same 1,044-game fit as the lagged-prior model.

| Prior definition | Holdout stint RMSE | Holdout game-margin RMSE | Frozen playoff possession RMSE | Frozen playoff game-margin RMSE |
| --- | ---: | ---: | ---: | ---: |
| Completed 2024-25 RAPM | **103.7747** | **15.2350** | 1.191755 | 15.3103 |
| Full-history aging forecast | 103.8514 | 15.3933 | **1.191519** | **15.2805** |

The age-informed prior is weaker on the locked regular-season holdout, while
the frozen playoff result is better on this 85-game cohort. This remains an
informative negative regular-season ablation, not sufficient evidence to
override the prespecified regular-season selection target. The immutable RAPM
artifact is `aging-prior-rapm-2025-26-20260803T214725Z-9df2aa04`; it pins aging
run `aging-2025-26-20260803T214653Z-94ce6277` and all game assignments.

## Blended Aging And Lagged Prior

The next ablation gives each frozen prior an explicit nonnegative share:

\[
\mu_i = w\mu_i^{\text{lagged}} + (1-w)\mu_i^{\text{aging}},
\qquad 0 \leq w \leq 1.
\]

The candidate grid is \(w \in \{0,0.25,0.5,0.75,1\}\), crossed with the
standard RAPM lambda grid and selected by pooled chronological validation MSE
within the first 1,044 2025-26 regular-season games. The endpoints reproduce
the two earlier prior definitions exactly.

The selected weight is \(w=1\) on lagged RAPM and \(w=0\) on the aging
forecast, with \(\lambda=0.03\). In the full-history selection surface, the
best interior candidate, \(w=0.75\), has validation weighted MSE 10,838.03
versus 10,836.97 for lagged-only. Consequently, its regular-holdout and
frozen-playoff metrics exactly match the lagged-prior model. There is no
evidence here that a linear blend adds information beyond the recursive
lagged-RAPM prior.

The selection surface and frozen outputs are retained in
`blended-prior-rapm-2025-26-20260803T214825Z-2dbb1766`.

## 2025-26 Ranking

This table is the regular-only forward-prior fit on the first 1,044 2025-26
regular-season games. `RAPM = Prior + Adjustment`, where the prior is the
frozen 2024-25 lagged-RAPM estimate. These values use a prior-centered scale
and are not directly interchangeable with zero-centered one-season RAPM.

| Rank | Player | Team | RAPM | Prior | Adjustment | Possessions |
| ---: | --- | :---: | ---: | ---: | ---: | ---: |
| 1 | Nikola Jokić | DEN | 13.07 | 11.47 | 1.61 | 4,786 |
| 2 | Shai Gilgeous-Alexander | OKC | 9.76 | 7.85 | 1.91 | 4,730 |
| 3 | Victor Wembanyama | SAS | 8.62 | 1.91 | 6.71 | 3,896 |
| 4 | Giannis Antetokounmpo | MIL | 8.56 | 7.76 | 0.80 | 2,129 |
| 5 | Jimmy Butler III | GSW | 8.37 | 7.41 | 0.95 | 2,449 |
| 6 | Derrick White | BOS | 8.25 | 3.91 | 4.34 | 5,180 |
| 7 | Alex Caruso | OKC | 7.75 | 5.71 | 2.04 | 2,125 |
| 8 | Donovan Mitchell | CLE | 7.64 | 5.96 | 1.69 | 4,925 |
| 9 | Joel Embiid | PHI | 7.58 | 7.73 | -0.14 | 2,479 |
| 10 | Stephen Curry | GSW | 7.40 | 9.47 | -2.06 | 2,822 |
| 11 | Bam Adebayo | MIA | 7.12 | 3.40 | 3.71 | 5,031 |
| 12 | Kawhi Leonard | LAC | 7.00 | 4.02 | 2.98 | 4,167 |
| 13 | Marcus Smart | LAL | 6.67 | 3.77 | 2.90 | 3,607 |
| 14 | Aaron Gordon | DEN | 6.65 | 4.73 | 1.92 | 2,057 |
| 15 | Karl-Anthony Towns | NYK | 6.43 | 5.53 | 0.90 | 4,716 |
| 16 | Devin Booker | PHX | 6.39 | 4.79 | 1.60 | 4,442 |
| 17 | Cade Cunningham | DET | 6.27 | 2.66 | 3.61 | 4,490 |
| 18 | Jayson Tatum | BOS | 6.25 | 5.76 | 0.50 | 1,046 |
| 19 | Rudy Gobert | MIN | 6.15 | 5.92 | 0.23 | 4,951 |
| 20 | Jrue Holiday | POR | 6.12 | 5.71 | 0.41 | 3,295 |
| 21 | Chris Paul | LAC | 5.82 | 8.57 | -2.75 | 457 |
| 22 | Lauri Markkanen | UTA | 5.77 | 3.40 | 2.37 | 3,080 |
| 23 | Chet Holmgren | OKC | 5.73 | 2.86 | 2.87 | 4,129 |
| 24 | Paul George | PHI | 5.72 | 6.52 | -0.80 | 2,328 |
| 25 | De'Anthony Melton | GSW | 5.67 | 2.56 | 3.12 | 2,311 |

## Largest 2025-26 Adjustments

The lists below rank players by the fitted movement from the frozen prior, with
a 500-possession floor. They are not literal measures of improvement or
regression: an adjustment can reflect real change, different health or role,
lineup context, or estimation noise.

### Largest Positive Adjustments

| Player | Team | RAPM | Prior | Adjustment | Possessions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Victor Wembanyama | SAS | 8.62 | 1.91 | 6.71 | 3,896 |
| Devin Vassell | SAS | 3.79 | -1.35 | 5.14 | 4,258 |
| Moussa Diabaté | CHA | 5.14 | 0.43 | 4.71 | 3,790 |
| Derrick White | BOS | 8.25 | 3.91 | 4.34 | 5,180 |
| Julian Champagnie | SAS | 4.35 | 0.09 | 4.26 | 4,745 |
| Collin Gillespie | PHX | 3.80 | -0.12 | 3.93 | 4,626 |
| Josh Green | CHA | 3.41 | -0.43 | 3.84 | 1,804 |
| Hugo González | BOS | 3.80 | 0.00 | 3.80 | 2,133 |
| LaMelo Ball | CHA | 5.43 | 1.64 | 3.79 | 4,101 |
| Bam Adebayo | MIA | 7.12 | 3.40 | 3.71 | 5,031 |
| Cade Cunningham | DET | 6.27 | 2.66 | 3.61 | 4,490 |
| Davion Mitchell | MIA | 4.82 | 1.27 | 3.55 | 4,265 |
| Oso Ighodaro | PHX | 3.35 | 0.04 | 3.31 | 3,634 |
| Kon Knueppel | CHA | 3.31 | 0.00 | 3.31 | 5,207 |
| Donte DiVincenzo | MIN | 3.22 | -0.02 | 3.24 | 5,223 |
| Brandon Miller | CHA | 2.58 | -0.65 | 3.23 | 3,968 |
| Jalen Smith | CHI | 3.97 | 0.76 | 3.21 | 2,316 |
| Dyson Daniels | ATL | 4.40 | 1.28 | 3.12 | 5,317 |
| De'Anthony Melton | GSW | 5.67 | 2.56 | 3.12 | 2,311 |
| Scottie Barnes | TOR | 3.39 | 0.31 | 3.07 | 5,512 |
| Neemias Queta | BOS | 3.13 | 0.07 | 3.06 | 3,752 |
| Kawhi Leonard | LAC | 7.00 | 4.02 | 2.98 | 4,167 |
| Marcus Smart | LAL | 6.67 | 3.77 | 2.90 | 3,607 |
| Amen Thompson | HOU | 4.43 | 1.56 | 2.88 | 5,936 |
| Chet Holmgren | OKC | 5.73 | 2.86 | 2.87 | 4,129 |

### Largest Negative Adjustments

| Player | Team | RAPM | Prior | Adjustment | Possessions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Draymond Green | GSW | 0.14 | 4.23 | -4.10 | 3,874 |
| Isaiah Collier | UTA | -4.84 | -0.78 | -4.06 | 3,237 |
| Drake Powell | BKN | -4.02 | 0.00 | -4.02 | 2,655 |
| Gary Trent Jr. | MIL | -5.41 | -1.51 | -3.90 | 2,800 |
| Kobe Brown | IND | -4.19 | -0.43 | -3.76 | 1,960 |
| Tre Mann | CHA | -3.07 | 0.50 | -3.58 | 1,319 |
| LeBron James | LAL | 3.23 | 6.78 | -3.56 | 4,077 |
| Royce O'Neale | PHX | -1.35 | 1.98 | -3.33 | 4,548 |
| Andrew Nembhard | IND | -2.19 | 1.13 | -3.31 | 3,732 |
| Patrick Williams | CHI | -4.25 | -0.95 | -3.30 | 3,168 |
| Mike Conley | MIN | 1.91 | 5.13 | -3.22 | 2,071 |
| Luguentz Dort | OKC | 1.36 | 4.36 | -2.99 | 3,809 |
| Nic Claxton | BKN | -3.97 | -1.13 | -2.84 | 3,868 |
| Bub Carrington | WAS | -4.51 | -1.68 | -2.83 | 4,827 |
| Jarace Walker | IND | -3.33 | -0.50 | -2.82 | 4,157 |
| De'Andre Hunter | CLE | -0.94 | 1.87 | -2.81 | 2,489 |
| Bruce Brown | DEN | -4.58 | -1.79 | -2.79 | 4,097 |
| Buddy Hield | GSW | 0.09 | 2.83 | -2.74 | 1,718 |
| DeMar DeRozan | SAC | -0.25 | 2.45 | -2.70 | 4,983 |
| Tyus Jones | ORL | -2.74 | -0.04 | -2.69 | 2,021 |
| Darius Garland | CLE | 1.73 | 4.39 | -2.66 | 2,806 |
| Brooks Barnhizer | OKC | -2.64 | 0.00 | -2.64 | 703 |
| Myles Turner | MIL | 1.50 | 4.11 | -2.61 | 3,938 |
| Rayan Rupert | POR | -3.24 | -0.66 | -2.58 | 2,244 |
| Caris LeVert | DET | -0.75 | 1.82 | -2.57 | 2,370 |

## Historical Coverage Boundary

Historical NBA Stats V3 processing retains named failed games rather than
silently repairing or dropping them. The approved first-exemplar policy uses
the audited pass/warning subset, records excluded IDs and their source
coverage in the model manifest, and evaluates the final 2025-26 model only on
the shared Leaderboard holdout.

## Historical Playoff Ablation

The initial six-season comparison also fits a second prior chain that appends
each completed historical season's available playoff stints to its
regular-season stints before forming the next season's prior. Both variants use
the identical 2025-26 first-1,044-game fit and are evaluated with that frozen
state. It has not yet been regenerated over the full 1996-97 history because
historical playoff coverage is a separate acquisition/quality boundary.

| Historical prior source | 2025-26 holdout stint RMSE | Holdout game-margin RMSE | Frozen-state playoff possession RMSE | Frozen-state playoff game-margin RMSE |
| --- | ---: | ---: | ---: | ---: |
| Regular season only | **103.8090** | 15.4400 | **1.191612** | **15.4165** |
| Regular season plus playoffs | 103.8118 | **15.4367** | 1.191635 | 15.4357 |

The playoff-inclusive chain used 372 successfully processed historical playoff
games: 74, 65, 76, 65, 56, and 36 from 2019-20 through 2024-25 respectively.
At this coverage level, it does not improve the frozen 2025-26 prediction
task. The result is an ablation rather than a reason to discard historical
playoffs permanently; coverage should be completed before drawing a stronger
conclusion.

## Correctness Checks

`tests/test_prior_rapm.py` verifies that residualization restores the
prior-centered coefficient exactly, that chronological lambda selection works,
and that an unseen player receives the explicit zero cold-start prior.

## Next Extensions

- use returning and cold-start error scales as prior precision weights;
- add box-score plus-minus and draft-position inputs to cold-start priors;
- replace the two-stage pipeline with the joint dynamic RAPM design in the
  [Modeling Roadmap](roadmap.md).
