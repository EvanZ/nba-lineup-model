<p class="project-kicker">Forward priors / regularized adjusted plus-minus</p>

# Prior-Centered RAPM

<p class="project-lead">
Prior-centered RAPM keeps the canonical one-number lineup design but shrinks
each player toward a forward-looking estimate instead of toward zero. The
first exemplar uses only prior-season RAPM; age adjustment is deliberately
deferred to a later ablation.
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
2019-20 is fitted as ordinary zero-centered Ridge RAPM; its player estimates
become the 2020-21 priors, then the procedure repeats forward through 2025-26.
The target season's games, box-score outcomes, and RAPM estimates are not
inputs to its prior.

Players absent from the frozen prior table are explicit cold starts and receive
a zero prior. The output records `prior_available` so
missing prior coverage cannot be mistaken for an observed zero estimate.

This ablation intentionally excludes age, experience, box-score, and draft
information. Once its value relative to zero-centered RAPM is understood, the
aging adjustment becomes a separate, directly comparable extension.

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

The historical pass/warning panel from 2019-20 through 2024-25 is used only
to construct frozen lagged-RAPM priors. It does not change the Leaderboard
evaluation split: the final model trains on the same first 1,044 2025-26
regular-season games used by the other exemplars and predicts the same final
186 regular-season holdout games. The prior is frozen before every target
season fold; lambda is selected only by chronological validation within those
1,044 training games.

The Leaderboard row will use the same possession-level and eligible-game-margin
metrics as canonical Ridge RAPM, on exactly the same holdout game IDs. The
implementation is intentionally present before that promotion so the model's
assumptions are reviewable without retrofitting them to an outcome.

## 2025-26 Ranking

This table is the regular-only forward-prior fit on the first 1,044 2025-26
regular-season games. `RAPM = Prior + Adjustment`, where the prior is the
frozen 2024-25 lagged-RAPM estimate. These values use a prior-centered scale
and are not directly interchangeable with zero-centered one-season RAPM.

| Rank | Player | Team | RAPM | Prior | Adjustment | Possessions |
| ---: | --- | :---: | ---: | ---: | ---: | ---: |
| 1 | Nikola Jokić | DEN | 11.16 | 9.09 | 2.08 | 4,786 |
| 2 | Shai Gilgeous-Alexander | OKC | 10.13 | 8.59 | 1.54 | 4,730 |
| 3 | Derrick White | BOS | 9.26 | 5.66 | 3.60 | 5,180 |
| 4 | Victor Wembanyama | SAS | 8.78 | 2.35 | 6.43 | 3,896 |
| 5 | Giannis Antetokounmpo | MIL | 8.25 | 7.57 | 0.68 | 2,129 |
| 6 | Alex Caruso | OKC | 8.19 | 6.51 | 1.68 | 2,125 |
| 7 | Kawhi Leonard | LAC | 7.99 | 5.97 | 2.01 | 4,167 |
| 8 | Bam Adebayo | MIA | 7.62 | 4.43 | 3.19 | 5,031 |
| 9 | Jimmy Butler III | GSW | 7.38 | 5.70 | 1.67 | 2,449 |
| 10 | Joel Embiid | PHI | 6.96 | 7.15 | -0.18 | 2,479 |
| 11 | Donovan Mitchell | CLE | 6.86 | 4.92 | 1.94 | 4,925 |
| 12 | Aaron Gordon | DEN | 6.49 | 4.70 | 1.79 | 2,057 |
| 13 | Devin Booker | PHX | 6.37 | 4.88 | 1.49 | 4,442 |
| 14 | Cade Cunningham | DET | 6.32 | 2.99 | 3.32 | 4,490 |
| 15 | Lauri Markkanen | UTA | 6.21 | 3.96 | 2.25 | 3,080 |
| 16 | Jayson Tatum | BOS | 6.15 | 5.69 | 0.47 | 1,046 |
| 17 | Marcus Smart | LAL | 5.80 | 1.98 | 3.82 | 3,607 |
| 18 | Kevin Durant | HOU | 5.77 | 5.04 | 0.72 | 5,732 |
| 19 | Paul George | PHI | 5.55 | 6.32 | -0.76 | 2,328 |
| 20 | Michael Porter Jr. | BKN | 5.53 | 3.96 | 1.57 | 3,396 |
| 21 | Luka Dončić | LAL | 5.52 | 4.61 | 0.92 | 4,759 |
| 22 | Chet Holmgren | OKC | 5.42 | 2.55 | 2.88 | 4,129 |
| 23 | De'Anthony Melton | GSW | 5.32 | 2.01 | 3.31 | 2,311 |
| 24 | Stephen Curry | GSW | 5.29 | 6.34 | -1.05 | 2,822 |
| 25 | Jrue Holiday | POR | 5.29 | 4.57 | 0.71 | 3,295 |

## Largest 2025-26 Adjustments

The lists below rank players by the fitted movement from the frozen prior, with
a 500-possession floor. They are not literal measures of improvement or
regression: an adjustment can reflect real change, different health or role,
lineup context, or estimation noise.

### Largest Positive Adjustments

| Player | Team | RAPM | Prior | Adjustment | Possessions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Victor Wembanyama | SAS | 8.78 | 2.35 | 6.43 | 3,896 |
| Devin Vassell | SAS | 2.81 | -2.69 | 5.50 | 4,258 |
| Moussa Diabaté | CHA | 5.23 | 0.76 | 4.47 | 3,790 |
| Julian Champagnie | SAS | 4.40 | 0.30 | 4.10 | 4,745 |
| Marcus Smart | LAL | 5.80 | 1.98 | 3.82 | 3,607 |
| Collin Gillespie | PHX | 3.58 | -0.21 | 3.78 | 4,627 |
| Josh Green | CHA | 3.15 | -0.58 | 3.73 | 1,804 |
| LaMelo Ball | CHA | 4.90 | 1.19 | 3.71 | 4,101 |
| Hugo González | BOS | 3.69 | 0.00 | 3.69 | 2,133 |
| Derrick White | BOS | 9.26 | 5.66 | 3.60 | 5,180 |
| Jalen Smith | CHI | 3.07 | -0.31 | 3.38 | 2,316 |
| Davion Mitchell | MIA | 3.98 | 0.64 | 3.34 | 4,265 |
| Cade Cunningham | DET | 6.32 | 2.99 | 3.32 | 4,490 |
| De'Anthony Melton | GSW | 5.32 | 2.01 | 3.31 | 2,311 |
| Dyson Daniels | ATL | 3.88 | 0.58 | 3.29 | 5,317 |
| Oso Ighodaro | PHX | 3.27 | 0.02 | 3.25 | 3,634 |
| Bam Adebayo | MIA | 7.62 | 4.43 | 3.19 | 5,031 |
| Donte DiVincenzo | MIN | 3.39 | 0.27 | 3.12 | 5,223 |
| Kon Knueppel | CHA | 3.09 | 0.00 | 3.09 | 5,207 |
| Brandon Miller | CHA | 2.27 | -0.81 | 3.08 | 3,968 |
| CJ McCollum | ATL | 4.22 | 1.32 | 2.90 | 4,762 |
| Neemias Queta | BOS | 2.35 | -0.53 | 2.89 | 3,752 |
| Chet Holmgren | OKC | 5.42 | 2.55 | 2.88 | 4,129 |
| Amen Thompson | HOU | 4.04 | 1.29 | 2.75 | 5,936 |
| Grant Williams | CHA | 3.03 | 0.29 | 2.75 | 1,438 |

### Largest Negative Adjustments

| Player | Team | RAPM | Prior | Adjustment | Possessions |
| --- | :---: | ---: | ---: | ---: | ---: |
| Drake Powell | BKN | -4.13 | 0.00 | -4.13 | 2,655 |
| Isaiah Collier | UTA | -4.98 | -0.87 | -4.11 | 3,237 |
| Draymond Green | GSW | 0.91 | 5.01 | -4.10 | 3,874 |
| Kobe Brown | IND | -4.15 | -0.38 | -3.77 | 1,960 |
| Nic Claxton | BKN | -4.41 | -0.81 | -3.60 | 3,868 |
| Tre Mann | CHA | -3.70 | -0.14 | -3.56 | 1,319 |
| Gary Trent Jr. | MIL | -6.20 | -2.75 | -3.46 | 2,800 |
| Royce O'Neale | PHX | -2.01 | 1.15 | -3.17 | 4,548 |
| Luguentz Dort | OKC | 1.38 | 4.51 | -3.14 | 3,809 |
| Darius Garland | CLE | 1.84 | 4.89 | -3.05 | 2,807 |
| Bub Carrington | WAS | -4.68 | -1.70 | -2.98 | 4,827 |
| Myles Turner | MIL | 1.80 | 4.78 | -2.98 | 3,938 |
| Jarace Walker | IND | -3.48 | -0.59 | -2.88 | 4,157 |
| De'Andre Hunter | CLE | -1.42 | 1.42 | -2.83 | 2,489 |
| Patrick Williams | CHI | -5.02 | -2.19 | -2.83 | 3,168 |
| Andrew Nembhard | IND | -2.79 | 0.02 | -2.80 | 3,732 |
| Mike Conley | MIN | 1.04 | 3.84 | -2.80 | 2,071 |
| LeBron James | LAL | 2.42 | 5.13 | -2.71 | 4,077 |
| Brooks Barnhizer | OKC | -2.69 | 0.00 | -2.69 | 703 |
| Rayan Rupert | POR | -3.43 | -0.74 | -2.68 | 2,244 |
| Caris LeVert | DET | -1.06 | 1.60 | -2.66 | 2,370 |
| Will Riley | WAS | -2.65 | 0.00 | -2.65 | 3,434 |
| Jeremy Sochan | SAS | -2.10 | 0.54 | -2.64 | 958 |
| DeMar DeRozan | SAC | -0.38 | 2.22 | -2.60 | 4,983 |
| Bruce Brown | DEN | -4.68 | -2.08 | -2.60 | 4,097 |

## Historical Coverage Boundary

Historical NBA Stats V3 processing retains named failed games rather than
silently repairing or dropping them. The approved first-exemplar policy uses
the audited pass/warning subset, records excluded IDs and their source
coverage in the model manifest, and evaluates the final 2025-26 model only on
the shared Leaderboard holdout.

## Historical Playoff Ablation

The initial comparison also fits a second prior chain that appends each
completed historical season's available playoff stints to its regular-season
stints before forming the next season's prior. Both variants use the identical
2025-26 first-1,044-game fit and are evaluated with that frozen state.

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

- add the forward age adjustment to the lagged RAPM prior;
- use returning and cold-start error scales as prior precision weights;
- add box-score plus-minus and draft-position inputs to cold-start priors;
- replace the two-stage pipeline with the joint dynamic RAPM design in the
  [Modeling Roadmap](roadmap.md).
