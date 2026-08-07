---
last_updated: "2026-08-05"
---

# Offense/Defense RAPM

This model separates each player's offensive contribution from their defensive
contribution while retaining the project's ridge-regularized RAPM framework.
It is evaluated as a frozen preseason forecast, not with a target-season
player refit.

## Identification

A net-rating row alone cannot identify separate offensive and defensive player
coefficients: both sides of a player's contribution enter the same lineup
difference. Instead, every lineup stint supplies two offensive-rating rows.

For home offense in stint $s$,

\[
100\frac{P_{H,s}}{N_{H,s}} =
\alpha + \sum_{p\in H_s}o_p - \sum_{p\in A_s}d_p + h + \epsilon_{H,s},
\]

and for away offense,

\[
100\frac{P_{A,s}}{N_{A,s}} =
\alpha + \sum_{p\in A_s}o_p - \sum_{p\in H_s}d_p - h + \epsilon_{A,s}.
\]

$o_p$ is offensive RAPM and $d_p$ is defensive RAPM measured as points
prevented per 100 possessions, so a stronger defender has a larger positive
value. $N_{H,s}$ and $N_{A,s}$ are the separate offensive-possession exposures,
which are also the row weights. $h$ is the home-offense shift; its net home
margin implication is $2h$.

Ridge regularization centers otherwise arbitrary offense/defense level shifts.
The identifiable net contribution for a player is $o_p+d_p$.

## Forward Prior

The first exemplar uses regular seasons only. It fits seasons 1996-97 through
2024-25 in chronological order. Each completed season's offense and defense
coefficients become the ridge center for the following season; unseen players
receive zero on both dimensions. Lambda is selected separately for every
season with the established expanding chronological game folds.

The completed 2024-25 O/D state is then frozen and scored on every 2025-26
regular-season and playoff possession using realized lineup exposure as the
explicit oracle. No 2025-26 outcome is used in fitting a coefficient.

## First Result

| Cohort | One-number lagged RAPM RMSE | Frozen O/D RAPM RMSE | Difference (O/D - one-number) |
| --- | ---: | ---: | ---: |
| Regular-season possessions | 1.199000 | **1.198853** | -0.000147 |
| Regular-season game margin | **14.8894** | 14.8901 | +0.0007 |
| Playoff possessions | **1.192895** | 1.194642 | +0.001747 |
| Playoff game margin | **17.5409** | 17.8037 | +0.2627 |

The regular-season possession improvement is real in this single frozen
holdout but extremely small and does not extend to game-margin or playoff
prediction. The O/D model is therefore not promoted over one-number lagged
RAPM yet. It establishes a useful two-dimensional player representation for
subsequent side-specific priors and nonlinear models.

The immutable run is
`frozen-offense-defense-rapm-2025-26-20260805T050511Z-62b718bd` under
`artifacts/models/frozen_offense_defense_rapm/2025-26/`. It contains the
historical season-by-season coefficients and lambda selections, frozen player
priors, regular/playoff predictions, team evaluations, and file hashes.

## 2025-26 Retrospective Rankings

The following rankings are a separate completed-season descriptive fit. They
start from the frozen 2024-25 O/D state, select lambda with chronological
folds within the 2025-26 regular season, and refit all 1,230 regular-season
games. They must not be read as a preseason forecast or compared directly to
the Frozen Preseason Leaderboard. The selected lambda was 0.03. Each list
requires at least 500 side-specific possessions.

Offense is points added per 100 possessions. Defense is points prevented per
100 possessions, so larger positive values are better on both lists.
Overall RAPM is their sum, which is the identifiable net contribution from the
joint fit. Click any column header in the three tables to sort it.

### Top 25 Overall

| Rank | Player | Team | Net RAPM | Total possessions | Offense RAPM | Defense RAPM |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Nikola Jokić | DEN | 11.89 | 4,786 | 9.73 | 2.16 |
| 2 | Shai Gilgeous-Alexander | OKC | 9.59 | 4,730 | 7.75 | 1.84 |
| 3 | Giannis Antetokounmpo | MIL | 8.76 | 2,129 | 5.25 | 3.51 |
| 4 | Joel Embiid | PHI | 7.77 | 2,479 | 3.74 | 4.03 |
| 5 | Alex Caruso | OKC | 7.72 | 2,125 | 1.14 | 6.58 |
| 6 | Stephen Curry | GSW | 7.50 | 2,822 | 8.43 | -0.93 |
| 7 | Jimmy Butler III | GSW | 7.38 | 2,449 | 5.42 | 1.96 |
| 8 | Victor Wembanyama | SAS | 7.00 | 3,896 | 1.57 | 5.43 |
| 9 | Derrick White | BOS | 6.99 | 5,181 | 2.53 | 4.45 |
| 10 | Rudy Gobert | MIN | 6.75 | 4,951 | -0.59 | 7.33 |
| 11 | Bam Adebayo | MIA | 6.66 | 5,031 | 2.60 | 4.06 |
| 12 | Franz Wagner | ORL | 6.48 | 2,190 | 2.91 | 3.56 |
| 13 | Jrue Holiday | POR | 6.35 | 3,295 | 2.72 | 3.64 |
| 14 | Aaron Gordon | DEN | 6.25 | 2,057 | 3.51 | 2.74 |
| 15 | Jayson Tatum | BOS | 6.23 | 1,046 | 4.86 | 1.37 |
| 16 | Kawhi Leonard | LAC | 6.22 | 4,167 | 4.49 | 1.72 |
| 17 | Jarrett Allen | CLE | 5.88 | 3,191 | 3.91 | 1.97 |
| 18 | Devin Booker | PHX | 5.73 | 4,442 | 5.76 | -0.04 |
| 19 | Pascal Siakam | IND | 5.64 | 4,322 | 3.44 | 2.19 |
| 20 | Paul George | PHI | 5.51 | 2,328 | 1.91 | 3.60 |
| 21 | Donovan Mitchell | CLE | 5.49 | 4,925 | 5.10 | 0.39 |
| 22 | Cade Cunningham | DET | 5.26 | 4,490 | 3.71 | 1.56 |
| 23 | Luka Dončić | LAL | 5.16 | 4,759 | 5.30 | -0.14 |
| 24 | Lauri Markkanen | UTA | 5.02 | 3,080 | 3.21 | 1.81 |
| 25 | Herbert Jones | NOP | 5.01 | 3,289 | 1.79 | 3.23 |

### Top 25 Offense

| Rank | Player | Team | Offense RAPM | Offensive possessions | Net RAPM |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Nikola Jokić | DEN | 9.73 | 4,795 | 11.89 |
| 2 | Stephen Curry | GSW | 8.43 | 2,823 | 7.50 |
| 3 | Shai Gilgeous-Alexander | OKC | 7.75 | 4,735 | 9.59 |
| 4 | James Harden | LAC | 5.95 | 4,944 | 4.02 |
| 5 | Trae Young | ATL | 5.94 | 853 | 1.90 |
| 6 | Devin Booker | PHX | 5.76 | 4,448 | 5.73 |
| 7 | LaMelo Ball | CHA | 5.62 | 4,120 | 4.00 |
| 8 | Jimmy Butler III | GSW | 5.42 | 2,448 | 7.38 |
| 9 | Luka Dončić | LAL | 5.30 | 4,758 | 5.16 |
| 10 | Giannis Antetokounmpo | MIL | 5.25 | 2,128 | 8.76 |
| 11 | Donovan Mitchell | CLE | 5.10 | 4,927 | 5.49 |
| 12 | Karl-Anthony Towns | NYK | 4.95 | 4,722 | 4.80 |
| 13 | Jayson Tatum | BOS | 4.86 | 1,048 | 6.23 |
| 14 | Jalen Brunson | NYK | 4.85 | 5,245 | 3.07 |
| 15 | Kawhi Leonard | LAC | 4.49 | 4,169 | 6.22 |
| 16 | Michael Porter Jr. | BKN | 4.11 | 3,402 | 2.99 |
| 17 | Kevin Durant | HOU | 3.97 | 5,739 | 4.81 |
| 18 | De'Aaron Fox | SAS | 3.92 | 4,707 | 2.90 |
| 19 | Jarrett Allen | CLE | 3.91 | 3,190 | 5.88 |
| 20 | Luke Kennard | ATL | 3.90 | 3,472 | 2.26 |
| 21 | Joel Embiid | PHI | 3.74 | 2,484 | 7.77 |
| 22 | Cade Cunningham | DET | 3.71 | 4,490 | 5.26 |
| 23 | CJ McCollum | ATL | 3.70 | 4,776 | 3.87 |
| 24 | Julius Randle | MIN | 3.69 | 5,476 | 3.07 |
| 25 | Jamal Murray | DEN | 3.60 | 5,453 | 1.52 |

### Top 25 Defense

| Rank | Player | Team | Defense RAPM | Defensive possessions | Net RAPM |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Rudy Gobert | MIN | 7.33 | 4,961 | 6.75 |
| 2 | Alex Caruso | OKC | 6.58 | 2,124 | 7.72 |
| 3 | Victor Wembanyama | SAS | 5.43 | 3,892 | 7.00 |
| 4 | Wendell Carter Jr. | ORL | 4.49 | 4,770 | 4.51 |
| 5 | Derrick White | BOS | 4.45 | 5,178 | 6.99 |
| 6 | Draymond Green | GSW | 4.24 | 3,874 | 2.16 |
| 7 | Jaren Jackson Jr. | MEM | 4.17 | 3,066 | 4.31 |
| 8 | Bam Adebayo | MIA | 4.06 | 5,027 | 6.66 |
| 9 | Joel Embiid | PHI | 4.03 | 2,474 | 7.77 |
| 10 | Clint Capela | HOU | 3.96 | 1,849 | 3.80 |
| 11 | Jrue Holiday | POR | 3.64 | 3,292 | 6.35 |
| 12 | Matisse Thybulle | POR | 3.63 | 1,008 | 2.62 |
| 13 | Paul George | PHI | 3.60 | 2,325 | 5.51 |
| 14 | Franz Wagner | ORL | 3.56 | 2,191 | 6.48 |
| 15 | Giannis Antetokounmpo | MIL | 3.51 | 2,130 | 8.76 |
| 16 | Jusuf Nurkić | UTA | 3.48 | 2,337 | 2.32 |
| 17 | OG Anunoby | NYK | 3.43 | 4,537 | 4.55 |
| 18 | Herbert Jones | NOP | 3.23 | 3,293 | 5.01 |
| 19 | Chet Holmgren | OKC | 3.15 | 4,127 | 4.94 |
| 20 | Anthony Davis | DAL | 3.09 | 1,295 | 4.27 |
| 21 | Brook Lopez | LAC | 2.99 | 3,278 | 2.74 |
| 22 | Evan Mobley | CLE | 2.97 | 4,319 | 4.37 |
| 23 | Aaron Gordon | DEN | 2.74 | 2,056 | 6.25 |
| 24 | Dillon Brooks | PHX | 2.68 | 3,506 | 4.28 |
| 25 | Jalen Suggs | ORL | 2.67 | 3,373 | 2.10 |

The immutable ranking run is
`all-season-offense-defense-rapm-2025-26-20260805T125051Z-7f5ef527` under
`artifacts/models/offense_defense_rapm/2025-26/`. It includes both published
tables, all coefficients and exposures, the 2025-26 lambda-selection evidence,
and hashes of the exact 2024-25 frozen source state.

## Verification

`tests/test_offense_defense_rapm.py` verifies that the two-row sparse design
places the offense lineup in offensive columns and the opponent lineup in
defensive columns with the required sign. It also verifies that frozen scoring
uses the two coefficient types separately and that published side rankings use
the corresponding offensive or defensive possession exposure threshold.
