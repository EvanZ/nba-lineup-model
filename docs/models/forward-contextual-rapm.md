---
last_updated: "2026-08-07"
---

# Forward Contextual RAPM

Forward Contextual RAPM is the current gold-standard preseason model in this
project. It retains an interpretable, one-number additive value for every
player while carrying a separate nonlinear estimate of lineup composition from
one completed season into the next. On the frozen 2025-26 regular-season
holdout, it has the best game-margin RMSE, team net-rating error, and
Pythagorean win-total error in the [Frozen Preseason
Leaderboard](preseason-leaderboard.md).

It is intentionally **not** one 30-season pooled nonlinear model. The state is
updated one completed season at a time. That keeps both the player prior and
the lineup-composition function on the same forward-looking information
boundary.

## Model State

For each season \(t\), the model maintains three pieces of state:

- \(\mu_{i,t}\): player \(i\)'s preseason RAPM prior.
- \(\beta_{i,t}\): player \(i\)'s completed-season additive coefficient.
- \(g_t(z)\): a nonlinear home-minus-away lineup-composition correction, where
  \(z\) is built from the two five-player lineups' trait profiles.

The published player ranking is \(\beta_{i,t}\), carried forward as the
additive part of \(\mu_{i,t+1}\). The model does not try to allocate \(g_t\) to
individual players because its value depends on the five-player combination.
For example, the same high-usage guard can receive a different contextual
correction next to four shooters than next to another ball-dominant creator.

## Training Data And Player Profiles

The additive fit uses regular-season reconstructed stints. A row is a fixed
home/away lineup interval with target \(y_{s,t}\), the home team's net rating
over the stint, and weight \(w_{s,t}\), the number of possessions in that
stint. The signed RAPM design vector \(x_{s,t}\) has \(+1\) for each home player
and \(-1\) for each away player.

Every player in a season also receives a leakage-safe trait profile. Returning
players use their immediately preceding season's possession-native box-score
rates. Each rate is stabilized toward the historical league rate with 300
pseudo-possessions:

\[
r_{i,t} = 100\,
\frac{c_{i,t-1} + 300\,r_{\mathrm{league},t-1}/100}
{P_{i,t-1} + 300}.
\]

The raw profile contains three-point attempts and makes, assists, turnovers,
usage events, offensive and defensive rebounds, steals, and blocks per 100
possessions. A first-season player has no previous NBA rates, so the profile is
a draft-cohort profile blended toward the historical low-exposure replacement
profile. The blend weight is the same forward exposure-gate probability used
by the cold-start RAPM prior. Thus cold-start profiles are explicit estimates,
not silent zero fills.

## How This Differs From A Typical Box-Score Prior

Most box-score RAPM priors predict a single prior value for each player:

\[
\mu_{i,t} = f(\text{box-score profile of player }i).
\]

That scalar becomes the prior mean for player \(i\)'s RAPM coefficient. The
box score is therefore answering a player-level question: how valuable should
this player be expected to be in an average lineup context?

Forward Contextual RAPM does not turn these rates into another permanent player
rating. It preserves the forward RAPM and cold-start state as the additive
player prior, then uses the rate profiles only to construct a lineup-level
term:

\[
g_t\bigl(\phi(H),\phi(A)\bigr).
\]

The correction belongs to the combination of five players, not to a single
player. It can therefore represent diminishing returns and complements. A
second high-usage player may have a different correction beside four shooters
than beside another ball-dominant creator; an extra shooter may matter more in
a lineup whose bottom two shooters are weak; and rebounding can saturate rather
than rising linearly with each player's individual rebound rate.

This makes the box-score layer a structured lineup-synergy model rather than a
conventional box-score prior. The distinction also preserves interpretability:
the published player ranking remains the additive RAPM state, while \(g_t\) is
reported and applied only when evaluating a concrete lineup.

## Lineup Context Function

For a home lineup \(H\) and away lineup \(A\), \(z=\phi(H,A)\) contains
home-minus-away differences in the nine player-rate totals above plus eleven
composition summaries:

- Bottom-two three-point makes and the number of credible shooters.
- Top-two assists and concentration of usage in the top two players.
- Square-root transformed offensive and defensive rebounding totals.
- Counts and weights of imputed/replacement profiles.
- Shooting-by-usage, shooter-by-passing, and rebounding-by-usage interactions.

This construction deliberately captures diminishing and complementary lineup
properties without pretending that a player has a fixed interaction value.
For example, bottom-two shooting can distinguish a five-player unit with no
credible floor spacing from one whose total shooting is carried by only one
elite shooter.

The current contextual function is a possession-weighted spline Ridge model:

\[
g_t(z) = h_{\alpha}(z),
\qquad \alpha = 10{,}000.
\]

`h` applies a quadratic spline basis with four knots to each feature, then
standardizes the expanded columns and fits Ridge regression. The large fixed
alpha is a conservative first-pass regularizer. It shrinks nonlinear effects
heavily while still allowing smooth saturation and interaction patterns. Alpha
tuning is deliberately deferred until this state-transition design has been
validated on additional frozen seasons.

## Seasonal Transition

For season \(t\), the completed prior-season function \(g_{t-1}\) is scored on
season-\(t\) preseason profiles and subtracted from each raw stint target:

\[
y^{\mathrm{adj}}_{s,t}=y_{s,t}-g_{t-1}(z_{s,t}).
\]

The player coefficients are then fit with prior-centered, possession-weighted
Ridge:

\[
(\hat a_t,\hat\beta_t) = \arg\min_{a,\beta}
\left[
\sum_s w_{s,t}\bigl(y_{s,t}-g_{t-1}(z_{s,t})-a-x_{s,t}^{\mathsf T}\beta\bigr)^2
+ \lambda_t\sum_i(\beta_i-\mu_{i,t})^2
\right].
\]

\(\lambda_t\) is held to the already-published season-specific value selected
by the forward exposure-gated RAPM run. Holding that schedule fixed isolates
the impact of the new contextual state transition instead of retuning two
sources of regularization simultaneously.

After season \(t\) completes, the model computes raw residuals
\(y_{s,t}-\hat a_t-x_{s,t}^{\mathsf T}\hat\beta_t\) and fits \(g_t\) to those
residuals. The resulting \(\hat\beta_t\) and \(g_t\) become the state used for
season \(t+1\). Returning players receive their completed \(\hat\beta_t\) as the
next prior; first-season players use the forward exposure-gated cold-start
prior. The first two historical seasons bootstrap the system before a completed
context function exists.

This sequencing is the essential distinction: history influences the present
only through repeated state updates, not because the model trains a pooled
cross-era context function and applies it retrospectively.

## Frozen Forecast Boundary

The 2025-26 holdout is scored using only:

- Completed 2024-25 player coefficients and cold-start state.
- The completed \(g_{2024-25}\) lineup-context function.
- 2025-26 pre-season player profiles, constructed from data through 2024-25.
- Realized 2025-26 lineups and possession exposure under the leaderboard's
  explicit oracle-allocation contract.

No 2025-26 regular-season or playoff result enters a 2025-26 prediction.
The run fits and stores \(g_{2025-26}\) only after the frozen predictions are
created, so that function is available for the 2026-27 state. A synthetic
three-season regression test enforces this order: the 2025-26 evaluator must
receive \(g_{2024-25}\), not \(g_{2025-26}\).

## First Recursive Exemplar

`artifacts/models/forward_contextual_rapm/2025-26/forward-contextual-rapm-2025-26-20260807T055233Z-b4da0c2c/`
rolls the state from 1996-97 through 2025-26. For this controlled first pass,
each RAPM season uses the already-published forward-RAPM lambda schedule and
each contextual residual uses fixed Ridge alpha 10,000. Contextual alpha tuning
is intentionally deferred so this experiment isolates the state transition.

The frozen 2025-26 forecast uses only the completed 2024-25 contextual model,
2024-25 player state, and 2025-26 preseason profiles.

A dedicated regression test enforces this boundary: the evaluator must receive
`g_2024-25`; the subsequently fitted `g_2025-26` may be stored for 2026-27 but
cannot score the 2025-26 holdout.

| Cohort | Possession RMSE | Possession MAE | Game-margin RMSE | Team NetRtg RMSE | Pythagorean win RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-26 regular season | 1.199008 | **1.141571** | **14.6525** | **4.1572** | **9.3153** |
| 2025-26 playoffs | 1.192898 | **1.135625** | 17.5493 | - | - |

The possession RMSE regression means this is not an unconditional replacement
for every use case. The substantially better game, team NetRtg, and win-total
results make it the project gold standard for its current oracle-lineup
preseason contract. It still requires replication across additional frozen
target seasons before claiming a general performance advantage.

## Outputs

`historical_player_coefficients.parquet` and `season_player_priors.parquet`
record the recursive player state. `season_context_models.joblib` stores one
completed contextual model per season; `season_context_metadata.parquet`
records its fixed alpha and fit diagnostics. Frozen possession, game, team
NetRtg, and Pythagorean-win outputs use the standard leaderboard contract.
The separate `forward_contextual_rankings` artifact records the completed
additive player state used for the following season's public rankings.

<!-- forward-contextual-rankings:start -->
## 2026-27 Player Rankings

These are the top 100 player priors carried from the completed 2025-26 forward contextual RAPM state. They are predictions for the next regular season, not retrospective rankings.

The one-number value is the model's additive player component. The completed lineup-context function `g_2025-26` remains a separate lineup-level term, so it is not assigned to individual players. The table covers players who appeared in 2025-26; it does not yet add the incoming rookie class or account for offseason roster moves.

The table is sortable by every column. `Adjustment` is the completed-season movement from the preseason prior that entered the fit; interpret limited exposure alongside possession count.

Immutable ranking artifact: `artifacts/models/forward_contextual_rankings/2026-27/forward-contextual-rankings-2026-27-20260807T130321Z-442539da`.

| Rank | Player | Pos. | 2026-27 contextual RAPM prior | 2025-26 preseason prior | Adjustment | 2025-26 possessions |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Nikola Jokić | C | +9.43 | +8.16 | +1.26 | 4,786 |
| 2 | Victor Wembanyama | F-C | +6.42 | +1.24 | +5.18 | 3,896 |
| 3 | Shai Gilgeous-Alexander | G | +6.41 | +5.74 | +0.66 | 4,730 |
| 4 | Kawhi Leonard | F | +6.30 | +3.18 | +3.12 | 4,167 |
| 5 | Jimmy Butler III | F | +6.29 | +5.33 | +0.96 | 2,449 |
| 6 | Derrick White | G | +6.07 | +2.32 | +3.75 | 5,180 |
| 7 | Giannis Antetokounmpo | F | +6.05 | +5.03 | +1.02 | 2,129 |
| 8 | Joel Embiid | C-F | +5.53 | +4.52 | +1.00 | 2,479 |
| 9 | Bam Adebayo | C-F | +5.17 | +2.02 | +3.15 | 5,031 |
| 10 | Jrue Holiday | G | +4.88 | +3.84 | +1.04 | 3,295 |
| 11 | Chet Holmgren | C-F | +4.66 | +2.00 | +2.65 | 4,129 |
| 12 | Devin Booker | G | +4.61 | +2.87 | +1.74 | 4,442 |
| 13 | Jayson Tatum | F-G | +4.35 | +4.36 | -0.01 | 1,046 |
| 14 | Lauri Markkanen | F-C | +4.22 | +1.96 | +2.26 | 3,080 |
| 15 | Stephen Curry | G | +4.14 | +6.02 | -1.87 | 2,822 |
| 16 | Aaron Gordon | F | +3.99 | +2.42 | +1.57 | 2,057 |
| 17 | Marcus Smart | G | +3.94 | +1.80 | +2.14 | 3,607 |
| 18 | Donovan Mitchell | G | +3.73 | +2.95 | +0.78 | 4,925 |
| 19 | Jarrett Allen | C | +3.71 | +3.93 | -0.22 | 3,191 |
| 20 | Chris Paul | G | +3.70 | +6.03 | -2.33 | 457 |
| 21 | Cade Cunningham | G | +3.67 | +2.00 | +1.67 | 4,490 |
| 22 | Alex Caruso | G | +3.67 | +3.86 | -0.19 | 2,125 |
| 23 | Franz Wagner | F | +3.66 | +4.37 | -0.71 | 2,190 |
| 24 | Karl-Anthony Towns | C-F | +3.53 | +3.54 | -0.00 | 4,716 |
| 25 | Kon Knueppel | G-F | +3.50 | -0.13 | +3.63 | 5,207 |
| 26 | Davion Mitchell | G | +3.37 | +1.10 | +2.27 | 4,265 |
| 27 | Pascal Siakam | F | +3.29 | +2.96 | +0.33 | 4,322 |
| 28 | OG Anunoby | F-G | +3.27 | +2.35 | +0.92 | 4,531 |
| 29 | Rudy Gobert | C | +3.23 | +3.51 | -0.28 | 4,951 |
| 30 | CJ McCollum | G | +3.17 | +0.41 | +2.76 | 4,762 |
| 31 | Brandon Miller | F | +3.15 | -0.41 | +3.57 | 3,968 |
| 32 | Kevin Huerter | G-F | +3.11 | +1.53 | +1.57 | 3,244 |
| 33 | Zach Edey | C | +3.07 | +0.75 | +2.32 | 588 |
| 34 | Michael Porter Jr. | F | +2.98 | +0.34 | +2.64 | 3,396 |
| 35 | Jalen Smith | F-C | +2.93 | -0.16 | +3.09 | 2,316 |
| 36 | Cedric Coward | G | +2.92 | -0.50 | +3.43 | 3,386 |
| 37 | Tobias Harris | F | +2.91 | +2.74 | +0.18 | 3,604 |
| 38 | Kevin Durant | F | +2.90 | +2.78 | +0.12 | 5,732 |
| 39 | Steven Adams | C | +2.85 | +0.89 | +1.97 | 1,455 |
| 40 | Paul George | F | +2.79 | +3.90 | -1.11 | 2,328 |
| 41 | Luka Dončić | F-G | +2.78 | +2.67 | +0.12 | 4,759 |
| 42 | Collin Gillespie | G | +2.74 | -0.14 | +2.89 | 4,626 |
| 43 | Isaiah Hartenstein | C-F | +2.58 | +1.35 | +1.22 | 2,340 |
| 44 | Dyson Daniels | G | +2.56 | -0.12 | +2.68 | 5,317 |
| 45 | Julius Randle | F-C | +2.49 | +1.67 | +0.82 | 5,470 |
| 46 | Al Horford | C-F | +2.46 | +4.44 | -1.99 | 1,986 |
| 47 | Kyle Lowry | G | +2.44 | +3.86 | -1.42 | 251 |
| 48 | Nickeil Alexander-Walker | G | +2.41 | +0.45 | +1.95 | 5,561 |
| 49 | Jalen Brunson | G | +2.39 | +2.66 | -0.28 | 5,240 |
| 50 | Wendell Carter Jr. | C-F | +2.36 | +1.58 | +0.78 | 4,768 |
| 51 | Herbert Jones | F | +2.31 | +1.95 | +0.36 | 3,289 |
| 52 | Svi Mykhailiuk | G-F | +2.30 | +0.35 | +1.95 | 2,453 |
| 53 | Amen Thompson | G-F | +2.25 | +0.47 | +1.78 | 5,936 |
| 54 | LaMelo Ball | G | +2.25 | -0.72 | +2.96 | 4,101 |
| 55 | Aaron Holiday | G | +2.17 | +1.47 | +0.70 | 1,579 |
| 56 | Dylan Harper | G | +2.17 | +0.05 | +2.11 | 3,207 |
| 57 | Jalen Duren | C | +2.16 | +0.21 | +1.95 | 4,080 |
| 58 | Devin Vassell | G-F | +2.11 | -2.38 | +4.49 | 4,258 |
| 59 | Ausar Thompson | G-F | +2.08 | +0.74 | +1.35 | 3,897 |
| 60 | Julian Champagnie | F | +2.01 | -2.17 | +4.18 | 4,745 |
| 61 | Joe Ingles | F-G | +1.95 | +2.91 | -0.95 | 338 |
| 62 | VJ Edgecombe | G | +1.94 | -0.01 | +1.95 | 5,402 |
| 63 | De'Anthony Melton | G | +1.89 | +0.39 | +1.50 | 2,311 |
| 64 | James Harden | G | +1.85 | +5.62 | -3.77 | 4,936 |
| 65 | Obi Toppin | F | +1.84 | +0.78 | +1.06 | 901 |
| 66 | Evan Mobley | C | +1.82 | +1.98 | -0.16 | 4,315 |
| 67 | Miles McBride | G | +1.79 | -0.24 | +2.03 | 2,172 |
| 68 | Neemias Queta | C | +1.77 | -2.47 | +4.24 | 3,752 |
| 69 | Isaiah Joe | G | +1.77 | +1.57 | +0.19 | 3,108 |
| 70 | Cameron Johnson | F | +1.76 | +1.07 | +0.69 | 3,421 |
| 71 | LeBron James | F | +1.74 | +3.98 | -2.24 | 4,077 |
| 72 | Payton Pritchard | G | +1.74 | +0.32 | +1.42 | 5,014 |
| 73 | Grant Williams | F | +1.73 | -0.85 | +2.59 | 1,438 |
| 74 | Aaron Nesmith | G-F | +1.70 | +1.82 | -0.12 | 2,809 |
| 75 | Desmond Bane | G | +1.68 | +0.10 | +1.58 | 5,768 |
| 76 | Hugo González | G | +1.66 | -1.92 | +3.58 | 2,133 |
| 77 | Coby White | G | +1.63 | +0.72 | +0.91 | 2,587 |
| 78 | Josh Green | G | +1.60 | -1.93 | +3.53 | 1,804 |
| 79 | Jalen Suggs | G | +1.58 | +0.42 | +1.16 | 3,370 |
| 80 | Nique Clifford | G | +1.57 | -1.22 | +2.79 | 3,873 |
| 81 | Dwight Powell | F-C | +1.57 | +0.33 | +1.24 | 1,906 |
| 82 | Duncan Robinson | F | +1.56 | -0.60 | +2.16 | 4,362 |
| 83 | Jaren Jackson Jr. | F-C | +1.52 | +2.12 | -0.61 | 3,071 |
| 84 | Tim Hardaway Jr. | G-F | +1.52 | +1.65 | -0.13 | 4,401 |
| 85 | Immanuel Quickley | G | +1.51 | +1.96 | -0.44 | 4,561 |
| 86 | Ethan Thompson | G | +1.49 | +0.00 | +1.49 | 1,359 |
| 87 | Cooper Flagg | F | +1.48 | +0.00 | +1.47 | 4,943 |
| 88 | Stephon Castle | G | +1.47 | -1.09 | +2.56 | 4,266 |
| 89 | Jakob Poeltl | C | +1.46 | +2.42 | -0.97 | 2,355 |
| 90 | De'Aaron Fox | G | +1.44 | -0.71 | +2.15 | 4,707 |
| 91 | Moussa Diabaté | F | +1.40 | -2.42 | +3.82 | 3,790 |
| 92 | Anthony Davis | F-C | +1.39 | +2.00 | -0.61 | 1,296 |
| 93 | RJ Barrett | F-G | +1.35 | -1.06 | +2.41 | 3,561 |
| 94 | Kevin Love | F-C | +1.30 | +1.44 | -0.14 | 1,306 |
| 95 | Monte Morris | G | +1.26 | +1.72 | -0.46 | 138 |
| 96 | Harrison Barnes | F | +1.26 | +1.08 | +0.18 | 4,124 |
| 97 | Ja Morant | G | +1.25 | +2.53 | -1.28 | 1,238 |
| 98 | Dorian Finney-Smith | F | +1.23 | +3.00 | -1.78 | 1,249 |
| 99 | Scottie Barnes | F-G | +1.22 | -0.69 | +1.91 | 5,512 |
| 100 | Seth Curry | G | +1.19 | +1.05 | +0.14 | 268 |
<!-- forward-contextual-rankings:end -->
