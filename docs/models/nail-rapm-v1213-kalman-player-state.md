# Post-Hoc Kalman Player-State Audit (Invalid Candidate)

*Last updated: 2026-08-24*

This audit tested whether the player component of NAIL should retain an
explicit uncertainty state between seasons. Production NAIL v1.2.1.2 carries a
forward aging-adjusted point prior. This branch treated the completed RAPM fit
as a noisy annual observation and carried both a posterior mean and variance.

It is **not a valid model candidate**. The completed RAPM coefficient is
already a ridge-regularized estimate centered on that season's player prior.
Applying a Kalman gain to that coefficient treats a prior-shrunk posterior as
an independent measurement and shrinks the same signal again. The results are
retained as a diagnostic, not included in the frozen leaderboard or model tree.

## State Update

For player \(i\) in completed season \(t\), \(m^-_{i,t}\) is the exact prior
used in that season's RAPM fit and \(y_{i,t}\) is its completed RAPM estimate:

\[
m^+_{i,t}=m^-_{i,t}+K_{i,t}(y_{i,t}-m^-_{i,t}),
\quad
K_{i,t}=\frac{P^-_{i,t}}{P^-_{i,t}+R_{i,t}},
\quad
R_{i,t}=\frac{4000}{n_{i,t}}.
\]

Low-exposure estimates have larger observation variance \(R_{i,t}\), so they
move the state less. The next-season mean is the existing forward,
value-conditioned aging projection of \(m^+_{i,t}\). The variance transition is

\[
P^-_{i,t+1}=P^+_{i,t}+1.0.
\]

The implementation did not use \(P^-\) to vary the RAPM penalty itself; it
used the filtered posterior mean as a conventional player prior. That is the
source of the invalid observation model described above.

## Information Boundary

The update for season \(t\) is applied only after season \(t\) is complete.
The prior for \(t+1\) therefore uses no \(t+1\) possession, box-score, or
context outcome. Gap-returner transitions similarly start from the last
filtered observed state and use only known biographical inputs during the gap.

## Diagnostic Replay

The recursive fit completed through 2025-26 and the strict replay forecasts
2023-24 through 2025-26 from each matching pre-season state. The incumbent is
the production B2B-control model; all player, profile, lineup-context, and
schedule contracts are identical apart from the filtered player prior.

| Split | Model | Poss. RMSE | Poss. MAE | Eligible game RMSE | Full-game RMSE | Winner accuracy | Team NetRtg RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regular | v1.2.1.2 production | **1.197946** | **1.141313** | **14.0107** | **14.2330** | 67.96% | **3.2847** |
| Regular | v1.2.1.3 Kalman player state | 1.197999 | 1.141539 | 14.0638 | 14.2988 | **68.33%** | 3.4788 |
| Playoffs | v1.2.1.2 production | 1.192710 | **1.137604** | 16.6032 | -- | -- | -- |
| Playoffs | v1.2.1.3 Kalman player state | **1.192666** | 1.137834 | **16.5362** | -- | -- | -- |

The diagnostic has been **invalidated**, rather than merely not promoted. It
produces a small playoff improvement and a higher pooled game-winner accuracy,
but loses every primary pooled regular-season margin metric and substantially
worsens team net-rating RMSE. More importantly, those comparisons are not a
valid test of a Kalman state-space RAPM because of double shrinkage.

The correct follow-up is a precision-weighted prior-centered RAPM: retain the
filtered state mean as the prior, use the state variance to set each player's
ridge precision, and derive the completed posterior directly from that fit.
With equal precision for every player, that implementation must reproduce the
incumbent exactly before any state-space comparison is admitted to the
leaderboard.

## State Audit

`kalman_player_states.parquet` records the prior, observation, posterior, and
variance for every player-season state used while constructing a target prior.
For the state available before 2025-26, the Kalman gain distribution was:

| Percentile | Gain |
| --- | ---: |
| 10th | 0.157 |
| 25th | 0.373 |
| Median | 0.562 |
| 75th | 0.657 |
| 90th | 0.739 |

This confirms the implemented calculation: a low-exposure coefficient receives
only a small update, while a high-exposure coefficient can move the next prior
substantially. It does not validate the calculation as a correctly specified
state-space RAPM observation model.

## Artifacts

- Recursive fit: `artifacts/models/forward_nail_rapm_v1213_kalman_player_state/2025-26/forward-nail-rapm-v1213-kalman-player-state-2025-26-20260824T211727Z-b02c15b2`
- Frozen replay: `artifacts/models/nail_v1213_kalman_player_state_frozen_backtest/frozen_multiseason_backtest/2023-24_to_2025-26/frozen_multiseason_backtest-2023-24-to-2025-26-20260824T214244Z-ae4dd903`
