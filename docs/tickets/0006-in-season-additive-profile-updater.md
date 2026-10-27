# Slice 8: shadow in-season additive-profile updater

GitHub: [#33](https://github.com/EvanZ/nba-lineup-model/issues/33)

## Objective

Build and evaluate a leakage-safe in-season updater for the eight player-level
additive profiles in production NAIL-RAPM. Combine it with the existing
possession-based scalar updater without double counting, and determine whether
the decomposed updater predicts future games better than the scalar updater
alone.

This is an offline and production-shadow study. It must not change public
ratings, game projections, frozen historical snapshots, or promotion status.

## Motivation

The current in-season NAIL updater treats the complete preseason rating as one
scalar prior:

```text
updated_nail_i,t = frozen_preseason_nail_i + scalar_residual_i,t
```

That updater can detect that a player's impact has moved, but it cannot show
whether the evidence comes from shooting, creation, ball security, rebounding,
or defensive events. Production NAIL already attributes part of every player
rating to a strictly lagged additive profile, so updating those player profile
rates is the natural next decomposition.

The experiment must distinguish two different ideas:

1. update each player's current profile rates using new season evidence; and
2. refit the global coefficients that convert profile rates into value.

Only the first is in scope. The production coefficients, feature transforms,
reference field, and additive attribution convention remain frozen. The prior
v1.4 Kalman experiment evolved global coefficients between completed seasons;
it did not implement the nightly player-profile state proposed here and was not
promoted.

## Production profile contract

The first candidate must use exactly the eight production coordinates:

- `three_pa_per_100`;
- `three_pm_per_100`;
- `assists_per_100`;
- `turnovers_per_100`;
- `usage_pct`;
- `steals_per_100`;
- `blocks_per_100`; and
- `offensive_rebound_claim_total`.

The non-additive lineup terms `top_two_assists` and `usage_concentration` are
excluded. They are lineup interactions rather than additive player state.

All names must map explicitly to the corresponding stored production columns,
including the existing `usage_per_100` and `offensive_rebound_pct` aliases.

## Player-profile state

For player `i`, profile `k`, and pregame cutoff `t`, maintain event counts and
the correct exposure denominator rather than averaging already-computed rates.
A conceptual posterior mean is:

```text
profile_i,k,t =
    (prior_exposure_k * preseason_profile_i,k + observed_numerator_i,k,t)
    / (prior_exposure_k + observed_exposure_i,k,t)
```

The implementation must use the proper numerator and exposure for every
coordinate. The existing stat-specific Medvedovsky padding is the starting
prior contract, not an assumption that one common update speed fits all eight
features. A compact historical calibration may scale the prior exposure by
feature, with all choices locked before the final holdout is read.

The state must be strictly chronological:

- only events from games completed before the projection cutoff are eligible;
- a game's observed profile events cannot affect its own prediction;
- same-date or timestamp ordering must not leak later games backward;
- source revisions replace the affected sufficient statistics rather than
  being counted twice; and
- zero-current-season-exposure players reproduce their exact frozen preseason
  profiles.

## Frozen transformation into NAIL value

Let `a_i,0` be the frozen preseason additive-profile contribution and `a_i,t`
the contribution obtained by applying the frozen production transforms,
coefficients, and reference field to the updated player rates. The candidate's
profile movement is:

```text
profile_delta_i,t = a_i,t - a_i,0
```

The standardization and centering reference must remain frozen. Recomputing a
league mean or scale from the partial current season would move every rating
for reasons unrelated to player evidence.

The implementation must reproduce the published preseason additive profile at
zero current-season exposure to floating-point tolerance before any backtest is
accepted.

## Double-count-safe scalar integration

The existing scalar updater already absorbs all unexplained current-season
stint outcomes, including effects correlated with changed box-score profiles.
Simply adding `profile_delta` to its existing residual would double count the
same evidence.

For every historical game contribution, reconstruct the profile state that was
available before that game and subtract that game's profile delta from the
stint target before accumulating scalar sufficient statistics. The combined
candidate is:

```text
combined_nail_i,t =
    frozen_preseason_nail_i
    + profile_delta_i,t
    + profile_residualized_scalar_delta_i,t
```

The immutable game-contribution lineage must identify the exact pregame
profile-state checksum used to build each residualized contribution. A profile
source correction must invalidate and replace every downstream contribution
whose pregame state changes.

## Historical study design

Compare four arms on identical games, cutoffs, minutes assumptions, and
non-player context:

1. frozen production NAIL;
2. the existing scalar possession updater;
3. the player-profile updater without a scalar residual; and
4. the player-profile updater plus the profile-residualized scalar updater.

Use chronological historical development seasons for prior-strength selection
and keep the final 2025-26 season locked as the untouched holdout. The 2026-27
season is monitoring evidence only and must not select hyperparameters.

Report at minimum:

- possession residual RMSE;
- full-game margin MAE and RMSE;
- Brier score and log loss;
- winner accuracy and calibration;
- ATS record as secondary descriptive evidence;
- movement and uncertainty by profile coordinate; and
- paired game-level uncertainty for every candidate-versus-control difference.

The report must include interpretable player trajectories for Stephen Curry,
at least one rookie or cold-start player, one high-usage role changer, and one
player whose profile update worsens the forecast. Each trajectory must separate
the eight profile contributions, the residual scalar contribution, and total
NAIL rather than showing only a final rating.

## Data locality and artifacts

- Run full calibration and replay beside the S3 lake on EC2.
- Read only frozen model artifacts, existing profile inputs, and incremental
  current-season CDN/play-by-play or box-score data.
- Do not copy a large dataset from S3 to the local machine without Evan's
  explicit approval after stating the source, destination, size, and purpose.
- Persist immutable, checksummed per-game profile sufficient statistics,
  dated player-profile states, residualized scalar contributions, and reports.
- Make the final report available through a browser URL and state the verified
  findings directly to the user; artifact files alone are not a deliverable.

## Required audits

- Exact zero-evidence reconstruction of all eight preseason player profiles and
  their published additive contribution.
- Synthetic one-event tests for every numerator and exposure denominator.
- Strict pregame, same-date, and corrected-source leakage tests.
- Incremental replay parity with a clean full chronological rebuild.
- Idempotent reruns that create no duplicate event or game evidence.
- Fixed reference-field and scaling checks across every cutoff.
- An algebraic reconstruction showing that combined NAIL equals frozen NAIL
  plus profile movement plus residualized scalar movement.
- A negative-control test demonstrating the inflation caused by naively adding
  profile movement to the unmodified scalar updater.

## Acceptance criteria

- Unit and integration tests cover the player state, feature aliases, temporal
  boundary, corrections, immutable lineage, and double-count prevention.
- Hyperparameters are chosen without reading the locked holdout.
- All four arms are evaluated on identical eligible games with explicit sample
  sizes and exclusions.
- The report shows aggregate metrics, paired uncertainty, and the required
  player-level decompositions.
- The user receives a concise written conclusion explaining whether profile
  updating adds value beyond the scalar updater and where it helps or hurts.
- No production rating or public projection changes without a separate explicit
  approval after reviewing the evidence.

## Non-goals

- Nightly refitting of the production additive-profile coefficients.
- Adding or removing profile coordinates.
- Updating non-additive lineup interactions, home court, B2B, or other context
  coefficients.
- Replacing the separate in-season minutes updater.
- Promoting an updater based on the small 2026-27 opening-week sample.

## Implemented result

The leakage-safe updater and three-season replay were completed on 2026-10-26.
The implementation:

- reproduces all eight frozen 2026-27 production profile coordinates exactly at
  zero current-season exposure;
- updates event counts against their proper possession, attempt, opportunity, or
  minutes-weighted exposure;
- preserves the production three-point volume-times-accuracy construction;
- keeps the production coefficients and reference field frozen;
- date-batches predictions before incorporating that day's final games;
- content-addresses profile contributions and state snapshots; and
- residualizes the scalar updater against the profile movement before combining
  the two signals.

The common prior multiplier was selected on 2,281 games from 2023-24 and
2024-25. The finite grid preferred its most conservative value, `8x`, but the
scalar-only null still had lower development-season RMSE, Brier score, and log
loss. This is evidence for stronger damping, not evidence that `8x` adds value.

On the locked 1,230-game 2025-26 holdout:

| Arm | Margin MAE | Margin RMSE | Winner accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Frozen production | 11.3370 | 14.4143 | 68.37% | 0.203118 | 0.596013 |
| Scalar updater | 11.0692 | 14.0853 | 70.89% | 0.196627 | 0.580610 |
| Profile only | 11.3075 | 14.3568 | 68.86% | 0.202605 | 0.594409 |
| Profile + residualized scalar | 11.0685 | 14.0653 | 70.41% | 0.196885 | 0.580810 |
| Naive profile + scalar | 11.0876 | 14.0776 | 70.16% | 0.197621 | 0.582525 |

The combined candidate changed margin MAE by only `-0.0008` points versus the
scalar updater (paired 95% interval `[-0.0367, +0.0325]`). Its Brier score was
`+0.000258` worse and log loss was `+0.000200` worse; both paired intervals
crossed zero. Winner accuracy also declined by 0.49 percentage points. The
naive negative control was directionally worse on all three losses, supporting
the need for residualization.

### Decision

Do not promote the additive-profile updater. The decomposition is useful for
interpretability, but it has not demonstrated predictive value beyond the
existing possession-based scalar updater. Production serving remains unchanged.
The candidate may remain an offline research artifact; a new promotion ticket
requires materially better out-of-sample evidence.

The compact historical replay did not preserve a universal historical closing
line or the scalar needed to reconstruct exact possession-residual SSE. ATS and
possession-residual RMSE are therefore explicitly unavailable for this study
rather than being estimated from incomparable data.
