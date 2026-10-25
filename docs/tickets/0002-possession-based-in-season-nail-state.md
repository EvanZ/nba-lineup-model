# Possession-based in-season NAIL state updater

Status: complete
GitHub: [Issue #29](https://github.com/EvanZ/nba-lineup-model/issues/29)

## Decision record

The frozen 2026-27 NAIL-RAPM v1.2.1.3 forecast is the complete prior mean. For
player `i`, the nightly state is `NAIL_i,t = mu_i,0 + delta_i,t`; the scalar
aging prior and additive profile are not separated or recalculated in season.

Current-season evidence uses raw stint possessions. Each completed game is
reduced to sparse `X'WX` and `X'Wy` contributions after subtracting the frozen
player prior and all frozen non-player effects. A fixed equivalent-possession
prior is added to the cumulative current-season precision. The possession
weights are never normalized to a completed-season sample size.

Only games finalized strictly before a projection cutoff are eligible. A source
revision replaces that game's prior contribution through a game-keyed ledger;
it is never counted twice. Both game contributions and dated player states are
content-addressed and immutable.

## Data locality

- S3 is the durable source of truth and publication target.
- Calibration and replay run beside S3 on the existing EC2 instance.
- Nightly execution reads only the frozen artifacts, manifests, and new or
  revised current-season games.
- Local work uses fixtures and deliberately bounded samples.
- Any large S3-to-local transfer requires Evan's explicit prior approval after
  stating the estimated size, exact source scope, destination, and purpose.

## Implemented first slice

- Prior-centered, possession-weighted sufficient-statistic builder.
- Strict loader for the exact promoted preseason forecast, including validation
  that the published rating equals scalar prior plus additive profile.
- Strict finalized-before-cutoff cumulative solver.
- Game-keyed add/replace ledger for idempotence and corrections.
- Posterior state fields for exposure, information, weights, and uncertainty.
- Immutable local artifact contracts for per-game contributions and player
  state checkpoints, ready for the S3 publication adapter.
- Unit coverage for zero-possession identity, possession accumulation, leakage
  boundaries, correction replacement, incremental/full parity, and immutable
  round trips.
- Chronological historical calibration over configurable equivalent-possession
  strengths, with selection seasons separated from a final holdout season.
- Margin MAE/RMSE, winner accuracy, and fixed-link Brier/log-loss reporting
  against the unchanged frozen-preseason control.
- A checksummed, compressed sparse replay mart that allows subsequent prior
  grids to run without rereading historical stint data.
- A daily shadow Dagster job scheduled for 8:45 AM Eastern, after the morning
  CDN and final-archive reconciliation jobs. It builds only new or revised
  game contributions, reuses the local sparse cache, publishes immutable
  states and reports to S3, and never changes the serving projection path.
- A cumulative live-season control comparison covering margin MAE/RMSE,
  Brier score, log loss, winner accuracy, ATS record, projection movement,
  and explicit contribution exclusions.

## Historical calibration result

The refined offline run selected a 6,000-equivalent-possession prior on pooled
2023-24 and 2024-25 results. On the untouched 2025-26 holdout, the selected
updater improved margin RMSE from 14.414 to 14.085, winner accuracy from 68.4%
to 70.9%, and fixed-link Brier score from 0.2031 to 0.1966. The reconstructed
frozen baseline matched all 1,230 published 2025-26 game margins to machine
precision (maximum absolute difference below 6e-15). This is evidence for
continuing the offline updater work, not approval to change the production site.
The reproducible local run is
`in-season-nail-calibration-20261025T161109Z-ff8a4ba0`. Its immutable,
checksum-verified artifacts are published at
`s3://nba-gestalt-data/lake/analytical/in_season_nail_updater_calibration/run_id=in-season-nail-calibration-20261025T161109Z-ff8a4ba0`.

## Closure state

- The deployed preseason catalog safely extends the frozen player vocabulary
  for newly rostered players without changing any existing frozen row.
- CDN late corrections are semantically reordered while preserving score and
  event provenance, so all 36 completed games through October 25 reconstruct.
- The production shadow replay has 36 validated contributions and zero
  exclusions; a second replay read all 36 from cache and created none.
- The reviewed historical calibration is durably published to S3.
- The public projection path remains unchanged. Continued shadow monitoring
  and any decision to promote the updater are explicit follow-on work, not part
  of this implementation ticket.

## Shadow operations

The `in_season_nail_shadow_daily` schedule runs at 8:45 AM Eastern and scores
games through the prior date. Its expensive work is incremental: a game whose
current CDN checksum already has a validated sparse contribution is not fetched,
reconstructed, or rescored through the contextual endpoint. The chronological
comparison itself is regenerated from the compact contributions so corrections
remain reproducible.

The initial EC2 validation reconstructed 32 of 36 completed games. The final
production validation on release `gestalt-web-da1952b` reconstructed all 36
games with zero exclusions. Its immediate idempotence replay reused all 36
cached contributions, created none, and completed successfully in 12.6 seconds.
Through October 25, the shadow updater improved margin MAE by 0.145, margin
RMSE by 0.100, and Brier score by 0.0035 versus the frozen production control;
the sample remains too small for promotion.
