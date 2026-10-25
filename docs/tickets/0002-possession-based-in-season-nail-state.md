# Possession-based in-season NAIL state updater

Status: in progress  
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

## Remaining production wiring

1. Materialize current-season lineup stints from the CDN lake plus the required
   box-score/roster inputs.
2. Load the exact promoted 2026-27 player prior and frozen context artifacts.
3. Add S3 readers/writers and compact replay-mart manifests.
4. Calibrate equivalent-possession strength on recent historical seasons on
   EC2, then publish a separate immutable updater artifact.
5. Add resource-bounded Dagster calibration/replay and nightly update jobs.
6. Run a chronological offline 2026-27 replay and approve promotion before the
   public site consumes any in-season state.
