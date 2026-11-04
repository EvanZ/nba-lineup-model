# Resumable daily partitions (#56)

This implementation extends the existing daily data, coupled model, serving,
lineup, and release builders. It does not change their basketball algorithms.
No production deployment, real repair, or causal replay is authorized by the
code itself.

## Explicit daily work

A work specification identifies the Pacific date, exact official schedule game
set, source revisions, model/configuration identity, parent model, season
edition, sealed input file references, and code revision. A different input
receives a different work identity. Live score/odds/injury polling does not
mutate the selected historical work specification.

The worker executes data, model, serving, lineups, and release validation in
order. Each stage receives an immutable success receipt containing checksummed
outputs. A retry verifies earlier receipts before reusing them. Corrupt outputs
fail visibly rather than being silently accepted or replaced. Failures retain
their reports and do not invalidate another game's source capture.

Private work directories may contain discovery pointers used by established
builders. They are seeded from the explicitly selected immutable artifacts,
never from an independently advancing latest model. They are not serving
pointers. Only the existing #54 publisher may activate the resulting #55-bound
release.

## Observation and correction boundary

Official NBA raw captures are stored once by checksum. Normalized revisions
retain their capture, parser identity, game/feed identity, and prior revision.
Capturing a correction does not activate it. Selecting a historical correction
requires the exact impact plan and the recorded user approval.

Impact plans identify games/dates, changed inputs, affected outputs, any causal
suffix, the base release/edition, and whether forecasts would be replaced.
Plans can be inspected before approval. An approval for one plan or operation
does not authorize a broader replay or historical forecast replacement.

The pinned daily-data builder consumes NBA box, play-by-play, injury, and
inactive-list revisions without requiring a frozen forecast. Missing forecasts
remain serving/metric coverage errors under #55; they do not prevent valid
observations from being processed. Every scheduled game must be present before
causal model advancement. An explicit zero-game date is a valid empty partition.

## Execution and evidence

The standalone `partition_definitions` Dagster code location imports no legacy
all-in-one definitions. Its sensor is stopped by default, permits one in-flight
daily job, retains the existing five-minute evaluation / fifteen-minute retry
cooldown, and stops after three failed attempts. It consumes explicitly pinned
request files; it does not poll paid feeds or activate releases.

`scripts/inspect_daily_partition_gap.py` creates a read-only impact report for
missing days after a selected release. It inspects captured source coverage but
does not claim those sources have passed stint/model validation. The report
must be reviewed before executing the real historical scope.

## Approved dev exercise: 2026-11-02

The user approved the dev-only missing-day scope for all 15 Nov 2 games, with
no historical forecast or market replacement. Inspection found that official
data and the 107-game model checkpoint already existed; the failed boundary was
release validation/serving, not missing model training. The worker verified and
reused those artifacts, including the exact accepted Nov 1 parent and model
configuration. No historical model recomputation or new ingestion was needed.

- Data: `cf84c8df9187814a91f0f1c45a1580710c8f3b3cf0807da2ef84af6647a948c1`.
- Model: `bb54efe125b7da7ba8d908d95e2a2c1a6829edcfceecb08b351f02aa12aa2df2`.
- Work: `34c769474192641766d40eec4af37ca5673546f6253632e04533d61f22ac3bd6`.
- Successful Dagster run: `c779eac2-abf9-41eb-b615-29cce191afbe`.
- Unchanged retry: `698b7316-08bb-4e64-9c1b-6f4a33d8dbc0` reused all five
  stages in 4.49 seconds without rebuilding outputs.
- Dev runtime:
  `01996cf8ba671f57ba414755ac0e4ed9b21613f442e72b0541b231b730b0ed73`.
- Application code: `bd6a8100621158e6fc9a8e454fe3dc0ee284dd6a`.

The first attempt surfaced a validator that required the legacy `frozen/`
directory despite the complete, verified #55 forecast-selection map. The
validator now accepts that canonical map and fails closed on tampering.
Historical forecasts were not reconstructed or substituted. The selected
forecast/market map checksum and all historical performance windows remained
identical before and after the activation.

Verification covered 107 completed games, all 15 newest-day games, 3,675
player-game opportunities (including 1,262 missed-game rows), 528 rankings,
2,659 lineups, and 214 completed team-games in Wins. All 523 affected player
pages passed rendered game-log and trajectory-point checks: 334 played and
189 did not. Curry's Nov 2 PHX row and chart point are present. The focused
pipeline/release suite passed 177 tests.

These are coverage results, not an unconditional UI sign-off. At 768 pixels,
the unchanged frontend header overflows horizontally; 1440 and 390 pixels
had no page overflow. Curry's raw +/− is blank in the player API despite +21
in the model trajectory. NBA logo requests failed with HTTP/2 protocol errors
in the EC2 browser, so logo rendering was not validated. These findings were
disclosed without altering the UI, historical artifacts, or model inputs.

The review report is served on dev at `/api/reports/daily-pipeline`; it lists
every affected game and player, exact artifacts, tests, and unresolved findings.
The general recurring daily Data Quality report remains #58.

## Packaging and activation boundary

Use a fixed detached code checkout when preparing/staging a release. The
development branch is not a valid long-lived runtime dependency. The approved
dev activation used:
`/opt/nba-gestalt-dev/releases/bd6a8100621158e6fc9a8e454fe3dc0ee284dd6a`.

The #54 publisher retains its root-owned lock and activation boundary. An
unprivileged publication attempt failed before changing the live pointer.
Git ownership checks were satisfied with command-scoped safe-directory entries
for the exact new and rollback release checkouts, not a wildcard or global
setting. Activation then completed with post-activation full API validation.

The new partition sensor remains stopped and uninstalled; this exercise used
the standalone Dagster job, not an automatic nightly production cutover.
Production and all three protected collection services were left unchanged.
Do not infer authorization to promote, start the new sensor, push publicly,
or post a public issue summary from this dev-only exercise.
