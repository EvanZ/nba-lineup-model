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

The checkpoint and adapter tests do not by themselves prove the Nov 2 repair.
Acceptance still requires the approved real data/model work, full API coverage,
and browser inspection of all affected played/missed-game rows and chart
points. The general daily Data Quality report remains #58.
