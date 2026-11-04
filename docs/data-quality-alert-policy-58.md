# Issue #58 alert policy proposal

Status: **not activated**. This document proposes monitoring behavior for user
approval. It does not change any Dagster schedule, collector cadence, paid API
usage, production service, or notification destination.

## Separation of responsibilities

The data-quality probe runs independently of Dagster and reads:

- immutable release and activation identities;
- the API surfaces actually served by the selected environment;
- official NBA schedule, injury, odds, and score artifacts;
- Dagster's persisted automation and run states.

Dagster may provide a supplemental run-status sensor later, but it cannot be
the only watchdog for its own orphaned QUEUED or STARTED runs.

## Proposed evaluation cadence

Run the watchdog **once per hour**. Collector polling remains unchanged. An
hourly watchdog does not cause an additional odds API request because the probe
only reads local persisted artifacts.

## Proposed incident thresholds

| Scope | Proposed incident condition | Quiet condition |
|---|---|---|
| Schedule + injury collector | No successful run or refreshed schedule artifact for 90 minutes; a queued/started run older than 45 minutes | None during the regular season |
| Injury report view | Schedule collector is healthy but today's/tomorrow's served report artifact has not refreshed for 90 minutes | No games today or tomorrow |
| Odds | No successful run/artifact within 30 minutes after an existing configured odds poll time | No scheduled poll is due; never alert merely because the last poll was overnight |
| Official NBA scores | A game is live and the official score artifact has not refreshed for 15 minutes; queued/started run older than 10 minutes | No official NBA game is live |
| Daily model/site update | Latest eligible completed slate is newer than the active served evidence date after the approved nightly grace window | Zero-game date or slate still in progress |
| Activation | Staging/activating state exceeds 20 minutes, or active API release/component identity differs from the activation | Fully active matching release |

The exact nightly grace window is deliberately left unset until the user
chooses it. The report already derives the latest eligible completed slate from
the official schedule, so no date or game-count constant is required.

## Notification behavior

- Notify once when a new error fingerprint opens.
- Stay silent while the same error remains unchanged.
- Notify once when that fingerprint recovers.
- A materially different scope/code opens a separate incident.
- Keep the full finding in every browsable report even when no repeat
  notification is emitted.
- Report-generation failure itself is an incident, and its failed HTML/JSON
  report remains under the stable report root.

## Activation decision still required

Before enabling an hourly service/timer or any external notification, approve:

1. the thresholds above;
2. the nightly grace window;
3. the notification destination.

Until then, dev reports may be generated manually and include collector
timestamps/run states, but age thresholds are not applied.

