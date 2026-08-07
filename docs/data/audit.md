---
last_updated: "2026-07-26"
---

# Cross-Season Audit

The audit evaluates reconstruction behavior across a versioned game manifest.
It writes compact reports instead of all processed game tables.

## Committed matrix

`config/audit_manifest.json` currently contains 21 games from 2019-20 through
2025-26:

- one same-ordinal regular-season game per season;
- one Finals opener per season;
- one confirmed overtime game per season;
- two double-overtime games within the overtime stratum.

## Exact checks

Each reconstructed game is evaluated for:

- final event score matching the boxscore;
- possession points matching the final score;
- segment points matching every parent possession;
- segment durations matching every parent possession;
- home and away possession counts differing by at most one within each period;
- expected overtime status;
- structured lineup, possession, and segment errors.

## Diagnostic checks

The traditional estimate

```text
FGA + 0.44 * FTA - ORB + TOV
```

is recorded for comparison. It is approximate and does not define the exact
possession count. A difference above the configured tolerance produces a warning,
not a failure.

## Status values

| Status | Meaning |
| --- | --- |
| `pass` | All exact checks pass and no warnings are present |
| `warning` | Exact checks pass with pipeline or diagnostic warnings |
| `fail` | Reconstruction completes but an exact invariant fails |
| `error` | Fetching, reconstruction, or audit evaluation raises |

## Reports

`data/audit/games.parquet` contains one row per manifest game. It includes counts,
check booleans, issue codes, and exception details.

`data/audit/summary.parquet` aggregates status and volume metrics by season,
season type, and sample group.

One game failure does not abort the run.
