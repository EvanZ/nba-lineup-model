---
last_updated: "2026-08-12"
---

# Historical Modeling Coverage

Historical raw availability and historical model eligibility are different
contracts. This audit compares each final regular-season game in the catalog to
the games actually present in the RAPM stint input. It is therefore the
coverage report that applies to historical player ratings, lineup ratings, and
the player-season modeling panel.

## Definitions

- **Catalog games**: final regular-season games in `data/catalog/games.parquet`.
- **Modeled games**: distinct games represented in
  `data/analytical/rapm_stints/<season>/regular/`.
- **Modeled GP**: a player's positive-minute box-score appearances within that
  modeled subset. It is not official NBA games played.

The audit assigns `pass` at at least 95% coverage, `warning` from 90% through
less than 95%, and `critical` below 90%. The threshold applies to both the
season and each team-season row.

This distinction matters for player profiles. For example, a historical
profile's displayed `GP` must be described as **Modeled GP** unless it is
replaced with a separately acquired authoritative player-season totals source.

## Artifacts

Run the audit after building or changing historical RAPM stints:

```bash
uv run nba-audit-modeling-coverage
```

It writes the following ignored, shared-workspace artifacts under
`data/audit/historical_modeling_coverage/`:

| File | Contents |
| --- | --- |
| `season_coverage.parquet` | Catalog and RAPM-input game coverage for every regular season. |
| `team_coverage.parquet` | The equivalent team-season coverage table. |
| `manifest.json` | Thresholds, source paths, and SHA-256 hashes for the catalog and per-season input manifests. |

The manifest is the provenance boundary. A training or publishing workflow can
read it and reject seasons or team cohorts below its own minimum threshold.

## 2017-18 Recovery

The first audit exposed an avoidable processing gap in 2017-18: only 903 of
1,230 catalog games reached the RAPM stint dataset, despite complete V3 raw
play-by-play and box-score coverage. This was a modeled-data limitation, not a
source-availability limitation.

The recovery reprocessed the 327 excluded games with two bounded corrections:

- Historical V3 scoreboard values are made monotone when a later administrative
  record carries a lower cumulative score than an earlier record. The raw JSON
  remains unchanged; the adapted event stream carries the preceding score
  forward rather than emitting a negative score delta.
- A per-period possession-count imbalance is now a warning when event scores,
  possession scores, lineup-segment scores, and segment durations all conserve.
  Those hard conservation failures still exclude the game.

This recovered 266 games. The current 2017-18 RAPM input has **1,169 of 1,230
games (95.0%)**. Golden State improved from 54 to 77 modeled games and Chicago
from 65 to 80. Stephen Curry's modeled positive-minute appearances rose from
31 to 46; Lauri Markkanen's rose from 54 to 66. The remaining 61 games retain
hard reconstruction or source-data failures and are not published to the
curated subset.

The persistent audit artifacts above now record the repaired manifests and
coverage. The same recovery can be applied season by season to the other
historical gaps before any new long-horizon model refit. See
[Historical Source Coverage](historical-source-coverage.md) for the separate
raw-source contract.

To recover one or more seasons, the following command selects only catalog
games that are absent from the current RAPM stints, force-reprocesses those
games from the local raw cache, compacts the successful pass/warning subset,
and rebuilds the season's RAPM stints. It writes a durable per-season result
manifest under the audit directory.

```bash
uv run nba-recover-modeling-coverage \
  --season 2016-17 \
  --season 2017-18 \
  --max-workers 4 \
  --run-id historical-coverage-recovery-r1
```

Run `uv run nba-audit-modeling-coverage` after a recovery batch. Refit a
historical model only after all of the seasons it consumes have been rebuilt.

## Current Audit

The two local-cache recovery batches rebuilt every previously incomplete
regular-season RAPM partition. They recovered **1,990** games, raising the
historical RAPM input from 32,620 of 35,546 catalog games (91.77%) to **34,610
of 35,546 (97.37%)**. Twenty-five of 30 seasons now pass the 95% threshold.

The five remaining below-threshold seasons are not raw-data gaps: 2019-20 is
88.76%, while 2021-22 through 2024-25 range from 92.52% to 93.41%. Their
remaining excluded games have hard score-conservation, ambiguous-period-lineup,
or contradictory substitution evidence. They remain visible in the append-only
build ledger and quality report, and are deliberately absent from model input.

The recovery manifests are ignored workspace artifacts at:

```text
data/audit/historical_modeling_coverage/recovery_runs/
```

Each records the selected missing games, recovered builds, unresolved builds,
curation outcome, and rebuilt RAPM-stint manifest for every season.
