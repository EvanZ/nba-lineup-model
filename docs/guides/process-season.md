# Process a Season

The season processing flow converts validated local raw responses into
per-game event, lineup, possession, and possession-segment Parquet tables.
It does not make network requests.

## Prerequisites

Discover and fetch the season first:

```bash
uv run nba-discover-season 2025-26
uv run nba-fetch-season 2025-26 --max-workers 4
```

Every selected game must have valid play-by-play and boxscore JSON plus its
provenance sidecar under `data/raw/`.

## Representative pilot

Sample up to three games from every season-type and overtime stratum:

```bash
uv run nba-process-season 2025-26 \
  --sample-per-stratum 3 \
  --seed 7 \
  --max-workers 4
```

For the 2025-26 catalog this selects 27 games, including regulation and
overtime examples where available across preseason, regular season, the NBA
Cup final, All-Star, play-in, and playoffs.

## Full season

```bash
uv run nba-process-season 2025-26 --max-workers 4
```

The flow creates one Prefect task per final catalog game. Each task:

1. Validates both local raw documents and their exact-byte hashes.
2. Reconstructs canonical events, lineups, lineup stints, possessions, and
   fixed-lineup possession segments.
3. Applies score, duration, possession, lineup, overtime, and catalog
   invariants.
4. Writes six per-game Parquet tables only after the hard quality gate passes.
5. Returns terminal build and quality records to the flow's single writer.

Warnings remain usable and visible in the quality report. Hard failures do not
count as successful builds and do not stop unrelated game tasks.

## Source anomaly policy

The processor preserves raw source fields and records narrow, named warnings
when a final NBA feed requires deterministic reconciliation:

- consecutive substitutions are atomic only at the same period and clock;
- an impossible full-clock period-start batch may be reconciled from declared
  entrants and early on-court actors;
- an exact replay of an already-applied team substitution is idempotent;
- boundary events may precede a delayed period-start marker at the same clock;
- a nonmonotonic source clock retains its raw clock and order while derived
  elapsed time is clamped to the preceding event and flagged;
- held-ball companions, possession-retaining foul shots, and same-play clock
  drift are handled without creating phantom possessions.

A period possession-count imbalance remains a hard failure unless every
imbalanced period contains a flagged nonmonotonic source clock. In that narrow
case the record remains `balanced_possession_counts=false`, receives
`audit:unbalanced_period_possession_counts_with_clock_anomaly`, and is usable as
a warning that downstream consumers can exclude.

## Outputs

Per-game reconstruction tables are written to:

```text
data/processed/
  events/{game_id}.parquet
  players/{game_id}.parquet
  event_lineups/{game_id}.parquet
  lineup_stints/{game_id}.parquet
  possessions/{game_id}.parquet
  possession_segments/{game_id}.parquet
```

Operational and quality metadata are written to:

```text
data/manifests/builds.parquet
data/quality/games.parquet
data/quality/summary.parquet
```

The build ledger is append-only. The game-quality report keeps the latest
validated result for each game, while the summary aggregates by season and
season type.

## Resume behavior

A game is skipped only when all of the following agree:

- play-by-play and boxscore SHA-256 digests;
- a fingerprint of processing-owned reconstruction and quality source;
- a prior successful build-ledger record;
- a prior passing or warning quality record;
- six readable, non-empty Parquet outputs containing the expected lexical
  `game_id`.

Any mismatch rebuilds the game. Use `--force` to bypass the resume check.

The flow checkpoints quality records and terminal build records every 25 games
by default. Change the interval with `--checkpoint-size`. After an interruption,
a new run can reuse completed checkpoints and rebuild any uncheckpointed work.

## Validation baseline

The 2025-26 final catalog currently contains 1,400 games. A full same-fingerprint
run validates:

| Result | Count |
| --- | ---: |
| Pass | 956 |
| Warning | 444 |
| Fail | 0 |
| Error | 0 |
| Per-game Parquet tables | 8,400 |

The reconciliation checks current source hashes and code fingerprints, all six
expected tables per game, non-empty files, and the expected lexical `game_id`.

## Selection

Filter season types or game IDs with repeatable options:

```bash
uv run nba-process-season 2025-26 \
  --season-type regular \
  --season-type playoffs

uv run nba-process-season 2025-26 \
  --game-id 0022500001
```

Use `--limit N` for a deterministic prefix or `--sample-per-stratum N` for a
representative pilot. Those two options are mutually exclusive.

## Prefect UI

Start the persistent local UI:

```bash
uv run prefect server start
```

Then run the processor from another terminal:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-process-season 2025-26 --max-workers 4
```

See [Use the Prefect web UI](prefect-ui.md) for persistent profile settings.
