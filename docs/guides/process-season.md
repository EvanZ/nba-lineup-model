---
last_updated: "2026-08-03"
---

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
provenance sidecar under `data/raw/`. Each endpoint may come from liveData or
Stats V3. When both are retained, Stats V3 is selected because it retains both
sides of historical substitutions.

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

1. Selects and validates both local raw documents and their exact-byte hashes.
2. Adapts any selected Stats V3 documents to the processing boundary.
3. Reconstructs canonical events, lineups, lineup stints, possessions, and
   fixed-lineup possession segments.
4. Applies score, duration, possession, lineup, overtime, and catalog
   invariants.
5. Writes six per-game Parquet tables only after the hard quality gate passes.
6. Returns terminal build and quality records to the flow's single writer.

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
- a missing Stats V3 period-start marker is represented by a derived start
  boundary, and historical blank or reset score placeholders carry forward the
  preceding cumulative score;
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

## Game Rotation recovery

Cached `gamerotation` responses can supply exact-five lineup evidence at a
period start. The processor uses that auxiliary source only when the interval
table has a valid five-player state at the boundary. It then preserves the
play-by-play substitution stream and requires the normal lineup, possession,
score, and catalog validations to pass.

Direct Game Rotation acquisition is suspended because source coverage is
nonuniform and frequently returns HTTP 500. This recovery command never
requests NBA data, but it can use retained cache entries or a documented
external interval export.

### External rotation import

The shared Riley Gisseman export is a local source artifact, not an NBA network
response. Import only latest regular-season failures with exact five-player
states at the starts of periods 2-4:

```bash
uv run nba-import-external-game-rotations \
  data/external/nba_rotations_shared_riley_gisseman.csv \
  --run-id external-rotation-import-r1
```

The importer reads the CSV in chunks, preserves ten-character game IDs, drops
nonpositive intervals, validates the home/away teams against the game catalog,
and emits compatible `data/raw/stats/gamerotation/{game_id}.json` entries. It
does not overwrite an existing valid cached rotation unless `--overwrite` is
explicit. Its report under
`artifacts/reports/external_game_rotation_import/{run_id}/` records the source
file SHA-256, per-game cache state, and every excluded game.

The first full import, `external-rotation-import-r1`, retained 3,119 new cache
entries and reused 705 valid local entries from a 2,107,125-row CSV. Its source
digest is
`sha256:c68403b13f519a685cb953f2101d77210012ff1eaf2dc90b7ee5cfeea63c111b`.
Of 3,777 structurally valid recovery candidates, 1,507 (`39.9%`) passed the
complete reconstruction and quality contract in
`external-rotation-recovery-r1`. The other games remain explicit failures,
most commonly for unbalanced possession counts or source score/possession
anomalies that interval evidence cannot repair.

Reprocess only latest regular-season failures that already have structurally
valid cached rotations:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
  uv run nba-recover-game-rotation \
  --max-workers 4 \
  --run-id game-rotation-recovery-r1
```

This command never requests NBA data. It appends new terminal attempts to the
build ledger, updates the canonical quality rows, and writes a per-game report
to `artifacts/reports/game_rotation_recovery/{run_id}/games.parquet`. A failed
recovery remains excluded; the rotation feed is evidence, not an override of
the quality contract.

The first cached cohort (`game-rotation-recovery-r1`) contained 35 structurally
valid rotations from the failed-game probe. Seventeen games (`48.6%`) passed the
complete reconstruction and quality contract. The remaining 18 were retained as
failures, predominantly because of possession or score defects outside a
period-start lineup ambiguity.

## Resume behavior

A game is skipped only when all of the following agree:

- selected play-by-play and boxscore source names;
- play-by-play, boxscore, and any consumed Game Rotation SHA-256 digests;
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

## Audit-selected historical processing

To materialize only the historical games approved by the reproducible raw-cache
audit, point the processor at its combined `games.parquet` report:

```bash
uv run nba-process-season 2019-20 \
  --season-type regular \
  --audit-games data/audit/historical_regular/games.parquet \
  --max-workers 4
```

The selector accepts only `pass` and `warning` rows for the requested regular
season. It does not reinterpret failed or errored audit rows as eligible. Use
`--audit-offset` with `--limit` to run durable bounded batches.

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
