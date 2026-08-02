# Player Bios

Player reference data comes directly from the NBA Stats service. It uses bulk
endpoints rather than an open-source NBA API wrapper.

## Sources

| Endpoint | Scope | Normalized use |
| --- | --- | --- |
| `PlayerIndex` | Historical NBA player universe | Identity catalog |
| `LeagueDashPlayerBioStats` | Players appearing in one season type | Player-season bios |

The public [NBA player bio table](https://www.nba.com/stats/players/bio)
exposes the corresponding age, height, weight, college, country, and draft
fields.

Both response bodies are preserved byte-for-byte:

```text
data/raw/playerindex/2025-26.json
data/raw/playerindex/2025-26.meta.json
data/raw/leaguedashplayerbiostats/2025-26/regular.json
data/raw/leaguedashplayerbiostats/2025-26/regular.meta.json
```

The sidecars record endpoint, exact query parameters, URL, fetch time, and
SHA-256 digest. A changed query contract does not reuse an incompatible cache.

## Historical player catalog

`data/catalog/players.parquet` contains one integer `player_id` row per
historical player. It includes:

- first, last, display, and slug names;
- latest listed position, height, weight, team, and jersey;
- college and country;
- normalized draft year, round, number, and undrafted status;
- first and last NBA career years;
- roster and source status;
- source season, URL, fetch time, and exact response hash.

Latest listed team and physical fields are source snapshots, not historical
truth for every season.

Collection updates the catalog monotonically. Player IDs from an existing
newer catalog are preserved during an older-season backfill, and the row from
the latest source season wins when the same player appears in both snapshots.

## Player-season bios

The season partition is:

```text
data/curated/player_seasons/2025-26/regular/
  _manifest.json
  part-00000.parquet
```

Every row includes `season` and `season_type` directly. Physical and draft
fields use nullable integer columns. Raw height remains available alongside
validated `height_inches`.

NBA source sentinels are normalized explicitly:

- `Undrafted` becomes `is_undrafted=true` with nullable pick fields;
- numeric zero round or number also identifies an undrafted source record;
- missing draft history remains unknown rather than automatically undrafted;
- historical season rows can omit height, which is preserved as null rather
  than inferred from another season;
- missing weight remains null;
- source text such as accented names remains Unicode.

The manifest records ordered player IDs, both raw source hashes, part size,
part hash, and exact row count.

## Leakage policy

The season endpoint also returns GP, PTS, REB, AST, net rating, usage, shooting,
and rebounding percentages. They are retained in raw JSON but excluded from the
normalized bio table.

Those aggregates summarize the full requested season and therefore cannot be
used as static predictors for games within that same season. Future rolling or
prior-season features require a separate as-of-time contract.

## Current baseline

For 2025-26 regular-season data:

| Contract | Rows |
| --- | ---: |
| Historical player catalog | 5,204 |
| Regular-season player bios | 582 |
| Unique players who appeared in curated regular-season games | 582 |

Six bio rows have no listed weight. All 582 have a validated height, and bio
player IDs reconcile exactly with the players who appeared in the game data.
