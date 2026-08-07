---
last_updated: "2026-07-27"
---

# Season Discovery

Season discovery creates the canonical game inventory before any game-level
play-by-play processing begins.

## Direct NBA source

Historical discovery calls the NBA Stats schedule endpoint directly:

```text
GET https://stats.nba.com/stats/scheduleleaguev2
    ?LeagueID=00
    &Season=2025-26
```

The public NBA schedule page uses a regional CDN document such as
`scheduleLeagueV2_1.json`. That file is useful for the current schedule but its
URL is not season-addressable. The NBA Stats endpoint accepts an explicit
`Season` value and is therefore the source for repeatable historical discovery.

The project does not use an NBA client package. It owns the HTTP request, result
set parsing, cache, and canonical mapping.

## Discover a season

```bash
uv run nba-discover-season 2025-26
```

The command writes:

```text
data/raw/scheduleleaguev2/
  2025-26.json
  2025-26.meta.json

data/catalog/
  games.parquet
```

The raw JSON contains the exact response bytes. Its sidecar records the season,
request URL, UTC fetch time, and SHA-256 digest. A valid cached response is
reused by default.

Use `--refresh` to request a new source response:

```bash
uv run nba-discover-season 2025-26 --refresh
```

## Catalog update semantics

`games.parquet` is a multi-season catalog. Discovering a season replaces all
existing rows for that season and retains every other season. This makes a
repeated discovery idempotent and lets a corrected NBA schedule remove as well
as add games.

Use `--replace-catalog` only when the output should contain the newly discovered
season and nothing else.

## Source mapping

The normalizer validates the nested `leagueSchedule`, game-date, game, and team
objects and rejects malformed rows or source seasons that differ from the
request.

| Source evidence | Canonical field |
| --- | --- |
| `gameId` | Lexical ten-character `game_id` |
| `seasonYear` | `season` |
| `gameDateEst` | League schedule `game_date` |
| `gameDateTimeUTC` | Optional UTC game time |
| `gameStatus`, `gameStatusText` | Canonical game status |
| Home and away team fields | Integer IDs and string tricodes |
| Labels plus game ID prefix | Normalized `season_type` |

Known NBA game ID prefixes map to `preseason`, `regular`, `all_star`,
`playoffs`, `play_in`, and `nba_cup_final`. The prefix is authoritative because
NBA Cup quarterfinals and semifinals have `002` regular-season IDs even though
their labels contain the word "final." Text labels are a fallback for an
otherwise unknown source prefix.

## Overtime evidence

The schedule does not provide a dedicated final-period field. It distinguishes
regulation finals from overtime in `gameStatusText`: `Final`, `Final/OT`, and
`Final/OT2` appeared in the 2025-26 response. Discovery maps those values to
four, five, and six periods respectively and accepts the alternate `Final/2OT`
form defensively.

Scheduled and live games retain null period and overtime fields. Exact final
period counts can also be checked against the game boxscore when the season
execution pipeline processes the catalog.

## Failure policy

Discovery does not fall back to a third-party schedule. HTTP failures, non-JSON
responses, nested schedule contract drift, duplicate game IDs, unknown game ID
prefixes, and invalid team identity all fail the command before the catalog is
rewritten.
