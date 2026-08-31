---
last_updated: "2026-08-31"
---

# Team Rosters

Preseason active-roster profiles come directly from NBA Stats
`commonteamroster`, one response per active NBA franchise. This endpoint is the
authoritative pre-game source for roster membership and basic player bios; the
league-wide season bio endpoint is empty before regular-season games begin.

## Storage

Each source response remains byte-preserved with request metadata:

```text
data/raw/commonteamroster/<season>/<team_id>.json
data/raw/commonteamroster/<season>/<team_id>.meta.json
data/raw/commonteamroster/<season>/snapshots/<utc_timestamp>/<team_id>.json
data/raw/commonteamroster/<season>/snapshots/<utc_timestamp>/<team_id>.meta.json
```

The normalized all-team table is:

```text
data/curated/team_rosters/<season>/part-00000.parquet
data/curated/team_rosters/<season>/_manifest.json
```

The manifest records the 30 raw input paths, fetch timestamps, row count, and,
on a refresh, the immutable prior snapshot directory. The canonical paths hold
the latest response; `snapshots/` preserves exactly what was replaced.

## Contract

`player_id` and `team_id` are nullable-safe strings so they cannot be coerced
through floating point. The normalized player-team rows include the canonical
team abbreviation, player name, listed position, height in raw and inches
forms, weight, birth date, age, experience, school, acquisition description,
and supplemental status.

The roster is a snapshot, not a season-long transaction ledger. A player may
be absent before signing, waived after the pull, or appear on a later roster.
For the draft cold-start ranking, the table is joined to Draft History by
`player_id`; an unmatched drafted player retains a clearly documented
historical-reference fallback instead of disappearing from the ranking.

## Movement Graph

The local Roster Moves view compares this current roster snapshot with a
persisted final-observed regular-season snapshot from the prior season:

```text
data/curated/team_rosters/2025-26/final_regular_season_snapshot.parquet
```

For every player, the snapshot selects the team in their last available
regular-season box-score roster, including inactive players. It is a much
better end-of-season origin than a minutes-weighted primary team: a player
traded during the prior season is shown as already being on the destination
team, rather than incorrectly displayed as a new offseason move. It remains a
roster snapshot, not a complete transaction ledger. A current player without a
prior NBA roster record is published as an external arrival: the graph draws a
dashed path from beyond the continental map boundary to their current team.
When possible, the path starts at a mapped college campus from the current
roster's `school` field. This remains true for an international player who
played at an NCAA school. Otherwise, the path begins at a country-labelled
exterior anchor from the player catalog's `country` field; only records lacking
both sources use the generic exterior fallback.

The graph's Rookies filter uses the official current-roster experience code
`R`, rather than inferring rookie status from acquisition text.

### Shareable animations

Select a team and use **Copy animation link** to make a URL that restores the
same filtered map and starts the player-path animation on load. The URL contract
is:

```text
#moves?team=MIN&filter=trade&autoplay=1
```

`filter` is optional and accepts `all`, `external`, `rookies`, `trade`,
`signing`, or `waiver`. Animation links are intentionally limited to views with
24 or fewer paths so the player labels remain legible.

Build the prior-season source snapshot after that regular season has been
processed:

```bash
uv run nba-build-final-roster-snapshot 2025-26
```

Refresh the active-roster snapshot after transactions with:

```bash
uv run nba-fetch-team-rosters 2026-27 --refresh
```

The refresh makes one request per franchise with a half-second delay and
archives the previous 30-response snapshot before publishing the new table.

See [Fetch Draft History](../guides/fetch-draft-history.md) for the commands
that fetch a new roster snapshot and publish the drafted-rookie ranking.
