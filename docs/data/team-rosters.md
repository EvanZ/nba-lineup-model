---
last_updated: "2026-08-05"
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
```

The normalized all-team table is:

```text
data/curated/team_rosters/<season>/part-00000.parquet
data/curated/team_rosters/<season>/_manifest.json
```

The manifest records the 30 raw input paths, fetch timestamps, and row count.

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

See [Fetch Draft History](../guides/fetch-draft-history.md) for the commands
that fetch a new roster snapshot and publish the drafted-rookie ranking.
