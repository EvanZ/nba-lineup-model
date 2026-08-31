---
last_updated: "2026-08-30"
---

# Team-Game Travel Mart

`data/analytical/team_game_travel/` has one row for each team in each
competitive game from the canonical catalog. It is a schedule data product,
not a player or lineup feature set.

## Contract

For team (t) entering game (g), the mart identifies the venue of the
team's preceding competitive game in the same season and calculates the
great-circle distance to the current listed home venue.

\[
d_{t,g} = \operatorname{haversine}(v_{t,g-1}, v_g).
\]

The interval uses the exact scheduled UTC tipoff timestamps available in the
historical catalog:

\[
\Delta h_{t,g} = \operatorname{tipoff}_{t,g} - \operatorname{tipoff}_{t,g-1}.
\]

The initial modeling-ready signal is nullable for a team's first competitive
game and otherwise is

\[
T_{t,g} = \min(d_{t,g}, 2500)\,\mathbb{1}[0 < \Delta h_{t,g} \le 48].
\]

`travel_within_48_thousand_miles` is (T_{t,g}/1000). The raw, uncapped
distance and hours are retained for diagnostics.

## Fields

`team_game_travel.parquet` includes game/team identifiers, game side, current
and preceding venue tricodes and coordinates, scheduled-tipoff interval,
uncapped and capped miles, and the 48-hour signal. The mart also writes
`season_travel_coverage.parquet` and `_manifest.json` with source-catalog and
artifact checksums.

## Boundaries

- The catalog supplies scheduled tipoffs, not final-buzzer timestamps. The
  interval therefore represents scheduled-tipoff separation.
- The canonical schedule does not consistently identify neutral-site venues.
  Listed home games use the corresponding historical home-venue coordinate.
- Historical relocations use distinct tricodes such as `NJN`, `SEA`, and
  `VAN`. `NOK` remains a documented approximation because its displaced home
  schedule used more than one venue.
- First competitive games of a season preserve travel as missing. A future
  model must choose explicitly whether to impute offseason travel or exclude
  those rows from a travel control.
