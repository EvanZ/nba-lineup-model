# Historical Source Coverage

## Current Strategy

Historical seasons are discovered from the NBA Stats schedule endpoint and
archived directly from NBA Stats V3:

1. `scheduleleaguev2` defines the final game catalog.
2. `playbyplayv3` and `boxscoretraditionalv3` are retained byte-for-byte under
   `data/raw/stats/`.
3. Processing prefers a validated Stats V3 endpoint artifact.
4. A previously retained liveData CDN artifact is used only when its matching
   V3 endpoint is absent.

The old liveData historical pull is not the active acquisition strategy. Its
coverage was incomplete and unreliable across older seasons. The surviving
liveData files are a legacy cache layer, not a source to probe when extending
history. New historical acquisition goes through `nba-fetch-stats-history`.

## V3 Play-By-Play Boundary

Direct NBA Stats probes found populated `playbyplayv3` event streams from
1996-97 onward. Equivalent 1995-96 and earlier requests returned structurally
valid responses with an empty `game.actions` array. Since possession and lineup
reconstruction require events, 1996-97 is the practical lower boundary for the
V3 archive. Earlier `boxscoretraditionalv3` availability does not change that
boundary.

Older schedule responses can include unplayed exhibition and if-necessary
placeholders with non-final status and zero-valued team identities. Discovery
excludes precisely those rows; final games and identified games still undergo
strict schema validation. See
[ADR-0007](../architecture/decisions/0007-stats-v3-play-by-play-history-boundary.md).

## Regular-Season Archive Boundary

The following table describes the source mix for 2019-20 through 2024-25.
`Stats V3 preferred` is the number of game endpoint artifacts selected whenever
both required raw documents validate. `Legacy liveData only` is the small
remainder covered by the earlier cache.

| Season | Games | Stats V3 preferred PBP | Legacy liveData-only PBP | Stats V3 preferred box | Legacy liveData-only box |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019-20 | 1,059 | 264 | 795 | 266 | 793 |
| 2020-21 | 1,080 | 1,078 | 2 | 1,078 | 2 |
| 2021-22 | 1,230 | 1,228 | 2 | 1,228 | 2 |
| 2022-23 | 1,230 | 1,228 | 2 | 1,228 | 2 |
| 2023-24 | 1,230 | 1,228 | 2 | 1,228 | 2 |
| 2024-25 | 1,230 | 1,228 | 2 | 1,228 | 2 |

The 2019-20 split reflects an older partial V3 archive. It is retained as an
explicit source boundary, rather than silently presenting the legacy cache as
a successful live historical feed.

## Source Adaptation

Stats V3 is adapted in memory at the processing boundary. The adapter maps the
V3 event vocabulary, splits combined substitution records, preserves player
identifiers as strings or integers without float coercion, and normalizes
scores and period clocks to the common event schema. Original raw JSON is never
rewritten.

Older liveData substitutions can encode only the outgoing player. When that
legacy fallback is selected, the source adapter expands its paired identifiers
before lineup reconstruction. This is a compatibility path, not the preferred
historical representation.

## Processing And Quality

Source availability is distinct from modeling eligibility. Every selected game
still passes lineup, possession, score-conservation, and period-balance checks.
Games with contradictory or insufficient evidence remain named failures or
warnings in the quality report; they are never silently repaired.

The regular-season historical RAPM panel uses the approved pass/warning subset
recorded in `data/audit/historical_regular/games.parquet`. The current panel
contains 6,011 eligible games across the six historical seasons. It is the
input boundary for forward priors, not a claim that every catalog game is
model-ready.

## Postseason

Historical postseason raw responses follow the same Stats V3-first policy.
The first playoff-prior ablation currently uses the successfully processed
subset of those cached games. Its uneven processing coverage is reported with
the experiment in [Prior-Centered RAPM](../models/prior-rapm.md), rather than
being represented as complete historical playoff coverage.

## Reproduce

Fetch V3 responses for a season type directly from the historical catalog:

```bash
uv run nba-fetch-stats-history \
  --season 2024-25 \
  --season-type regular \
  --endpoint playbyplayv3 \
  --endpoint boxscoretraditionalv3 \
  --max-workers 2
```

The raw cache is the resume boundary. Endpoint-level fetch provenance is stored
in `data/manifests/stats_fetches.parquet`; processing provenance and selected
source hashes are stored with per-game outputs.
