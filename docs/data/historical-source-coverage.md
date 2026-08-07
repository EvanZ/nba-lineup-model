---
last_updated: "2026-08-02"
---

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

## Regular-Season Source Coverage

This table is generated from the final-game catalog, byte-preserved raw cache,
and player-season bio partitions. Every listed season uses Stats V3 for its
play-by-play and box-score archive except `2019-20*`. Source completeness is
not processing eligibility: only games passing the downstream quality contract
are model-ready.

| Season | Catalog regular games | PBP source | Box source | Player-season bios |
| --- | ---: | ---: | ---: | ---: |
| 1996-97 | 1,189 | 1,189 V3 | 1,189 V3 | 441 |
| 1997-98 | 1,189 | 1,189 V3 | 1,189 V3 | 439 |
| 1998-99 | 725 | 725 V3 | 725 V3 | 440 |
| 1999-00 | 1,189 | 1,189 V3 | 1,189 V3 | 439 |
| 2000-01 | 1,189 | 1,189 V3 | 1,189 V3 | 441 |
| 2001-02 | 1,189 | 1,189 V3 | 1,189 V3 | 440 |
| 2002-03 | 1,189 | 1,189 V3 | 1,189 V3 | 428 |
| 2003-04 | 1,189 | 1,189 V3 | 1,189 V3 | 442 |
| 2004-05 | 1,230 | 1,230 V3 | 1,230 V3 | 464 |
| 2005-06 | 1,230 | 1,230 V3 | 1,230 V3 | 458 |
| 2006-07 | 1,230 | 1,230 V3 | 1,230 V3 | 458 |
| 2007-08 | 1,230 | 1,230 V3 | 1,230 V3 | 451 |
| 2008-09 | 1,230 | 1,230 V3 | 1,230 V3 | 445 |
| 2009-10 | 1,230 | 1,230 V3 | 1,230 V3 | 442 |
| 2010-11 | 1,230 | 1,230 V3 | 1,230 V3 | 452 |
| 2011-12 | 990 | 990 V3 | 990 V3 | 478 |
| 2012-13 | 1,229 | 1,229 V3 | 1,229 V3 | 469 |
| 2013-14 | 1,230 | 1,230 V3 | 1,230 V3 | 482 |
| 2014-15 | 1,230 | 1,230 V3 | 1,230 V3 | 492 |
| 2015-16 | 1,230 | 1,230 V3 | 1,230 V3 | 476 |
| 2016-17 | 1,230 | 1,230 V3 | 1,230 V3 | 486 |
| 2017-18 | 1,230 | 1,230 V3 | 1,230 V3 | 540 |
| 2018-19 | 1,230 | 1,230 V3 | 1,230 V3 | 530 |
| 2019-20* | 1,059 | 264 V3 + 795 CDN | 266 V3 + 793 CDN | 529 |
| 2020-21 | 1,080 | 1,078 V3 + 2 CDN | 1,078 V3 + 2 CDN | 540 |
| 2021-22 | 1,230 | 1,228 V3 + 2 CDN | 1,228 V3 + 2 CDN | 605 |
| 2022-23 | 1,230 | 1,228 V3 + 2 CDN | 1,228 V3 + 2 CDN | 539 |
| 2023-24 | 1,230 | 1,228 V3 + 2 CDN | 1,228 V3 + 2 CDN | 572 |
| 2024-25 | 1,230 | 1,228 V3 + 2 CDN | 1,228 V3 + 2 CDN | 569 |
| 2025-26 | 1,230 | 1,230 V3 | 1,230 V3 | 582 |

\* The earlier 2019-20 V3 cache is partial, so 795 PBP and 793 box-score
responses use the retained liveData CDN fallback. The two CDN fallbacks in
2020-21 through 2024-25 are isolated cache gaps, not a second acquisition
strategy.

## Source Adaptation

Stats V3 is adapted in memory at the processing boundary. The adapter maps the
V3 event vocabulary, splits combined substitution records, preserves player
identifiers as strings or integers without float coercion, and normalizes
scores and period clocks to the common event schema. Original raw JSON is never
rewritten.

The 1996-era archive also uses a few deterministic legacy encodings. When a
box score has exactly five non-empty position values, those players are its
starters; when it lists positions for reserves too, the traditional first-five
box-score ordering defines the starters. Blank score fields and a regressive
`0-0` score on non-scoring actions carry forward the preceding cumulative
score. If an otherwise valid period has no start action, the adapter inserts a
derived period-start boundary. These are representation reconciliations, not
changes to the preserved source documents.

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
recorded in `data/audit/historical_regular/games.parquet`. This processing
eligibility is intentionally separate from the raw-coverage table above: the
append-only build ledger and latest `data/quality/games.parquet` rows are the
authority while the 1996-97 through 2018-19 reconstruction run is in progress.
Only the approved subset is a model input; raw completeness is never presented
as a claim that every catalog game is model-ready.

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
