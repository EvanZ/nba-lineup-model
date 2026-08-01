# Historical Source Coverage

Historical game completeness is measured per endpoint, not per game directory.
A game is covered when either:

1. its primary liveData CDN artifact passes JSON, game-ID, schema, metadata, and
   SHA-256 validation; or
2. the corresponding NBA Stats fallback artifact passes the same class of
   endpoint-specific checks.

The source priority is intentional. Canonical processing prefers Stats V3 for
each endpoint when it is retained and falls back to liveData otherwise. Older
liveData play-by-play snapshots can encode only the outgoing side of a
substitution; the Stats V3 adapter reconstructs both directions without
altering retained source bytes. A liveData fallback expands that legacy
two-player identifier field in memory before lineup reconstruction.

## Completed coverage

The completed acquisition covers every final regular-season game from 2019-20
through 2024-25:

| Season | Scope | Endpoint | CDN primary | Stats fallback | Unresolved | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2019-20 | Regular season | Play-by-play | 795 | 264 | **0** | 1,059 |
| 2019-20 | Regular season | Box score | 793 | 266 | **0** | 1,059 |
| 2020-21 | Regular season | Play-by-play | 2 | 1,078 | **0** | 1,080 |
| 2020-21 | Regular season | Box score | 2 | 1,078 | **0** | 1,080 |
| 2021-22 | Regular season | Play-by-play | 2 | 1,228 | **0** | 1,230 |
| 2021-22 | Regular season | Box score | 2 | 1,228 | **0** | 1,230 |
| 2022-23 | Regular season | Play-by-play | 2 | 1,228 | **0** | 1,230 |
| 2022-23 | Regular season | Box score | 2 | 1,228 | **0** | 1,230 |
| 2023-24 | Regular season | Play-by-play | 2 | 1,228 | **0** | 1,230 |
| 2023-24 | Regular season | Box score | 2 | 1,228 | **0** | 1,230 |
| 2024-25 | Regular season | Play-by-play | 2 | 1,228 | **0** | 1,230 |
| 2024-25 | Regular season | Box score | 2 | 1,228 | **0** | 1,230 |

The 2019-20 Stats batch retained 530 endpoint artifacts across 266 games. Two
artifacts were validated smoke-test cache hits; 528 were new downloads. Full
read-back validation found:

- 264 valid `playbyplayv3` responses;
- 266 valid `boxscoretraditionalv3` responses;
- 62,636,863 total response bytes;
- `x-datasource: S3` on every retained response;
- zero duplicate `(run_id, game_id, endpoint)` manifest rows.

Full cross-season read-back reconciliation found:

- 7,059 covered regular-season games;
- 6,254 Stats V3 play-by-play fallback artifacts;
- 6,256 Stats V3 box-score fallback artifacts;
- 1,473,538,333 exact response bytes across both fallback endpoints;
- `x-datasource: S3` on every fallback response;
- zero unresolved endpoint artifacts;
- 12,510 successful manifest rows, two smoke-test skips, and zero failures;
- zero duplicate `(run_id, game_id, endpoint)` manifest rows.

## Postseason coverage

Play-in and playoff games use the same endpoint-level fallback policy:

| Season | Games | Play-by-play CDN | Play-by-play fallback | Box-score CDN | Box-score fallback | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019-20 | 84 | 1 | 83 | 1 | 83 | **0** |
| 2020-21 | 91 | 1 | 90 | 1 | 90 | **0** |
| 2021-22 | 93 | 1 | 92 | 1 | 92 | **0** |
| 2022-23 | 90 | 1 | 89 | 1 | 89 | **0** |
| 2023-24 | 88 | 1 | 87 | 1 | 87 | **0** |
| 2024-25 | 90 | 1 | 89 | 1 | 89 | **0** |

The postseason acquisition added 1,060 successful S3 fallback artifacts and
123,962,043 exact response bytes. Combined regular-season and postseason
coverage now represents:

- 7,595 games;
- 13,570 Stats fallback artifacts;
- 1,597,500,376 exact fallback response bytes;
- zero unresolved play-by-play or box-score endpoints.

## Complete final-game catalog

The remaining preseason, All-Star, and NBA Cup final games were acquired after
the modeling-critical regular-season and postseason layers. Final coverage by
season type is:

| Season type | Games | Play-by-play CDN | Play-by-play fallback | Box-score CDN | Box-score fallback | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regular season | 7,059 | 805 | 6,254 | 803 | 6,256 | **0** |
| Playoffs | 505 | 6 | 499 | 6 | 499 | **0** |
| Play-in | 31 | 0 | 31 | 0 | 31 | **0** |
| Preseason | 331 | 0 | 331 | 0 | 331 | **0** |
| All-Star | 21 | 0 | 21 | 0 | 21 | **0** |
| NBA Cup final | 2 | 0 | 2 | 0 | 2 | **0** |

Across every final catalog game from 2019-20 through 2024-25, full read-back
reconciliation found:

- 7,949 games and 15,898 endpoint slots;
- 14,278 successful Stats fallback artifacts;
- 1,683,923,204 exact fallback response bytes;
- `x-datasource: S3` on every fallback response;
- zero unresolved endpoint slots;
- 14,278 successful manifest rows, two smoke-test skips, and zero failures;
- zero duplicate `(run_id, game_id, endpoint)` manifest rows.

## 2025-26 paired source corpus

Stats V3 was also acquired for every final 2025-26 catalog game, including
games that already have liveData. This paired corpus supports source-to-source
adapter validation and provides the preferred production source.

| Season type | Games | Play-by-play V3 | Box-score V3 | Missing |
| --- | ---: | ---: | ---: | ---: |
| Regular season | 1,230 | 1,230 | 1,230 | **0** |
| Playoffs | 85 | 85 | 85 | **0** |
| Play-in | 6 | 6 | 6 | **0** |
| Preseason | 71 | 71 | 71 | **0** |
| All-Star | 7 | 7 | 7 | **0** |
| NBA Cup final | 1 | 1 | 1 | **0** |
| **Total** | **1,400** | **1,400** | **1,400** | **0** |

Read-back reconciliation found 2,800 successful endpoint responses,
340,412,444 exact response bytes, `x-datasource: S3` on every response, and
unique `(game_id, endpoint)` keys. Across 2019-20 through 2025-26, the retained
Stats V3 archive now contains 17,078 endpoint artifacts and 2,024,335,648
exact response bytes.

## Game rotation probe

`gamerotation` is not included in the completeness totals. A representative
one-game-per-season probe produced:

| Season | Result |
| --- | --- |
| 2019-20 | HTTP 500 |
| 2020-21 | HTTP 500 |
| 2021-22 | HTTP 500 |
| 2022-23 | Succeeded in 24.59 seconds |
| 2023-24 | HTTP 500 |
| 2024-25 | Succeeded in 21.60 seconds |

The two successful responses are cached. A full rotation acquisition is not
justified against a 2-of-6 probe success rate, 20-30 second response latency,
and an independently backed source. Rotation can be revisited if a more
reliable archive endpoint is identified.

## What complete means

`Unresolved = 0` establishes source availability, not a passing reconstruction.
The Stats V3 adapter is covered by synthetic contract tests and cross-season
samples. Full-season processing still applies the same strict lineup,
possession, score, and period-balance quality gates used for liveData; games
that expose insufficient or contradictory V3 evidence remain failed builds
rather than being silently accepted.

### Adapter validation sample

A full reconstruction scan covered all 7,138 retained V3 play-by-play and
box-score pairs across every 2019-20 through 2024-25 season type. It
reconstructed 7,060 games and raised explicit exceptions for 78, a 98.9%
reconstruction rate. Regular-season compatibility was 6,224 of 6,254 games,
or 99.5%.

| Season type | Reconstruction exceptions |
| --- | ---: |
| Regular season | 30 |
| Preseason | 36 |
| All-Star | 6 |
| Play-in | 3 |
| Playoffs | 3 |
| **Total** | **78** |

The exceptions comprise 44 ambiguous or missing historical player names, 15
contradictory period-opening lineup evidence cases, seven contradictory
substitution states, six malformed exhibition-game minute strings, three
malformed substitution descriptions, and three duplicate substitution
batches. These remain explicit failures pending evidence-based repair rules.

Reconstruction compatibility is weaker than production acceptance. To measure
the latter, a deterministic evenly spaced sample of 200 regular-season Stats
V3 pairs was drawn from the 6,254-game fallback population spanning 2019-20
through 2024-25. Every sampled pair completed adaptation and reconstruction:

| Season | Pass | Warning | Fail | Adapter error |
| --- | ---: | ---: | ---: | ---: |
| 2019-20 | 3 | 4 | 2 | 0 |
| 2020-21 | 8 | 26 | 0 | 0 |
| 2021-22 | 3 | 33 | 3 | 0 |
| 2022-23 | 9 | 26 | 4 | 0 |
| 2023-24 | 8 | 23 | 8 | 0 |
| 2024-25 | 10 | 29 | 1 | 0 |
| **Total** | **41** | **141** | **18** | **0** |

Thus 182 of 200 games, or 91%, passed the production quality gate or completed
with named warnings. The 18 failures were retained as failures. Their dominant
hard conditions were non-monotonic source score corrections and unbalanced
within-period possession counts. This sample validates the compatibility
boundary; it does not replace a full-catalog processing report.

### 2025-26 paired validation

The current-season paired corpus permits a stronger controlled comparison.
For all 1,230 regular-season games, the audit used the same liveData box score
and changed only the play-by-play input between liveData and adapted Stats V3.

| V3 audit result | Games |
| --- | ---: |
| Pass | 215 |
| Warning | 817 |
| Fail | 187 |
| Reconstruction exception | 11 |
| **Total** | **1,230** |

Of the 1,219 reconstructed games:

- 1,217 matched the final liveData box-score score;
- 498 had exactly the same possession count as liveData;
- 1,042, or 85.5%, were within one possession of liveData;
- the signed V3-minus-liveData possession difference averaged `+0.700`, with
  mean absolute difference `0.770`.

The 11 explicit exceptions comprise five ambiguous one-word substitution
names, five over-complete period-opening lineup evidence cases, and one event
outside an active period. The frequent warning and failure statuses primarily
reflect possession-boundary differences, not missing source responses. These
results support retaining both sources: liveData remains primary for 2025-26,
while Stats V3 provides a complete paired validation corpus and historical
fallback.

Run the focused adapter and source-selection contract tests with:

```bash
uv run pytest tests/test_stats_v3_adapter.py
```

## Reproduce the gap acquisition

Use the endpoint-level Prefect flow:

```bash
uv run nba-fetch-stats-history \
  --season 2019-20 \
  --cdn-missing-only \
  --run-id stats-gaps-2019-20
```

Raw artifacts are the resume boundary. The portable execution history is
`data/manifests/stats_fetches.parquet`.
