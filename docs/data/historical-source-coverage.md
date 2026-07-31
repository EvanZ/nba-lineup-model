# Historical Source Coverage

Historical game completeness is measured per endpoint, not per game directory.
A game is covered when either:

1. its primary liveData CDN artifact passes JSON, game-ID, schema, metadata, and
   SHA-256 validation; or
2. the corresponding NBA Stats fallback artifact passes the same class of
   endpoint-specific checks.

The source priority is intentional. Existing canonical processing consumes the
liveData schema. Stats V3 responses are preserved as fallback evidence but need
a source adapter before processing.

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

`Unresolved = 0` establishes source availability, not processing
compatibility. The Stats V3 play-by-play action schema differs from liveData,
including its substitution representation. A season is ready for canonical
processing only after every fallback artifact can be normalized through a
tested adapter and passes the existing reconstruction quality gates.

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
