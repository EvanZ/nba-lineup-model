# Raw Responses

The raw cache is the reproducibility boundary between external NBA services and
project-owned transformations.

## Endpoints

The direct client currently supports:

| Endpoint | Cache location |
| --- | --- |
| Play-by-play | `data/raw/playbyplay/{game_id}.json` |
| Boxscore | `data/raw/boxscore/{game_id}.json` |
| Today's scoreboard | `data/raw/scoreboard/todays_scoreboard_00.json` |
| Historical season schedule | `data/raw/scheduleleaguev2/{season}.json` |
| Stats V3 play-by-play | `data/raw/stats/playbyplayv3/{game_id}.json` |
| Stats V3 traditional box score | `data/raw/stats/boxscoretraditionalv3/{game_id}.json` |
| Stats game rotation | `data/raw/stats/gamerotation/{game_id}.json` |

Game IDs must be ten-digit strings. Keeping them as strings preserves leading
zeros.

## Byte preservation

The `.json` file contains the exact response body returned by the NBA source.
The cache does not reserialize successful network responses.

Each response has a `.meta.json` sidecar:

| Field | Meaning |
| --- | --- |
| `endpoint` | Logical NBA source endpoint |
| `game_id` | Ten-digit game ID, when applicable |
| `url` | Requested source URL |
| `fetched_at` | UTC fetch timestamp |
| `sha256` | Digest of the exact response bytes |

On cache read, the digest and path metadata are validated before the payload is
returned.

Schedule sidecars use `season` instead of endpoint and game ID fields. They
retain the same URL, UTC fetch time, and exact-byte digest guarantees.

Stats endpoint sidecars also retain selected response provenance headers,
including `x-datasource` when supplied. The Stats namespace is intentionally
separate because its V3 play-by-play schema is not identical to the liveData
CDN schema. Processing adapts V3 in memory through
`nba_lineup_model.normalize.stats_v3`; it never rewrites the retained raw
response.

## Refresh behavior

Commands use valid cached documents by default. `--refresh` ignores existing
responses and fetches new play-by-play and boxscore documents.

The season fetch flow validates the JSON document, path metadata, and exact-byte
digest before counting a cache hit. If only one game document is valid, it is
retained while the other endpoint is fetched. An invalid cache document is
replaced rather than treated as completed work.

Refreshing changes external state and can expose feed corrections. The sidecar
timestamp and digest make that change observable, but raw files are not retained
as a built-in version history.

The historical Stats flow resumes at endpoint granularity. A cached
`playbyplayv3` response remains complete even if the corresponding box score or
rotation request fails.

## Data policy

Raw responses are ignored by Git. Tests use small synthetic or reduced JSON
fixtures that represent specific feed semantics.
