# Build One Game

## Run

```bash
uv run nba-build-game 0022000180
```

Useful options:

```text
--raw-dir PATH         Raw response cache root
--processed-dir PATH   Processed Parquet root
--refresh              Ignore cached NBA responses
```

## Processing stages

1. Read cached documents, preferring Stats V3 per endpoint and falling back to
   retained liveData. Historical acquisition fetches Stats V3 if neither source
   exists.
2. Normalize the ordered event stream.
3. Reconstruct event lineups and lineup stints.
4. Reconstruct possessions.
5. Split possessions at substitution boundaries.
6. Write six Parquet tables atomically.

The command prints event, stint, possession, segment, and validation-issue
counts, followed by every output path.

## Inspect results

```python
import pandas as pd

game_id = "0022000180"
possessions = pd.read_parquet(
    f"data/processed/possessions/{game_id}.parquet"
)
segments = pd.read_parquet(
    f"data/processed/possession_segments/{game_id}.parquet"
)

assert possessions["points_home"].sum() == segments["points_home"].sum()
assert possessions["points_away"].sum() == segments["points_away"].sum()
```

## Refresh carefully

Use `--refresh` when testing source corrections or cache behavior:

```bash
uv run nba-build-game 0022000180 --refresh
```

Refreshing overwrites the local raw response and provenance sidecar.
