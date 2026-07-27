# Run Cross-Season Audits

## Run the committed matrix

```bash
uv run nba-audit-games config/audit_manifest.json
```

The command returns a nonzero status when any game fails or errors.

Treat warnings as failures in stricter automation:

```bash
uv run nba-audit-games \
  config/audit_manifest.json \
  --fail-on-warnings
```

## Read the report

```python
import pandas as pd

games = pd.read_parquet("data/audit/games.parquet")
summary = pd.read_parquet("data/audit/summary.parquet")

print(games[["game_id", "season", "status", "issue_codes"]])
print(summary)
```

## Create a larger manifest

Provide a Parquet or CSV catalog with:

- `game_id`
- `season`
- `season_type`
- optional `sample_group`
- optional `expected_overtime`

Then sample deterministically:

```bash
uv run nba-sample-audit data/external/game_catalog.parquet \
  --games-per-stratum 25 \
  --seed 7 \
  --output config/audit_manifest_sample.json
```

Sampling groups by season, season type, and sample group. Duplicate game IDs or
null stratum values are rejected.

## Add a regression case

When an audit exposes a new feed pattern:

1. Inspect the raw actions surrounding the issue.
2. Decide whether it is a valid basketball rule, a feed-version difference, or
   corrupt source data.
3. Add a small focused fixture or synthetic test.
4. Change the parser only as broadly as the evidence supports.
5. Rerun the complete manifest.

Do not remove a warning by weakening an exact invariant.
