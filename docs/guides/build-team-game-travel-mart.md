---
last_updated: "2026-08-30"
---

# Build Team-Game Travel Mart

Build the persisted schedule-travel features from the existing canonical game
catalog:

```bash
uv run nba-build-team-game-travel-mart
```

The command writes `data/analytical/team_game_travel/` atomically and validates
the resulting parquet files and manifest. It does not call an NBA endpoint.

Use a different catalog or output location when testing a revised venue
registry:

```bash
uv run nba-build-team-game-travel-mart \
  --catalog-path data/catalog/games.parquet \
  --output-dir /tmp/team_game_travel
```

See [Team-Game Travel Mart](../data/team-game-travel.md) for the time and venue
contract.
