---
last_updated: "2026-08-13"
---

# Train HPM v2.2

HPM v2.2 retains the empirical rebound realization state from v2.1 and swaps
its raw usage, turnover, and top-two usage summaries for a completed-season
conditional-logit terminal-action allocation state. It trains entirely from
cached curated regular-season data and makes no NBA API requests.

```bash
uv run nba-train-hpm-v22 --through-season 2025-26 \
  2>&1 | tee artifacts/logs/train-hpm-v22-2025-26.log
```

Follow the recursive seasonal fit with:

```bash
tail -f artifacts/logs/train-hpm-v22-2025-26.log
```

Each immutable artifact stores both calibration contracts:

- `season_rebound_calibration_metadata.parquet`
- `season_usage_allocation_metadata.parquet`

The fitted allocation model itself is embedded in each season's serialized
context model, so deployment needs no raw event data for arbitrary-lineup
inference.
