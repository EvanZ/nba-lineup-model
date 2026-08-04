# Compare RAPM Across Eras

This report standardizes each completed season's full-season forward-prior RAPM
against its own exposure-weighted league distribution. It then converts the
standardized rate to wins above an average player at a fixed 2,000-minute role.
It is a retrospective player-season comparison, not a preseason forecast.

```bash
uv run nba-build-era-comparison \
  --forward-lagged-run-dir artifacts/models/prior_rapm/2025-26/forward-lagged-rapm-2025-26-20260803T203054Z-c627d89d \
  --forward-calibration-run-dir artifacts/models/forward_calibration/2025-26/forward-calibration-2025-26-20260804T132355Z-7ec43aa9
```

Outputs are written under `artifacts/reports/era_comparison/2025-26/`:

- `player_season_comparisons.parquet` contains every player-season;
- `qualified_peak_seasons.parquet` applies the default 2,000-minute threshold;
- `top_25_qualified_peak_seasons.parquet` is the published peak-season table.

The fixed-role rate deliberately separates quality from availability.
`wins_above_average_actual_minutes` is included as a descriptive total, but it
depends on the player's realized usage and should not be treated as a pure
talent estimate.
