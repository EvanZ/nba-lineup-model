# Validate Forward Cold Starts

Validate frozen first-year priors against the completed forward RAPM refit for
the same regular season:

```bash
uv run nba-validate-forward-cold-start --season 2025-26 --render-docs-page
```

The command reads the completed forward exposure-gated RAPM artifact, joins its
frozen `exposure_gated_cold_start` prior rows to post-fit coefficients, and
reconstructs actual player exposure from canonical RAPM stints. It writes:

```text
artifacts/models/forward_cold_start_validation/<season>/<run_id>/
```

The artifact contains a player-level table, cohort metrics, an SVG scatterplot,
and an integrity manifest. The docs page separates drafted, undrafted,
realized-low-exposure, and rotation-exposure cohorts.
