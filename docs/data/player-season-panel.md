# Player-Season Panel

The player-season panel is the shared historical feature boundary for RAPM
priors, box-score plus-minus, aging models, and neural player tokens.

It intentionally publishes two different tables:

| Table | Purpose | Temporal meaning |
| --- | --- | --- |
| `player_seasons.parquet` | Research outcomes and same-season summaries | Full observed season |
| `transitions.parquet` | Predictive features for a target season | Prior-season performance only |

Keeping these contracts separate prevents a full target season's box score or
RAPM result from leaking into predictions for that season.

## Build

Every requested regular season must already have validated curated player data,
player bios, and a completed RAPM run:

```bash
uv run nba-build-player-season-panel \
  2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 2025-26
```

The build resolves each season's latest validated RAPM run unless an immutable
run is pinned:

```bash
uv run nba-build-player-season-panel \
  2024-25 2025-26 \
  --rapm-run-id 2024-25=rapm-2024-25-example
```

## Same-season table

One row represents one canonical NBA player in the RAPM outcome universe.
Every canonical RAPM player must have a season bio; bio-only players are not
outcomes and are excluded. The RAPM outcome universe controls panel membership.
Historical partial-season curation can leave positive-minute box-score rows for
players whose only games were excluded from RAPM; those box-score-only players
are dropped. Conversely, RAPM players with no played box-score summary remain
as outcomes with null box-score features. A few pre-modern Stats feeds expose
numeric roster placeholders that are absent from the NBA player catalog; these
unresolvable rows are excluded from the player-level panel because they cannot
support a longitudinal player estimate.

The table includes:

- RAPM, raw on-court net rating, exposure, possessions, seconds, and stints;
- games, starts, minutes, and counting totals from positive-minute game box
  scores, plus `boxscore_features_available`;
- selected per-36 rates and shooting percentages;
- season-specific age, position, size, country, college, and draft fields;
- career start year, derived experience, rookie status, and years since draft;
- primary team by box-score minutes and RAPM exposure;
- the exact source RAPM run ID.

This table contains outcomes. It is appropriate for fitting and evaluating
historical component models, not as a direct input for possessions from the
same season.

NBA rows marked as played with zero recorded minutes are retained in the
curated source table but excluded from season games and totals. When a RAPM/bio
player has no played box-score summary, its box-score features remain null and
`boxscore_features_available=false`; they are never imputed as zero during
panel construction.

Pre-modern NBA Stats player boxes omit the `played` flag, full name, two-point
totals, offensive fouls, and fouls drawn. Some Stats V3 boxes retain `played`
but leave it null. The panel derives participation from positive recorded
minutes only when the field is absent or null, preserves explicit DNP flags,
derives names and two-point totals from retained fields, and records unavailable
foul counts as structural zero. This keeps the historical box-score feature
schema stable without inventing player or possession outcomes.

## Transition table

For target season \(t\), each row contains:

\[
\left(
  \text{target player context}_{i,t},
  \text{prior performance}_{i,t-1},
  \text{target RAPM label}_{i,t}
\right)
\]

All lagged performance fields use a `prior_` prefix. They include prior RAPM,
exposure, games, minutes, per-36 box-score rates, and shooting percentages.
Target RAPM and exposure are retained as supervised labels and diagnostics.
Current-season box-score summaries are absent.

Players without a row in \(t-1\) remain in the target season with
`has_prior_season=false` and null prior features. This explicit cold-start
contract supports rookies, returning players, and players entering the NBA
data universe.

## Transformer use

The transition table can be joined to a target-season possession by
`(target_season, player_id)`. A future Transformer can project the numeric and
categorical context into a side-information embedding and combine it with the
learned player ID embedding.

The same lookup must be fitted or normalized using training seasons only.
Target RAPM columns are labels and must never enter token construction. Missing
prior features require an explicit learned cold-start representation or
training-fold imputation, not values calculated from the target season.

## Storage and integrity

```text
data/analytical/player_season_panel/
  _manifest.json
  player_seasons.parquet
  transitions.parquet
```

The manifest pins ordered seasons, source RAPM runs, hashes of RAPM, curated
players, and bio manifests, output hashes and row counts, the builder code
fingerprint, and the exact prior-feature column list. Publication is atomic and
the completed directory is validated against the manifest before return.
