---
last_updated: "2026-08-05"
---

# Exposure-Gated O/D Cold Starts

This model applies the [cold-start exposure gate](cold-start-exposure.md) to
the frozen [offense/defense RAPM](offense-defense-rapm.md) state. It is the
first cold-start model that preserves the O/D identification convention rather
than assigning a one-number prior to an O/D design.

Returning players retain their completed 2024-25 O/D RAPM values. For each
first-NBA-season player \(i\), the common probability of entering the
low-exposure pool mixes separately estimated offense and defense priors:

\[
\widehat O_i^{cold} = p_i^{low} O^{replacement} + (1-p_i^{low})\widehat O_i^{draft},
\]

\[
\widehat D_i^{cold} = p_i^{low} D^{replacement} + (1-p_i^{low})\widehat D_i^{draft}.
\]

The frozen O/D possession scorer then uses the home offense and away defense
with opposite signs:

\[
\widehat y_i = \overline y +
\frac{\sum_{p \in O_i}\widehat O_p - \sum_{p \in D_i}\widehat D_p}{100}
+ s_i\frac{h}{100}.
\]

Thus, the model never splits a one-number replacement estimate in half. It
fits and retains a distinct value for scoring and for points prevented.

## Frozen 2025-26 Inputs

The immutable run is
`artifacts/models/exposure_gated_offense_defense/2025-26/exposure-gated-od-2025-26-20260806T023559Z-4ffeb2bc/`.
Every fitted component ends with the 2024-25 regular season.

| Component | Training window | Result |
| --- | --- | --- |
| Returning O/D state | 1996-97 through 2024-25 forward O/D RAPM | 462 returning-player priors |
| Draft offense rate | First-year O/D estimates through 2024-25 | Ridge penalty 10.0 |
| Draft defense rate | First-year O/D estimates through 2024-25 | Ridge penalty 1.0 |
| Exposure gate | First-year exposure outcomes through 2024-25 | Player-specific \(p_i^{low}\) |
| O/D replacement token | 29 completed regular seasons, <5% team opportunity share | offense -2.702; defense -1.175 |

The token values are equal-season means of independently tokenized O/D RAPM
fits. Their net sum is -3.877, but only the separate side values enter the
lineup scorer.

The 100 first-year 2025-26 profiles are generated before target outcomes are
read. Evaluation uses realized target lineups and exposures only, with no
2025-26 score, possession outcome, target player refit, or playoff outcome in
the prior.

## Result

The added O/D cold starts reduce regular-season unknown player exposures by
90.4%, from 254,212 in the frozen O/D baseline to 24,356. On the same 1,230
regular-season games, they also improve O/D RAPM's possession RMSE and
game-margin RMSE. The one-number exposure-gated prior remains better on
game-margin RMSE, and this O/D version is slightly worse than plain frozen O/D
RAPM on the 85-game playoff holdout.

| Model | Cohort | Possession RMSE | Possession MAE | Game-margin RMSE | Unknown player exposures |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen O/D RAPM | Regular season | 1.198853 | 1.142664 | 14.8901 | 254,212 |
| Exposure-gated O/D cold starts | Regular season | **1.198792** | **1.142422** | **14.7631** | **24,356** |
| Frozen O/D RAPM | Playoffs | **1.194642** | **1.139203** | **17.8037** | 5,678 |
| Exposure-gated O/D cold starts | Playoffs | 1.194678 | 1.139209 | 17.8562 | **93** |

See the [Frozen Preseason Leaderboard](preseason-leaderboard.md) for the
comparison against all evaluated preseason models, including team NetRtg and
Pythagorean wins.

## Artifacts

| File | Contents |
| --- | --- |
| `rookie_od_draft_rates.parquet` | Outcome-free target draft offense and defense rates |
| `revised_rookie_od_rankings.parquet` | 100 blended first-year O/D priors and net ranks |
| `season_od_replacement_tokens.parquet` | Per-season separately estimated O/D replacement tokens |
| `od_rate_cross_validation.parquet` | Six forward validation seasons for each O/D ridge penalty |
| `frozen_player_priors.parquet` | Final 2025-26 O/D prior vector used for scoring |
| Evaluation parquet files | Possession, game, team NetRtg, and Pythagorean win outputs |
| `metadata.json` / `source_state.json` / `manifest.json` | Temporal boundary, input identities, and integrity hashes |
