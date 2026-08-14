---
last_updated: "2026-08-14"
---

# Profile Token Mart

The profile-token mart is the forward-safe input contract for set-based neural
lineup models. It has one row per player in a target-season forward RAPM state.
It is not a lineup dataset and does not assign a new player value by itself.

## Time Boundary

For target season \(t\), each row records `source_season = t-1`.

| Input group | Information used |
| --- | --- |
| `prior_rapm` | The no-context forward RAPM state formed before \(t\). |
| Prior exposure and returner flag | Player-season panel in \(t-1\). |
| Contextual box-score profile | Smoothed possession rates from \(t-1\), or a forward cold-start profile. |
| Shot taxonomy profile | Attempt rates and finishing from \(t-1\), or the corresponding forward cold-start profile. |
| Biography | Target-season known attributes: age, experience, draft status, height, and weight. |

The target player universe is an oracle roster universe for frozen evaluation:
it may include a player who appears in target-season lineups, but no target
season outcome, box score, shot event, possession, or RAPM estimate is read to
construct that player's token.

The current materialization begins in 1997-98 because 1996-97 has no prior
player season from which to form a token. It runs through 2025-26, the final
season in the current player panel. A future 2026-27 materialization requires a
known roster/bio universe first.

## Token Features

`player_profile_tokens.parquet` retains identifiers and provenance for joins,
but neural model input may use only the documented feature columns:

| Group | Columns |
| --- | --- |
| Player state | `prior_rapm` |
| Availability/provenance | `has_prior_player_season`, `prior_on_court_possessions`, `profile_imputed`, `profile_replacement_weight`, `shot_profile_imputed`, `shot_profile_replacement_weight` |
| Biography | `age`, `nba_experience_years`, `is_rookie`, `height_inches`, `weight_pounds`, `draft_number`, `has_draft_number`, `is_undrafted` |
| Missing-biography flags | `age_imputed`, `height_imputed`, `weight_imputed` |
| Lagged box-score profile | Three-point volume/makes, assists, turnovers, usage, offensive/defensive rebounding, steals, and blocks, all possession-native where applicable |
| Lagged shot profile | Rim, non-rim-two, and three attempt rates plus source-season league-shrunk finishing |
| Hierarchical shooting | Family-specific career attempts and career/league-anchored finishing percentages |
| Free throws and foul drawing | Free-throw attempts per 100, FTA/FGA, hierarchical FT%, and career FT attempts |

`has_prior_player_season` is `1` when the player has a panel row in \(t-1\),
not when their RAPM prior happens to be nonzero. `profile_imputed` is `1` when
that player's contextual box-score profile is synthesized rather than observed.
The analogous `shot_profile_imputed` flag covers the shot profile.

No player ID, player name, team, position, lineup slot, target-season outcome,
or target-season profile statistic is an allowed neural input. Models must fit
scalers from their own training window; this mart intentionally stores raw
values to avoid baking evaluation-period distributional information into a
feature transform.

## Example: Kobe Bryant

This is Kobe Bryant's actual token for target season 2006-07. It was formed
before that season, using his 2005-06 profile and forward player prior. The
identifier and name are retained for joins and inspection, but are not neural
inputs.

| Provenance | Value |
| --- | --- |
| Target / source season | 2006-07 / 2005-06 |
| Player | 977, Kobe Bryant |
| Contextual profile source | prior_season |
| Shot profile source | prior_season |
| Foul profile source | prior_season |

~~~text
prior_rapm                           5.447572
has_prior_player_season              1
prior_on_court_possessions        6291.041667
profile_imputed                       0
profile_replacement_weight            0.000000
shot_profile_imputed                  0
shot_profile_replacement_weight       0.000000
age_imputed / height_imputed /
weight_imputed                        0 / 0 / 0

age / nba_experience_years           28.0 / 10.0
is_rookie                             0
height_inches / weight_pounds        78.0 / 220.0
draft_number / has_draft_number      13.0 / 1
is_undrafted                          0

three_pa_per_100 / three_pm_per_100   8.004306 / 2.782114
assists_per_100 / turnovers_per_100   5.675282 / 3.934957
usage_per_100                         43.273700
offensive_rebounds_per_100 /
defensive_rebounds_per_100            1.197928 / 5.664899
steals_per_100 / blocks_per_100       2.308186 / 0.504819
offensive_rebound_pct /
defensive_rebound_pct                 2.613851 / 12.732509

rim_attempts_per_100 / rim_fg_pct_shrunk
                                       6.996848 / 0.600001
non_rim_two_attempts_per_100 /
non_rim_two_fg_pct_shrunk             18.738191 / 0.436542
three_attempts_per_100 /
three_fg_pct_shrunk                   8.018000 / 0.348863

rim_fg_pct_hierarchical /
rim_career_attempts                    0.604516 / 2857.0
non_rim_two_fg_pct_hierarchical /
non_rim_two_career_attempts            0.436607 / 5882.0
three_fg_pct_hierarchical /
three_career_attempts                 0.345892 / 1821.0

free_throw_attempts_per_100           12.687238
free_throw_rate                        0.375442
free_throw_pct_hierarchical            0.848084
free_throw_career_attempts             4263.0
foul_profile_imputed                   0
~~~

For example, prior_rapm = 5.447572 is the no-context forward RAPM estimate
available at the start of 2006-07. It is distinct from Kobe's 2006-07 fitted
coefficient, which is target-season information and therefore excluded from
this token. The rate fields are standardized during a neural run using its
training data only, not when this mart is built.

## Shot and Foul Revision

The mart's second contract revision adds hierarchical finishing for rim,
non-rim-two, and three-point attempts, as well as a free-throw/foul-drawing
profile. It contains 47 model inputs. The neural input now contains:

| Group | Added fields |
| --- | --- |
| Hierarchical shooting | Family-specific career attempts and career/league-anchored finishing percentages |
| Foul drawing | Free-throw attempts per 100 and stabilized FTA/FGA |
| Free throws | Hierarchical FT% and career free-throw attempts |
| Provenance | Foul-profile imputation and replacement-blend fields |

For target season \(t\), let \(s=t-1\). For each shot family \(k\), the
player-specific career prior excludes the source season:

\[
p^{\mathrm{career}}_{i,k,s} =
\frac{M_{i,k,<s} + 75\ell_{k,s}}
{A_{i,k,<s} + 75}.
\]

The source season then updates that prior once:

\[
\widetilde p_{i,k,t} =
\frac{M_{i,k,s} + 75p^{\mathrm{career}}_{i,k,s}}
{A_{i,k,s} + 75}.
\]

\(\ell_{k,s}\) is the source-season league percentage for the same family.
Thus, a player with a credible earlier career shrinks toward their own history;
a player with little history shrinks toward the era-appropriate league rate.
The fixed 75-attempt values are documented initial defaults and will be tuned
by expanding one-season-ahead validation before neural model selection.

Free-throw attempt volume uses the mart's 300-possession rate stabilization:

\[
\widetilde{\mathrm{FTA}}_{i}/100 =
100\frac{\mathrm{FTA}_{i,s} + 300\overline{\mathrm{FTA}}_s/100}
{n_{i,s}+300}.
\]

Free-throw rate is stabilized FTA/FGA:

\[
\widetilde{\mathrm{FTr}}_i =
\frac{\mathrm{FTA}_{i,s} + 75\overline{\mathrm{FTr}}_s}
{\mathrm{FGA}_{i,s} + 75}.
\]

Hierarchical FT% follows the same two-stage construction as shot-family
finishing. Every player and league statistic stops at \(s\); no target-season
shooting outcome is used.

## Cold Starts

For a player without a prior profile, the existing exposure-gated replacement
probability \(w_i\) is reused:

\[
x_i^{\mathrm{cold}} =
(1-w_i)x_{\mathrm{rookie\ cohort}} +
w_i x_{\mathrm{replacement}}.
\]

The box-score profile uses its established draft-cohort and low-exposure
replacement population. The shot and foul profiles apply the same blend to
possession-weighted historical rookie profiles and the historical low-exposure
replacement profile. Both populations are restricted to seasons before \(t\).

## Storage

~~~text
data/analytical/profile_tokens/
  _manifest.json
  player_profile_tokens.parquet
  season_profile_token_coverage.parquet
~~~

The atomic manifest hashes the player panel, shot taxonomy, source no-context
forward-RAPM run, and both output tables. It also records the one-season source
mapping for every target season.

## Current Historical Build

The current mart covers 29 target seasons, 1997-98 through 2025-26, and
contains **13,389** player-token rows. The coverage table makes all synthesized
inputs explicit:

| Check | Count |
| --- | ---: |
| Token rows | 13,389 |
| Cold-start contextual profiles | 1,877 |
| Cold-start shot profiles | 1,877 |
| Cold-start foul profiles | 1,877 |
| Age values filled from prior-history medians | 257 |
| Height values filled from prior-history medians | 313 |
| Weight values filled from prior-history medians | 321 |

The bulk of the biography fills occur in the earliest historical seasons, where
the recovered lineup/RAPM universe is more complete than the corresponding
box-score biography archive. Those players remain visible through their
imputation indicators; they are never silently discarded.

## Build

~~~bash
uv run nba-build-profile-token-mart
~~~

See [Build profile token mart](../guides/build-profile-token-mart.md) for the
validation command and alternate prior-run option.
