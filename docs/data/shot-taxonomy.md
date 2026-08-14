---
last_updated: "2026-08-13"
---

# Shot Taxonomy Profiles

The shot-taxonomy mart turns canonical play-by-play attempts into a stable
historical player-season profile. It is a reusable data contract for player
bios, future shot-ecosystem context models, and profile-token neural models;
it is not itself a player-value model.

## Canonical Families

The NBA's detailed `event_subtype` vocabulary changes across sources and eras.
The first release deliberately uses only three robust families:

| Family | Canonical definition |
| --- | --- |
| `rim` | Two-point subtype contains layup, dunk, tip, finger roll, or alley-oop. |
| `non_rim_two` | Every other two-point event, including jumpers, hooks, and floaters. |
| `three` | Every canonical three-point event. |

No shot is discarded because its detailed subtype is unfamiliar: a two-point
event that does not match the rim rule belongs to `non_rim_two`. The output
also retains a season-by-subtype coverage table so the mapping remains
auditable as future source data is added.

## Player-Season Profiles

For player \(i\), family \(k\), and on-court possession count \(n_i\), raw
attempt volume is recorded as \(A_{ik}\). The stabilized attempt rate is:

\[
\widetilde a_{ik} =
100 \frac{A_{ik} + 300\,\overline a_k/100}{n_i + 300},
\]

where \(\overline a_k\) is the completed season's possession-weighted league
rate. This prevents a low-exposure player from having an extreme rate solely
because of a tiny denominator while preserving raw attempts for inspection.

Finishing uses a 75-attempt beta-binomial equivalent shrinkage:

\[
\widetilde p_{ik} =
\frac{M_{ik} + 75\,\overline p_k}{A_{ik}+75}.
\]

The raw field-goal percentage remains nullable when a player has no attempts;
the shrunk percentage is always available. All rates use reconstructed
on-court possessions, never minutes or per-36 scaling.

## Forward Use

The mart is historical. A target-season model uses the immediately preceding
player-season row as its profile. It never reads target-season shot outcomes.
Players without a prior season are explicitly visible through missing prior
rows; a later cold-start profile model will decide how to blend draft/cohort
and replacement profiles.

## Storage

```text
data/analytical/shot_taxonomy/
  _manifest.json
  player_season_shot_profiles.parquet
  season_shot_taxonomy_coverage.parquet
  subtype_shot_taxonomy_coverage.parquet
  league_shot_taxonomy_references.parquet
```

The manifest hashes the validated player-season panel and every season's
curated event manifest, as well as the output files and smoothing contract.

## Current Historical Build

The current build covers 30 regular seasons from 1996-97 through 2025-26,
with **5,773,713** source shot events and **14,562** player-season rows.
Outcome and player attribution coverage are effectively complete:

| Check | Coverage |
| --- | ---: |
| Final made/missed result recorded | 99.9943% |
| Final-result shots with a player ID | 100.0000% |
| Player-attributed shots matched to the player-season panel | 99.9997% |

The 331 source shots without a final result are all from 2019-20 and are
excluded from attempt and make totals rather than silently counted as misses.
Nineteen 1996-97 player-attributed shots do not match a panel row; every other
season has complete player-panel matching.

The broad taxonomy remains stable despite detailed subtype drift:

| Season | Rim shots | Non-rim twos | Threes |
| --- | ---: | ---: | ---: |
| 1996-97 | 65,938 | 78,496 | 38,798 |
| 2015-16 | 62,310 | 85,693 | 58,877 |
| 2025-26 | 71,259 | 56,931 | 90,968 |

For example, modern V3 rows are mostly `Layup`, `DUNK`, `Jump Shot`, and
`Hook`, while older data distinguishes driving, reverse, running, and other
specific variants. The same canonical family mapping handles both forms.

## Build

```bash
uv run nba-build-shot-taxonomy
```

See [Build shot taxonomy](../guides/build-shot-taxonomy.md) for the command
and validation workflow.
