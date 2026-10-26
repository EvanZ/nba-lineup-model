# Shadow in-season participation and minutes updater

Status: implementation complete; production shadow monitoring active
GitHub: [Issue #30](https://github.com/EvanZ/nba-lineup-model/issues/30)

## Model contract

The exact promoted 2026-27 preseason conditional-minutes and normalized-minute
fields are the complete prior. The official injury report, not a learned
in-season probability, determines game-specific eligibility and status.

Appearances and healthy coach/active DNPs update conditional role. A healthy
DNP is a zero-minute observation because the player was available but omitted
from the rotation. Medical absences, known inactive/assignment outcomes, and
unknown DNPs do not enter that role. The frozen official pregame injury status
is authoritative when it conflicts with ESPN's box-score reason. Participation
state remains a research diagnostic; it is not used to allocate game-day
minutes or as a promotion metric. A pending report stays pending rather than
being converted to an availability claim or learned probability adjustment.

Actual team minutes are normalized to 240 before entering role state, including
overtime games. Ejections and appearances under four minutes receive half
weight. All other extreme outcomes are retained, but each observation is
robustly clipped to at most 12 minutes above or below the preseason conditional
prior. Each game is applied only after every forecast on that date is emitted.

## Historical calibration

A chronological 2025-26 replay now uses the exact 2025-26 forward conditional-
minutes predictions as the preseason-prior analogue and the canonical
availability mart as the outcome source. The corrected loss includes played
games and healthy coach/active DNPs at zero minutes while excluding medical,
inactive/assignment, and unknown outcomes. The grid covered 43,265 matched
player-games and 31,154 available-player observations. It selected **0.1
equivalent prior games** with conditional-minutes MAE 5.8410; the prior strength
of 2.0 produced 5.9591. The replay is strictly prior-day and uses the same
12-minute clipping policy as the live updater. Input checksums and the full grid
are recorded in `docs/reports/in-season-minutes-calibration-v3.json`.

## Artifact and execution contract

- Completed summary and ESPN box-score checksums identify an immutable
  player-game contribution.
- Byte-identical games reuse the contribution; a source revision produces a
  new content-addressed version.
- Dated player state is rebuilt from only the current contribution selected for
  each game, so corrections replace rather than double count evidence.
- The Dagster sensor waits until 9:05 AM Eastern, keys runs by the complete
  finalized-source snapshot, and suppresses an equivalent queued or active run.
- Player-game evaluation, four-cell evaluation, state, contribution, and
  manifest artifacts are designed for immutable publication under
  `s3://nba-gestalt-data/lake/analytical/in_season_minutes_shadow/`.
- Nothing reads the updater from the public schedule-page inference path.

## Bounded 2026-27 validation

The corrected replay used all 43 immutable final summaries and box scores
through October 25 plus the #29 sufficient-state cache. It produced 1,495
player-game rows and 1,172 available-player observations with zero exclusions.
An immediate replay reused all 43 minutes contributions and created none.

Relative to the exact frozen pregame minutes, the updated shadow reduced:

- conditional-minutes MAE from 6.5083 to 5.1302;
- conditional-minutes RMSE from 8.2918 to 6.8580;
- unconditional-minutes MAE from 5.3978 to 4.3226;
- unconditional-minutes RMSE from 7.3010 to 6.2240.

The downstream game sample is too small and remains mixed. Relative to the
frozen/frozen production baseline, updated ratings plus updated minutes improve
margin MAE from 11.6813 to 11.3918 and ATS from 18-25 to 19-24, but Brier moves
from 0.19794 to 0.19610 versus 0.19381 for updated ratings with frozen minutes.
Log loss is 0.57397 versus 0.56835 for that ratings-only cell. This is a genuine
early-season result, not a reason to tune against 43 games. No promotion is
recommended.

## Remaining issue work

- Accumulate the required live shadow period and write the final promotion
  recommendation before closing Issue #30 or changing public projections.
