# Shadow in-season participation and minutes updater

Status: implementation complete; production shadow deployment pending approval
GitHub: [Issue #30](https://github.com/EvanZ/nba-lineup-model/issues/30)

## Model contract

The exact promoted 2026-27 preseason availability, conditional-minutes, and
normalized-minute fields are the complete prior. The updater maintains two
separate states:

- probability that a player participates;
- expected regulation minutes conditional on playing.

Played games update both states. Medical absences, coach DNPs, and known
inactive/assignment outcomes update participation but never enter a zero into
conditional role. Unknown DNP outcomes update neither. Official pregame injury
status remains authoritative for a game-specific allocation, and a pending
report is retained as pending rather than converted to an availability claim.

Actual team minutes are normalized to 240 before entering role state, including
overtime games. Ejections and appearances under four minutes receive half
weight. All other extreme outcomes are retained, but each observation is
robustly clipped to at most 12 minutes above or below the preseason conditional
prior. Each game is applied only after every forecast on that date is emitted.

## Historical calibration

A chronological 2025-26 replay using 2024-25 player state as the prior selected
four equivalent prior games for participation and two for conditional minutes.
The grid covered 35,918 matched player-games. Selected participation Brier was
0.17144; selected conditional-minutes MAE was 5.2854 after applying the same
12-minute robust clipping policy used by the live updater.

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

The local replay used the 36 immutable final summaries and box scores through
October 25 (27.5 MiB) plus the 1.5 MiB #29 sufficient-state cache. It produced
1,253 player-game rows with zero game exclusions. An immediate replay reused
all 36 minutes contributions and created none.

Relative to the exact frozen pregame minutes, the updated shadow reduced:

- participation Brier from 0.25682 to 0.17458;
- conditional-minutes MAE from 5.4741 to 5.1208;
- conditional-minutes RMSE from 7.0557 to 6.5480;
- unconditional-minutes MAE from 5.3410 to 5.1195;
- unconditional-minutes RMSE from 7.2012 to 6.8810.

The downstream game sample is too small and remains mixed. Minutes-only margin
MAE improved from 11.3129 to 11.3043, while RMSE worsened from 15.5347 to
15.5975 and Brier worsened from 0.19942 to 0.19985. The #29 ratings-only cell
remains best on this sample (MAE 11.1680, RMSE 15.4342, Brier 0.19592); combining
the rating and minutes shadows produced MAE 11.1714, RMSE 15.4884, and Brier
0.19620. No promotion is recommended from 36 games.

## Remaining issue work

- Deploy the already-tested job and sensor to EC2 after explicit approval.
- Publish the calibration and first live shadow artifacts to S3.
- Accumulate the required live shadow period and write the final promotion
  recommendation before closing Issue #30 or changing public projections.
