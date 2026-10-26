# Slice 6: game-script-aware in-season minutes updater

GitHub: [#31](https://github.com/EvanZ/nba-lineup-model/issues/31)

## Objective

Prevent realized blowout rotations from masquerading as durable player-role
changes. Updated NAIL ratings remain fixed in the shadow evaluation so the only
game-level experiment is frozen versus updated minutes.

## Implemented contract

- A game becomes noncompetitive only in the fourth quarter, for at least 60
  remaining seconds, when the eventual winner reaches a clock-dependent lead
  (`max(10, 3 + 3 * ceiling(minutes remaining))`) and never again leads by
  fewer than eight points.
- Player exposure before that boundary is extrapolated to a 48-minute game.
  This produces a 240-minute competitive-role distribution per team. Overtime
  games remain competitive through their final event.
- Played games and healthy coach/active DNPs update the role state. Official
  medical, inactive/assignment, and unknown outcomes retain the existing
  exclusion semantics.
- Missing or invalid play-by-play fails closed to regulation-equivalent actual
  minutes and is surfaced in report coverage rather than excluding the game.
- Pregame expected blowout minutes use only the projected margin, a frozen
  12-point residual standard deviation, and a 20-point blowout threshold. The
  7.34-minute conditional duration is the 2025-26 mean among games finishing
  at least 20 points apart. The
  transform is applied identically to both A/B arms and iterated twice to
  resolve the small minutes/margin feedback loop.
- Zero-minute injury adjustments remain zero during the garbage-rotation
  blend, so the game-script transform cannot reactivate an unavailable player.

## Historical calibration

The local 2025-26 regular-season marts supplied 1,230 games without an S3
download. Competitive-role reconstruction identified at least one minute of
garbage time in 689 games. It reconstructed 24,688 player-game role targets;
1,955 played rows missing from the curated lineup stints used the documented
regulation-equivalent fallback.

The strictly prior-day replay still selected 0.1 equivalent prior games, with
MAE 5.6581 versus 5.7860 at 2.0 games. This result is recorded, not treated as
promotion evidence: 0.1 remains the lower grid boundary and requires a held-out
season/role-stratified evaluation. Checksums and the full grid are in
`docs/reports/game-script-minutes-calibration-v4.json`.

## Orchestration and artifacts

The minutes-shadow job reads the latest content-addressed CDN play-by-play for
each completed game, incorporates its checksum into contribution lineage, and
publishes reconstruction/fallback coverage. The Dagster sensor snapshot now
includes the CDN coverage index, so a newly archived or revised play-by-play
object can trigger a new idempotent report run.

The HTML report shows only:

1. updated ratings plus frozen minutes; and
2. updated ratings plus updated minutes.

Conditional player accuracy is scored against reconstructed competitive-role
minutes. Unconditional accuracy remains scored against realized minutes after
the pregame-only expected-game-script transform.

## Promotion boundary

This remains shadow-only. No public projection, historical frozen snapshot, or
site payload changes in this slice.
